"""Precompute the nearest counted health center sites for every locality.

Run with `python -m scripts.nearest_sites`.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from scripts import checks, config, download, download_log, http_client
from scripts.atomic_write import write_atomic
from scripts.checks import CheckResult
from scripts.sources import fetch_file_from_url

logger = logging.getLogger("scripts.nearest_sites")

CENTER_COLUMNS = (
    "STATEFP", "COUNTYFP", "COUNAME", "STNAME",
    "POPULATION", "LATITUDE", "LONGITUDE",
)
TEXT_ENCODINGS = ("utf-8-sig", "latin-1")
SITE_COLUMNS = (
    "site_id", "site_name", "city", "fips", "latitude", "longitude",
)
OUTPUT_COLUMNS = [
    "fips", "rank", "site_id", "site_name", "site_city",
    "site_fips", "site_locality", "miles",
]
DISPLAY_ACRONYMS = frozenset(
    {"CHC", "CVHS", "HCHC", "MHC", "PATHS", "RAHC", "SDHS", "SCH"}
)
WORD = re.compile(r"[A-Za-z0-9']+")


def centers_checks(df: pd.DataFrame) -> list[CheckResult]:
    """Check that the centers file has the expected columns."""
    return [checks.expect_columns(df, CENTER_COLUMNS)]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in miles between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * config.EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def read_text_csv(path: Path, encodings: Iterable[str]) -> pd.DataFrame:
    """Read a CSV with every cell as text, trying each encoding in turn."""
    for encoding in encodings:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False,
                               encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{Path(path).name}: not readable as "
                     f"{' or '.join(encodings)}")


def require_columns(df: pd.DataFrame, columns: Iterable[str],
                    label: str) -> None:
    """Raise ValueError if any of the named columns is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def to_float(values: pd.Series) -> pd.Series:
    """Convert text to floats; blanks and non-numbers become NaN."""
    return pd.to_numeric(values.str.strip(), errors="coerce").astype(float)


def is_true(values: pd.Series) -> pd.Series:
    """Return True where the text value reads as true."""
    return values.astype(str).str.strip().str.lower() == "true"


def read_centers(path: Path, state_fips: Iterable[str]) -> pd.DataFrame:
    """Return fips, latitude and longitude of the population centers."""
    raw = read_text_csv(path, TEXT_ENCODINGS)
    require_columns(raw, CENTER_COLUMNS, "centers file")
    states = {s.zfill(2) for s in state_fips}
    centers = pd.DataFrame({
        "fips": raw["STATEFP"].str.strip().str.zfill(2)
        + raw["COUNTYFP"].str.strip().str.zfill(3),
        "latitude": to_float(raw["LATITUDE"]),
        "longitude": to_float(raw["LONGITUDE"]),
    })
    centers = centers[centers["fips"].str[:2].isin(states)]
    return centers.reset_index(drop=True)


def read_counted_sites(path: Path) -> pd.DataFrame:
    """Return the sites counted in the supply measure, once each."""
    raw = read_text_csv(path, TEXT_ENCODINGS)
    require_columns(raw, (*SITE_COLUMNS, "in_supply_measure"), "sites file")
    counted = is_true(raw["in_supply_measure"])
    if "counted_once" in raw.columns:
        counted &= is_true(raw["counted_once"])
    sites = raw.loc[counted, list(SITE_COLUMNS)].copy()
    sites["fips"] = sites["fips"].str.strip().str.zfill(5)
    sites["latitude"] = to_float(sites["latitude"])
    sites["longitude"] = to_float(sites["longitude"])
    return sites.reset_index(drop=True)


def title_word(match: re.Match) -> str:
    """Title-case one word unless it is a known acronym."""
    word = match.group(0)
    if word in DISPLAY_ACRONYMS:
        return word
    return word[:1].upper() + word[1:].lower()


def display_name(name: str) -> str:
    """Return an all-uppercase name in title case, keeping acronyms."""
    if not isinstance(name, str) or not name.isupper():
        return name
    return WORD.sub(title_word, name)


def coordinate_problems(df: pd.DataFrame, label: str) -> list[str]:
    """Describe missing or out-of-range coordinates in a table."""
    problems = []
    for column, limit in (("latitude", 90), ("longitude", 180)):
        values = df[column]
        missing = int(values.isna().sum())
        if missing:
            problems.append(f"{label}: {missing} missing {column}")
        bad = int((values.abs() > limit).sum())
        if bad:
            problems.append(f"{label}: {bad} {column} outside "
                            f"[-{limit}, {limit}]")
    return problems


def validate_inputs(centers: pd.DataFrame, sites: pd.DataFrame,
                    locality_fips: Iterable[str]) -> None:
    """Raise ValueError if the centers or sites cannot be used."""
    problems = []
    missing = sorted(set(locality_fips) - set(centers["fips"]))
    if missing:
        problems.append(f"{len(missing)} localities have no population "
                        f"center: {missing}")
    duplicated = sorted(set(centers.loc[centers["fips"].duplicated(),
                                        "fips"]))
    if duplicated:
        problems.append(f"duplicate population centers: {duplicated}")
    if sites.empty:
        problems.append("no counted sites to search")
    problems += coordinate_problems(centers, "centers")
    problems += coordinate_problems(sites, "sites")
    if problems:
        raise ValueError("; ".join(problems))


def find_nearest(centers: pd.DataFrame, sites: pd.DataFrame,
                 k: int) -> pd.DataFrame:
    """Return the k nearest sites to each center, ranked by distance."""
    k = min(k, len(sites))
    tree = BallTree(np.radians(sites[["latitude", "longitude"]].to_numpy()),
                    metric="haversine")
    distance, index = tree.query(
        np.radians(centers[["latitude", "longitude"]].to_numpy()), k=k)
    found = sites.iloc[index.ravel()].reset_index(drop=True)
    result = pd.DataFrame({
        "fips": np.repeat(centers["fips"].to_numpy(), k),
        "site_id": found["site_id"],
        "site_name": found["site_name"],
        "site_city": found["city"],
        "site_fips": found["fips"],
        "miles": distance.ravel() * config.EARTH_RADIUS_MILES,
    })
    result = result.sort_values(["fips", "miles", "site_name"],
                                kind="stable").reset_index(drop=True)
    result["rank"] = result.groupby("fips").cumcount() + 1
    result["miles"] = result["miles"].round(1)
    return result


def validate_output(result: pd.DataFrame, locality_fips: Sequence[str],
                    k: int) -> None:
    """Raise ValueError if the output table breaks any rule."""
    problems = []
    expected_rows = len(locality_fips) * k
    if len(result) != expected_rows:
        problems.append(f"{len(result)} rows, expected {expected_rows}")
    if set(result["fips"]) != set(locality_fips):
        problems.append("fips values do not match the geography file")
    ranks = result.groupby("fips")["rank"].apply(list)
    if not all(r == list(range(1, k + 1)) for r in ranks):
        problems.append(f"ranks are not 1..{k} in every locality")
    nulls = result.isna().sum()
    if nulls.any():
        problems.append(f"null values: {nulls[nulls > 0].to_dict()}")
    far = int((result["miles"] > config.NEAREST_MAX_MILES).sum())
    if far:
        problems.append(f"{far} distances over "
                        f"{config.NEAREST_MAX_MILES} miles")
    if problems:
        raise ValueError("nearest_sites: " + "; ".join(problems))


def build_nearest_sites(centers_path: Path, sites_path: Path,
                        geography_path: Path,
                        k: int = config.NEAREST_K) -> pd.DataFrame:
    """Build the nearest-sites table from the three input files."""
    geography = read_text_csv(geography_path, TEXT_ENCODINGS)
    require_columns(geography, ("fips", "locality_name"), "geography file")
    locality_fips = geography["fips"].str.zfill(5).tolist()
    states = sorted({f[:2] for f in locality_fips})
    site_states = set(states) | set(config.NEAREST_EXTRA_SITE_STATES)
    centers = read_centers(centers_path, states)
    centers = centers[centers["fips"].isin(locality_fips)]
    sites = read_counted_sites(sites_path)
    sites = sites[sites["fips"].str[:2].isin(site_states)]
    sites = sites.reset_index(drop=True)
    sites["site_name"] = sites["site_name"].map(display_name)
    validate_inputs(centers, sites, locality_fips)
    logger.info("Nearest sites: %d localities, %d counted sites, k=%d",
                len(centers), len(sites), min(k, len(sites)))
    result = find_nearest(centers, sites, k)
    names = dict(zip(locality_fips, geography["locality_name"]))
    result["site_locality"] = result["site_fips"].map(names)
    result = result[OUTPUT_COLUMNS]
    validate_output(result, locality_fips, min(k, len(sites)))
    return result


def write_nearest_sites(result: pd.DataFrame, output_path: Path) -> Path:
    """Write the table as UTF-8 CSV and return its path."""
    output_path = Path(output_path)
    data = result.to_csv(index=False, lineterminator="\n")
    write_atomic(output_path, data.encode("utf-8"))
    logger.info("Output %s %d rows", output_path.name, len(result))
    return output_path


def logged_centers(raw_dir: Path, log_path: Path) -> Path | None:
    """Return the newest logged centers file, or None if there is none."""
    entries = [e for e in download_log.read_entries(log_path)
               if e.source == config.CENPOP_SOURCE.name]
    if not entries:
        return None
    entry = entries[-1]
    path = raw_dir / entry.file_name
    if not path.exists():
        return None
    if download_log.sha256_of(path) != entry.sha256:
        raise ValueError(f"{entry.file_name} changed after it was "
                         "downloaded; delete it and run again")
    return path


def download_centers(raw_dir: Path, log_path: Path,
                     today: date | None = None,
                     session: requests.Session | None = None) -> Path:
    """Download the centers file into raw_dir and log it."""
    source = config.CENPOP_SOURCE
    path = raw_dir / source.filename(today or download.today_utc())
    try:
        fetched = fetch_file_from_url.fetch(source, session)
    except http_client.DownloadError as exc:
        raise ValueError(f"{source.name}: {exc}") from exc
    download.save(path, fetched.content)
    outcome = download.record(source, path, fetched.url,
                              download.Paths(raw_dir, log_path),
                              download.Status.DOWNLOADED)
    passed = checks.report(source.name, path,
                           centers_checks(checks.load_table(path)))
    if outcome.status is not download.Status.DOWNLOADED or not passed:
        raise ValueError(f"{path.name} failed its download checks")
    return path


def centers_file(raw_dir: Path = config.RAW_DIR,
                 log_path: Path = config.DOWNLOAD_LOG_PATH) -> Path:
    """Return the logged centers file, downloading it the first time."""
    path = logged_centers(raw_dir, log_path)
    if path is None:
        logger.info("%s: downloading", config.CENPOP_SOURCE.name)
        path = download_centers(raw_dir, log_path)
    return path


def run(raw_dir: Path = config.RAW_DIR,
        log_path: Path = config.DOWNLOAD_LOG_PATH,
        out_dir: Path = config.PROCESSED_DIR,
        k: int = config.NEAREST_K,
        download_missing: bool = True) -> Path | None:
    """Build and write nearest_sites.csv; return None if skipped."""
    if download_missing:
        centers = centers_file(raw_dir, log_path)
    else:
        centers = logged_centers(raw_dir, log_path)
    if centers is None:
        logger.warning("Nearest sites skipped: no centers file in %s/; "
                       "run python -m scripts.nearest_sites once",
                       raw_dir.name)
        return None
    result = build_nearest_sites(
        centers,
        out_dir / config.SITES_PATH.name,
        out_dir / config.DIM_GEOGRAPHY_PATH.name,
        k,
    )
    output_path = out_dir / config.NEAREST_SITES_PATH.name
    return write_nearest_sites(result, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, build the table, and return an exit code."""
    parser = argparse.ArgumentParser(prog="python -m scripts.nearest_sites",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--centers", type=Path,
                        help="centers file (default: newest logged copy, "
                             "downloaded if missing)")
    parser.add_argument("--sites", type=Path, default=config.SITES_PATH)
    parser.add_argument("--geography", type=Path,
                        default=config.DIM_GEOGRAPHY_PATH)
    parser.add_argument("--output", type=Path,
                        default=config.NEAREST_SITES_PATH)
    parser.add_argument("--k", type=int, default=config.NEAREST_K)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show DEBUG messages")
    args = parser.parse_args(argv)
    download.configure_logging(args.verbose)
    if args.k < 1:
        parser.error("--k must be at least 1")
    try:
        result = build_nearest_sites(args.centers or centers_file(),
                                     args.sites, args.geography, args.k)
    except ValueError as exc:
        logger.error("Nearest sites stopped: %s", exc)
        return 1
    write_nearest_sites(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

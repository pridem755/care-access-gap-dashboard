"""Draw the locality background image and bounds for the site dot map.

Run with `python -m scripts.site_map_background`.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from scripts import config, download, download_log
from scripts.atomic_write import write_atomic
from scripts.nearest_sites import (
    TEXT_ENCODINGS, read_text_csv, require_columns,
)

logger = logging.getLogger("scripts.site_map_background")

BOUNDARY_COLUMNS = ("STATEFP", "COUNTYFP")
GEOGRAPHIC_CRS = "EPSG:4326"
POINTS_PER_INCH = 72
BOUNDS_COLUMNS = [
    "region", "lon_min", "lon_max", "lat_min", "lat_max",
    "aspect_ratio", "width_px", "height_px", "image_file",
]
OK_STATUSES = (download.Status.DOWNLOADED, download.Status.SKIPPED)


def read_localities(boundary_path: Path,
                    state_fips: Iterable[str]) -> gpd.GeoDataFrame:
    """Return fips and geometry for the given states in EPSG:4326."""
    raw = gpd.read_file(boundary_path, columns=list(BOUNDARY_COLUMNS))
    require_columns(raw, BOUNDARY_COLUMNS, "boundary file")
    if raw.crs is None:
        raise ValueError(f"{Path(boundary_path).name}: no coordinate "
                         "reference system")
    states = sorted({str(s).strip().zfill(2) for s in state_fips})
    fips = (raw["STATEFP"].astype(str).str.strip().str.zfill(2)
            + raw["COUNTYFP"].astype(str).str.strip().str.zfill(3))
    shapes = gpd.GeoDataFrame({"fips": fips}, geometry=raw.geometry,
                              crs=raw.crs)
    shapes = shapes[shapes["fips"].str[:2].isin(states)]
    if shapes.empty:
        raise ValueError(f"no shapes found for states {states}")
    if not shapes.crs.equals(GEOGRAPHIC_CRS):
        shapes = shapes.to_crs(GEOGRAPHIC_CRS)
    return shapes.sort_values("fips").reset_index(drop=True)


def check_localities(shapes: gpd.GeoDataFrame,
                     locality_fips: Iterable[str]) -> None:
    """Raise ValueError if a locality in the states has no shape."""
    states = set(shapes["fips"].str[:2])
    expected = {f for f in locality_fips if f[:2] in states}
    missing = sorted(expected - set(shapes["fips"]))
    if missing:
        raise ValueError(f"{len(missing)} localities have no shape: "
                         f"{missing}")


def validate_bounds(bounds: dict[str, float]) -> None:
    """Raise ValueError if the bounds or aspect ratio are unusable."""
    problems = []
    for key, limit in (("lon", 180), ("lat", 90)):
        for side in ("min", "max"):
            value = bounds[f"{key}_{side}"]
            if math.isnan(value):
                problems.append(f"{key}_{side} is missing")
            elif abs(value) > limit:
                problems.append(f"{key}_{side} {value} outside "
                                f"[-{limit}, {limit}]")
    low, high = config.MAP_ASPECT_RANGE
    aspect = bounds["aspect_ratio"]
    if not problems and not low <= aspect <= high:
        problems.append(f"aspect_ratio {aspect} outside [{low}, {high}]")
    if problems:
        raise ValueError("map bounds: " + "; ".join(problems))


def compute_bounds(shapes: gpd.GeoDataFrame,
                   pad_fraction: float) -> dict[str, float]:
    """Return the padded limits and undistorted aspect ratio."""
    x_min, y_min, x_max, y_max = (float(v) for v in shapes.total_bounds)
    x_pad = (x_max - x_min) * pad_fraction
    y_pad = (y_max - y_min) * pad_fraction
    bounds = {
        "lon_min": round(x_min - x_pad, 6),
        "lon_max": round(x_max + x_pad, 6),
        "lat_min": round(y_min - y_pad, 6),
        "lat_max": round(y_max + y_pad, 6),
    }
    lat_span = bounds["lat_max"] - bounds["lat_min"]
    middle = math.radians((bounds["lat_max"] + bounds["lat_min"]) / 2)
    lon_span = (bounds["lon_max"] - bounds["lon_min"]) * math.cos(middle)
    bounds["aspect_ratio"] = (round(lon_span / lat_span, 6)
                              if lat_span > 0 else math.nan)
    validate_bounds(bounds)
    return bounds


def figure_inches(pixels: int, dpi: int) -> float:
    """Return the size in inches that renders to exactly `pixels`."""
    size = pixels / dpi
    while int(size * dpi) < pixels:
        size = math.nextafter(size, math.inf)
    return size


def render_background(shapes: gpd.GeoDataFrame, bounds: dict[str, float],
                      output_path: Path,
                      width_px: int = config.MAP_IMAGE_WIDTH_PX
                      ) -> tuple[int, int]:
    """Write the transparent PNG and return its width and height."""
    dpi = config.MAP_IMAGE_DPI
    height_px = round(width_px / bounds["aspect_ratio"])
    figure = Figure(figsize=(figure_inches(width_px, dpi),
                             figure_inches(height_px, dpi)), dpi=dpi)
    axes = figure.add_axes([0, 0, 1, 1])
    shapes.plot(ax=axes, color=config.MAP_FILL_COLOR,
                edgecolor=config.MAP_EDGE_COLOR,
                linewidth=config.MAP_EDGE_WIDTH_PX * POINTS_PER_INCH / dpi)
    axes.set_xlim(bounds["lon_min"], bounds["lon_max"])
    axes.set_ylim(bounds["lat_min"], bounds["lat_max"])
    axes.set_aspect("auto")
    axes.margins(0)
    axes.set_axis_off()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, transparent=True)
    size = Image.open(io.BytesIO(buffer.getvalue())).size
    if size != (width_px, height_px):
        raise ValueError(f"image is {size[0]} x {size[1]} px, expected "
                         f"{width_px} x {height_px}")
    write_atomic(Path(output_path), buffer.getvalue())
    return width_px, height_px


def write_bounds(row: dict, output_path: Path) -> Path:
    """Write the one-row bounds table as UTF-8 CSV and return its path."""
    output_path = Path(output_path)
    data = pd.DataFrame([row], columns=BOUNDS_COLUMNS).to_csv(
        index=False, lineterminator="\n", float_format="%.6f")
    write_atomic(output_path, data.encode("utf-8"))
    return output_path


def build_site_map_background(
        boundary_path: Path,
        geography_path: Path = config.DIM_GEOGRAPHY_PATH,
        image_path: Path = config.SITE_MAP_BACKGROUND_PATH,
        bounds_path: Path = config.MAP_BOUNDS_PATH,
        state_fips: Iterable[str] = config.STATES,
        region: str = config.MAP_REGION,
        width_px: int = config.MAP_IMAGE_WIDTH_PX,
        pad_fraction: float = config.MAP_PAD_FRACTION) -> dict:
    """Write the background PNG and map_bounds.csv; return the row."""
    geography = read_text_csv(geography_path, TEXT_ENCODINGS)
    require_columns(geography, ("fips",), "geography file")
    shapes = read_localities(boundary_path, state_fips)
    check_localities(shapes, geography["fips"].str.zfill(5))
    bounds = compute_bounds(shapes, pad_fraction)
    width, height = render_background(shapes, bounds, image_path, width_px)
    row = {"region": region, **bounds, "width_px": width,
           "height_px": height, "image_file": Path(image_path).name}
    write_bounds(row, bounds_path)
    logger.info("Site map background: %d shapes, lon %.6f to %.6f, "
                "lat %.6f to %.6f, %d x %d px", len(shapes),
                bounds["lon_min"], bounds["lon_max"], bounds["lat_min"],
                bounds["lat_max"], width, height)
    logger.info("Output %s and %s", Path(image_path).name,
                Path(bounds_path).name)
    return row


def logged_boundaries(raw_dir: Path, log_path: Path) -> Path | None:
    """Return the newest logged boundary file, or None if there is none."""
    entries = [e for e in download_log.read_entries(log_path)
               if e.source == config.COUNTY_BOUNDARIES_SOURCE]
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


def boundaries_file(raw_dir: Path = config.RAW_DIR,
                    log_path: Path = config.DOWNLOAD_LOG_PATH) -> Path:
    """Return the logged boundary file, downloading it the first time."""
    path = logged_boundaries(raw_dir, log_path)
    if path is not None:
        return path
    source = config.SOURCES[config.COUNTY_BOUNDARIES_SOURCE]
    outcome = download.download_source(
        source, False, download.Paths(raw_dir, log_path),
        download.today_utc())
    path = logged_boundaries(raw_dir, log_path)
    if outcome.status not in OK_STATUSES or path is None:
        raise ValueError(f"{source.name}: download {outcome.status.value}"
                         f" {outcome.detail}".rstrip())
    return path


def fetch_command() -> str:
    """Return the command that downloads the boundary file."""
    return ("python -m scripts.download --source "
            f"{config.COUNTY_BOUNDARIES_SOURCE}")


def run(raw_dir: Path = config.RAW_DIR,
        log_path: Path = config.DOWNLOAD_LOG_PATH,
        out_dir: Path = config.PROCESSED_DIR,
        state_fips: Iterable[str] = config.STATES,
        download_missing: bool = True) -> dict | None:
    """Build the background image and bounds; return None if skipped."""
    image = out_dir / config.SITE_MAP_BACKGROUND_PATH.name
    bounds = out_dir / config.MAP_BOUNDS_PATH.name
    if download_missing:
        boundary_path = boundaries_file(raw_dir, log_path)
    else:
        boundary_path = logged_boundaries(raw_dir, log_path)
    if boundary_path is None:
        kept = [p.name for p in (image, bounds) if p.exists()]
        status = (f"kept the older {' and '.join(kept)}" if kept
                  else "no older image or bounds file exists")
        logger.warning("Site map background skipped: no %s file (%s) in "
                       "%s/; run %s once; %s",
                       config.COUNTY_BOUNDARIES_SOURCE,
                       Path(config.BOUNDARY_URL).name, raw_dir.name,
                       fetch_command(), status)
        return None
    return build_site_map_background(
        boundary_path, out_dir / config.DIM_GEOGRAPHY_PATH.name,
        image, bounds, state_fips)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, draw the image, and return an exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.site_map_background",
        description=__doc__.splitlines()[0])
    parser.add_argument("--boundaries", type=Path,
                        help="boundary ZIP (default: newest logged copy, "
                             "downloaded if missing)")
    parser.add_argument("--states", nargs="+", default=list(config.STATES),
                        metavar="FIPS", help="two-digit state FIPS codes")
    parser.add_argument("--geography", type=Path,
                        default=config.DIM_GEOGRAPHY_PATH)
    parser.add_argument("--image", type=Path,
                        default=config.SITE_MAP_BACKGROUND_PATH)
    parser.add_argument("--bounds", type=Path,
                        default=config.MAP_BOUNDS_PATH)
    parser.add_argument("--region", default=config.MAP_REGION)
    parser.add_argument("--width", type=int,
                        default=config.MAP_IMAGE_WIDTH_PX)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show DEBUG messages")
    args = parser.parse_args(argv)
    download.configure_logging(args.verbose)
    if args.width < 1:
        parser.error("--width must be at least 1")
    try:
        build_site_map_background(
            args.boundaries or boundaries_file(), args.geography,
            args.image, args.bounds, args.states, args.region, args.width)
    except ValueError as exc:
        logger.error("Site map background stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

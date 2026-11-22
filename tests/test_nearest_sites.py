"""Tests for scripts.nearest_sites."""
import math

import pandas as pd
import pytest

from scripts import config, nearest_sites
from scripts.sources import Fetched


def centers_frame(rows):
    """Build a centers table from (fips, latitude, longitude) rows."""
    return pd.DataFrame(rows, columns=["fips", "latitude", "longitude"])


def sites_frame(rows):
    """Build a sites table from (site_id, fips, latitude, longitude) rows."""
    sites = pd.DataFrame(
        rows, columns=["site_id", "fips", "latitude", "longitude"])
    sites["site_name"] = "Site " + sites["site_id"]
    sites["city"] = "Town"
    return sites


CENTERS = centers_frame([
    ("51001", 37.0, -78.0),
    ("51003", 38.0, -77.0),
])
SITES = sites_frame([
    ("A", "51001", 37.01, -78.01),
    ("B", "51003", 38.02, -77.02),
    ("C", "51003", 37.50, -77.50),
    ("D", "51001", 36.80, -78.30),
])


def test_haversine_one_degree_of_latitude():
    """One degree along a meridian is about 69.1 miles."""
    miles = nearest_sites.haversine_miles(37.0, -78.0, 38.0, -78.0)
    expected = math.pi / 180 * config.EARTH_RADIUS_MILES
    assert miles == pytest.approx(expected, abs=0.5)
    assert miles == pytest.approx(69.1, abs=0.5)


def test_haversine_quarter_of_the_equator():
    """A quarter turn of the equator is about 6,218 miles."""
    miles = nearest_sites.haversine_miles(0.0, 0.0, 0.0, 90.0)
    assert miles == pytest.approx(6218.5, abs=0.5)


def test_tree_distances_match_haversine():
    """BallTree miles equal the haversine formula."""
    result = nearest_sites.find_nearest(CENTERS, SITES, 1)
    first = result[result["fips"] == "51003"].iloc[0]
    expected = nearest_sites.haversine_miles(38.0, -77.0, 38.02, -77.02)
    assert first["miles"] == pytest.approx(expected, abs=0.05)


def test_site_inside_locality_is_rank_one():
    """A locality's own nearby site is ranked first."""
    result = nearest_sites.find_nearest(CENTERS, SITES, 3)
    top = result[result["rank"] == 1].set_index("fips")["site_id"]
    assert top.to_dict() == {"51001": "A", "51003": "B"}


def test_ranks_have_no_gaps_and_miles_never_decrease():
    """Ranks run 1..k and miles are non-decreasing within each fips."""
    k = 3
    result = nearest_sites.find_nearest(CENTERS, SITES, k)
    assert len(result) == len(CENTERS) * k
    for _, group in result.groupby("fips"):
        assert group["rank"].tolist() == list(range(1, k + 1))
        assert group["miles"].is_monotonic_increasing


def test_k_larger_than_site_count_uses_every_site():
    """When k exceeds the number of sites, every site is returned."""
    result = nearest_sites.find_nearest(CENTERS, SITES, 10)
    assert len(result) == len(CENTERS) * len(SITES)


def test_missing_coordinates_raise():
    """A site without a latitude stops the step."""
    sites = SITES.copy()
    sites.loc[0, "latitude"] = float("nan")
    with pytest.raises(ValueError, match="missing latitude"):
        nearest_sites.validate_inputs(CENTERS, sites, ["51001", "51003"])


def test_out_of_range_coordinates_raise():
    """A latitude outside [-90, 90] stops the step."""
    centers = CENTERS.copy()
    centers.loc[0, "latitude"] = 95.0
    with pytest.raises(ValueError, match="outside"):
        nearest_sites.validate_inputs(centers, SITES, ["51001", "51003"])


def test_locality_without_center_raises():
    """A locality missing from the centers file is named in the error."""
    with pytest.raises(ValueError, match="51005"):
        nearest_sites.validate_inputs(
            CENTERS, SITES, ["51001", "51003", "51005"])


def test_output_over_max_miles_raises():
    """A distance over the configured limit stops the step."""
    result = nearest_sites.find_nearest(CENTERS, SITES, 1)
    result["site_locality"] = "Somewhere"
    result.loc[0, "miles"] = config.NEAREST_MAX_MILES + 1
    with pytest.raises(ValueError, match="miles"):
        nearest_sites.validate_output(result, ["51001", "51003"], 1)


def test_counted_once_false_is_excluded(tmp_path):
    """Only rows in the supply measure and counted once are kept."""
    path = tmp_path / "sites.csv"
    pd.DataFrame({
        "site_id": ["A", "B", "C"],
        "site_name": ["One", "Two", "Three"],
        "city": ["X", "Y", "Z"],
        "fips": ["51001", "51001", "51003"],
        "latitude": ["37.0", "37.1", "38.0"],
        "longitude": ["-78.0", "-78.1", "-77.0"],
        "in_supply_measure": ["True", "True", "False"],
        "counted_once": ["True", "False", "True"],
    }).to_csv(path, index=False)
    sites = nearest_sites.read_counted_sites(path)
    assert sites["site_id"].tolist() == ["A"]


def test_read_centers_builds_fips_and_filters_states(tmp_path):
    """FIPS is built from padded state and county codes."""
    path = tmp_path / "centers.txt"
    path.write_text(
        "STATEFP,COUNTYFP,COUNAME,STNAME,POPULATION,LATITUDE,LONGITUDE\n"
        "51,1,Accomack,Virginia,1,+37.7,-075.6\n"
        "24,001,Allegany,Maryland,1,+39.6,-078.7\n",
        encoding="utf-8-sig",
    )
    centers = nearest_sites.read_centers(path, ["51"])
    assert centers.to_dict("records") == [
        {"fips": "51001", "latitude": 37.7, "longitude": -75.6}]


def test_build_writes_expected_table(tmp_path):
    """The full build returns every locality times k in column order."""
    centers = tmp_path / "centers.txt"
    centers.write_text(
        "STATEFP,COUNTYFP,COUNAME,STNAME,POPULATION,LATITUDE,LONGITUDE\n"
        "51,001,A,Virginia,1,37.0,-78.0\n"
        "51,003,B,Virginia,1,38.0,-77.0\n",
        encoding="latin-1",
    )
    sites = tmp_path / "sites.csv"
    frame = SITES.assign(in_supply_measure=True, counted_once=True)
    frame.to_csv(sites, index=False)
    geography = tmp_path / "geo.csv"
    pd.DataFrame({"fips": ["51001", "51003"],
                  "locality_name": ["Alpha", "Beta"]}).to_csv(
        geography, index=False)
    result = nearest_sites.build_nearest_sites(centers, sites, geography, 2)
    assert list(result.columns) == nearest_sites.OUTPUT_COLUMNS
    assert len(result) == 4
    assert result.notna().all().all()
    out = nearest_sites.write_nearest_sites(result, tmp_path / "x" / "o.csv")
    assert out.exists()


@pytest.mark.parametrize("name, expected", [
    ("AMELIA HEALTHCARE CENTER", "Amelia Healthcare Center"),
    ("CVHS Farmville", "CVHS Farmville"),
    ("SDHS FAMILY DENTISTRY", "SDHS Family Dentistry"),
    ("ST. MARY'S CLINIC", "St. Mary's Clinic"),
])
def test_display_name(name, expected):
    """Uppercase names become title case; acronyms and mixed case stay."""
    assert nearest_sites.display_name(name) == expected


def test_centers_file_downloads_once_then_uses_log(tmp_path, monkeypatch):
    """The centers file is fetched once, logged, then reused."""
    calls = []
    body = (b"STATEFP,COUNTYFP,COUNAME,STNAME,POPULATION,LATITUDE,"
            b"LONGITUDE\n51,001,A,Virginia,1,37.0,-78.0\n")

    def fake_fetch(source, session=None):
        """Return a small centers file instead of downloading."""
        calls.append(source.name)
        return Fetched(content=body, url=source.url)

    monkeypatch.setattr(nearest_sites.fetch_file_from_url, "fetch",
                        fake_fetch)
    log = tmp_path / "download_log.csv"
    first = nearest_sites.centers_file(tmp_path, log)
    second = nearest_sites.centers_file(tmp_path, log)
    assert first == second and first.read_bytes() == body
    assert calls == [config.CENPOP_SOURCE.name]


def test_run_skips_without_cached_centers(tmp_path, monkeypatch):
    """With downloads off and nothing logged, nothing is fetched or written."""
    def no_fetch(source, session=None):
        """Fail if a download is attempted."""
        raise AssertionError("download attempted")

    monkeypatch.setattr(nearest_sites.fetch_file_from_url, "fetch",
                        no_fetch)
    log = tmp_path / "download_log.csv"
    result = nearest_sites.run(tmp_path, log, tmp_path / "out",
                               download_missing=False)
    assert result is None
    assert not (tmp_path / "out").exists()

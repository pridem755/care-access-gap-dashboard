"""Tests for scripts.site_map_background."""
import logging
import math
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import box

from scripts import config, site_map_background
from scripts.sources import fetch_file_from_url

SQUARES = gpd.GeoDataFrame(
    {"fips": ["51001", "51003", "51005"]},
    geometry=[box(-80, 37, -79, 38), box(-79, 37, -78, 38),
              box(-78, 38, -77, 39)],
    crs="EPSG:4326",
)
LON_MIN, LON_MAX = -80 - 0.06, -77 + 0.06
LAT_MIN, LAT_MAX = 37 - 0.04, 39 + 0.04


def write_boundary_zip(path, frame):
    """Write a frame as a zipped shapefile with STATEFP and COUNTYFP."""
    folder = path.parent / "shp"
    folder.mkdir()
    frame.to_file(folder / "cb.shp")
    with zipfile.ZipFile(path, "w") as archive:
        for part in folder.iterdir():
            archive.write(part, part.name)
    return path


def boundary_frame(crs="EPSG:4326"):
    """Return the squares plus one North Carolina square as raw fields."""
    frame = gpd.GeoDataFrame(
        {"STATEFP": ["51", "51", "51", "37"],
         "COUNTYFP": ["1", "003", "005", "001"]},
        geometry=[*SQUARES.geometry, box(-80, 35, -79, 36)],
        crs="EPSG:4326",
    )
    return frame.to_crs(crs)


def test_bounds_include_padding():
    """Each side is padded by the pad fraction of the span."""
    bounds = site_map_background.compute_bounds(SQUARES, 0.02)
    assert bounds["lon_min"] == pytest.approx(LON_MIN)
    assert bounds["lon_max"] == pytest.approx(LON_MAX)
    assert bounds["lat_min"] == pytest.approx(LAT_MIN)
    assert bounds["lat_max"] == pytest.approx(LAT_MAX)


def test_aspect_ratio_uses_cosine_of_middle_latitude():
    """Width is scaled by cos(middle latitude) before dividing."""
    bounds = site_map_background.compute_bounds(SQUARES, 0.02)
    middle = math.radians((LAT_MIN + LAT_MAX) / 2)
    expected = (LON_MAX - LON_MIN) * math.cos(middle) / (LAT_MAX - LAT_MIN)
    assert bounds["aspect_ratio"] == pytest.approx(expected, abs=1e-6)


def test_png_has_exact_size(tmp_path):
    """The PNG is width_px by round(width_px / aspect_ratio)."""
    bounds = site_map_background.compute_bounds(SQUARES, 0.02)
    path = tmp_path / "bg.png"
    size = site_map_background.render_background(SQUARES, bounds, path, 400)
    expected = (400, round(400 / bounds["aspect_ratio"]))
    assert size == expected
    with Image.open(path) as image:
        assert image.size == expected
        assert image.mode == "RGBA"


def test_known_corner_lands_on_expected_pixel(tmp_path):
    """The top-right corner of a square maps linearly to its pixel.

    The outer edge of the border line sits half its width past the corner.
    """
    bounds = site_map_background.compute_bounds(SQUARES, 0.02)
    path = tmp_path / "bg.png"
    width, height = site_map_background.render_background(
        SQUARES, bounds, path, 800)
    with Image.open(path) as image:
        alpha = np.asarray(image)[:, :, 3]
    x = (-77 - bounds["lon_min"]) / (bounds["lon_max"]
                                     - bounds["lon_min"]) * width
    y = (bounds["lat_max"] - 39) / (bounds["lat_max"]
                                    - bounds["lat_min"]) * height
    row = int(y) + 20
    column = int(x) - 20
    right_edge = np.nonzero(alpha[row] > 0)[0].max() + 1
    top_edge = np.nonzero(alpha[:, column] > 0)[0].min()
    tolerance = 2 + config.MAP_EDGE_WIDTH_PX / 2
    assert right_edge == pytest.approx(x, abs=tolerance)
    assert top_edge == pytest.approx(y, abs=tolerance)
    assert alpha[0, 0] == 0


def test_read_localities_builds_fips_and_filters(tmp_path):
    """FIPS is padded and only the requested states are kept."""
    path = write_boundary_zip(tmp_path / "cb.zip", boundary_frame())
    shapes = site_map_background.read_localities(path, ["51"])
    assert shapes["fips"].tolist() == ["51001", "51003", "51005"]
    assert shapes.crs.equals("EPSG:4326")
    both = site_map_background.read_localities(path, ["51", "37"])
    assert len(both) == 4


def test_read_localities_reprojects(tmp_path):
    """Shapes in another CRS come back in longitude and latitude."""
    path = write_boundary_zip(tmp_path / "cb.zip",
                              boundary_frame("EPSG:3857"))
    shapes = site_map_background.read_localities(path, ["51"])
    assert shapes.crs.equals("EPSG:4326")
    np.testing.assert_allclose(shapes.total_bounds, [-80, 37, -77, 39],
                               atol=1e-6)


def test_empty_state_filter_raises(tmp_path):
    """A state with no shapes stops the step."""
    path = write_boundary_zip(tmp_path / "cb.zip", boundary_frame())
    with pytest.raises(ValueError, match="no shapes"):
        site_map_background.read_localities(path, ["99"])


def test_missing_locality_shape_is_named():
    """A geography fips without a shape is listed in the error."""
    with pytest.raises(ValueError, match="51007"):
        site_map_background.check_localities(
            SQUARES, ["51001", "51003", "51005", "51007"])


@pytest.mark.parametrize("key, value, message", [
    ("lon_min", float("nan"), "missing"),
    ("lat_max", 95.0, "outside"),
    ("aspect_ratio", 6.0, "aspect_ratio"),
])
def test_bad_bounds_raise(key, value, message):
    """NaN, out-of-range limits and odd aspect ratios stop the step."""
    bounds = {"lon_min": -80.0, "lon_max": -77.0, "lat_min": 37.0,
              "lat_max": 39.0, "aspect_ratio": 1.2, key: value}
    with pytest.raises(ValueError, match=message):
        site_map_background.validate_bounds(bounds)


def test_build_writes_image_and_bounds(tmp_path):
    """The full build writes the PNG and a one-row bounds table."""
    boundary = write_boundary_zip(tmp_path / "cb.zip", boundary_frame())
    geography = tmp_path / "geo.csv"
    pd.DataFrame({"fips": ["51001", "51003", "51005"]}).to_csv(
        geography, index=False)
    image = tmp_path / "out" / "bg.png"
    bounds = tmp_path / "out" / "bounds.csv"
    row = site_map_background.build_site_map_background(
        boundary, geography, image, bounds, ["51"], "Test", 300)
    table = pd.read_csv(bounds, dtype=str)
    assert list(table.columns) == site_map_background.BOUNDS_COLUMNS
    assert len(table) == 1
    assert table.loc[0, "lon_min"] == f"{LON_MIN:.6f}"
    assert table.loc[0, "image_file"] == "bg.png"
    assert row["width_px"] == 300 and image.exists()


def test_missing_boundary_file_warns_without_download(tmp_path, monkeypatch,
                                                      caplog):
    """With no cached file, nothing is fetched and the warning says why."""
    def no_fetch(source, session=None):
        """Fail if a download is attempted."""
        raise AssertionError("download attempted")

    monkeypatch.setattr(fetch_file_from_url, "fetch", no_fetch)
    monkeypatch.setitem(site_map_background.download.FETCHERS,
                        config.COUNTY_BOUNDARIES_SOURCE, no_fetch)
    out = tmp_path / "out"
    out.mkdir()
    (out / config.SITE_MAP_BACKGROUND_PATH.name).write_bytes(b"old")
    with caplog.at_level(logging.WARNING):
        result = site_map_background.run(
            tmp_path, tmp_path / "download_log.csv", out,
            download_missing=False)
    assert result is None
    assert "scripts.download --source boundaries" in caplog.text
    assert "kept the older site_map_background.png" in caplog.text
    assert not (out / config.MAP_BOUNDS_PATH.name).exists()

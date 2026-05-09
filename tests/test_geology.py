import math
import sys
from pathlib import Path

# Ensure local src/ is on sys.path so tests can import the package during CI/local runs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from wro_metro import planning as pl


def transform_linestring_wgs84_to_local(linestring_wgs84):
    gdf = gpd.GeoDataFrame(geometry=[linestring_wgs84], crs=pl.WGS84)
    gdf = gdf.to_crs(pl.LOCAL_CRS)
    return gdf.geometry.iloc[0]


def test_geology_excess_km_simple_full_within_zone():
    """A short line fully inside the demo central geology zone should produce
    geology_excess_km = length_km * (cost_factor - 1.0) (approximately).
    """
    geology = pl.demo_geology()  # already returned in LOCAL_CRS

    # create a short line in WGS84 that sits inside the central demo box
    line_wgs84 = LineString([(17.00, 51.05), (17.04, 51.07)])
    line_local = transform_linestring_wgs84_to_local(line_wgs84)

    # central box in demo_geology has cost_factor 1.35
    expected_factor = 1.35 - 1.0
    excess_km = pl.geology_excess_km_for_line(line_local, geology, sample_count=16)

    line_km = line_local.length / 1000.0
    assert line_km > 0
    assert excess_km == pytest.approx(line_km * expected_factor, rel=1e-2)


def test_route_score_applies_geology_penalty_per_km():
    """Score should subtract geology_excess_km * geology_penalty_per_km.

    Build a minimal demand (single point colocated with a station) so the
    served_weight is known and geology penalty effect is isolated.
    """
    geology = pl.demo_geology()

    # short line endpoints in WGS84, transform to local CRS
    line_wgs84 = LineString([(17.00, 51.05), (17.04, 51.07)])
    line_local = transform_linestring_wgs84_to_local(line_wgs84)
    start = Point(line_local.coords[0])
    end = Point(line_local.coords[-1])
    mid = line_local.interpolate(line_local.length / 2.0)

    # single demand point at midpoint with known population
    demand = gpd.GeoDataFrame({"population": [1000.0]}, geometry=[mid], crs=pl.LOCAL_CRS)

    # include a station at the demand point to ensure full coverage
    points = [mid, start, end]

    penalty = 12_345.0
    config = pl.MetroConfig(geology_penalty_per_km=penalty)

    metrics = pl.route_score_from_points(points, demand, None, geology, config, weight_col="population")

    geology_excess = metrics["geology_excess_km"]
    served_weight = metrics["served_weight"]
    score = metrics["score"]

    assert served_weight == pytest.approx(1000.0, rel=1e-6)
    expected_score = served_weight - geology_excess * penalty
    assert score == pytest.approx(expected_score, rel=1e-6)

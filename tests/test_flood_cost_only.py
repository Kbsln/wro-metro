import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from wro_metro import planning as pl


def test_cost_only_flood_zone_does_not_forbid_station_candidate_score():
    crs = pl.LOCAL_CRS
    line = LineString([(0, 0), (1_000, 0)])
    demand = gpd.GeoDataFrame({"population": [100.0]}, geometry=[Point(500, 0)], crs=crs)
    forbidden_geom = box(450, -100, 550, 100)
    distances = np.array([500.0])

    cost_only_scores = pl.station_candidate_scores_along_line(
        line,
        distances,
        demand,
        forbidden_geom,
        geology=None,
        config=pl.MetroConfig(flood_zones_are_cost_only=True),
    )
    flood_safe_scores = pl.station_candidate_scores_along_line(
        line,
        distances,
        demand,
        forbidden_geom,
        geology=None,
        config=pl.MetroConfig(flood_zones_are_cost_only=False),
    )

    assert cost_only_scores[0] == pytest.approx(100.0)
    assert flood_safe_scores[0] < -1_000_000.0


def test_cost_only_flood_zone_keeps_anchor_in_place():
    crs = pl.LOCAL_CRS
    demand_point = Point(1_000, 500)
    demand = gpd.GeoDataFrame(
        {"name": ["inside_mzp"], "population": [100.0]},
        geometry=[demand_point],
        crs=crs,
    )
    forbidden = gpd.GeoDataFrame({"risk": ["flood"]}, geometry=[box(0, 0, 2_000, 1_000)], crs=crs)
    config = pl.MetroConfig(
        flood_zones_are_cost_only=True,
        min_regional_anchor_candidates=0,
        min_directional_anchor_candidates_per_sector=0,
        min_central_anchor_candidates=0,
    )

    candidates = pl.candidate_station_sites(
        demand,
        forbidden=forbidden,
        config=config,
        max_candidates=1,
        min_separation_m=0.0,
    )

    assert len(candidates) == 1
    assert candidates["anchor_relocated_m"].iloc[0] == pytest.approx(0.0)
    assert candidates["in_flood_zone"].iloc[0]
    assert candidates.geometry.iloc[0].equals(demand_point)

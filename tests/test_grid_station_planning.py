import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
from shapely.geometry import LineString, Point, box

from wro_metro import planning as pl


def test_accessibility_to_corridors_collects_neighbouring_demand():
    crs = pl.LOCAL_CRS
    demand = gpd.GeoDataFrame(
        {"population": [100.0, 100.0, 100.0]},
        geometry=[Point(0, 0), Point(0, 1_000), Point(0, 3_000)],
        crs=crs,
    )
    line = gpd.GeoDataFrame(
        {"line_id": [1]},
        geometry=[LineString([(-1_000, 0), (1_000, 0)])],
        crs=crs,
    )

    coverage = pl.accessibility_to_corridors(
        demand,
        line,
        corridor_radius_m=2_000.0,
    )

    assert coverage["corridor_coverage_score"].tolist() == [1.0, 0.5, 0.0]
    assert coverage["corridor_served_weight"].tolist() == [100.0, 50.0, 0.0]


def test_grid_station_candidates_include_required_centre():
    crs = pl.LOCAL_CRS
    demand = gpd.GeoDataFrame(
        {"population": [100.0, 100.0]},
        geometry=[Point(-1_000, 0), Point(1_000, 0)],
        crs=crs,
    )
    city = gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(-2_000, -1_000, 2_000, 1_000)], crs=crs)
    centres = gpd.GeoDataFrame(
        {"name": ["Stare Miasto"], "role": ["required_city_centre"]},
        geometry=[Point(0, 0)],
        crs=crs,
    )
    config = pl.MetroConfig(grid_station_step_m=1_000.0, walk_radius_m=800.0)

    candidates = pl.grid_station_candidates(
        demand,
        city_boundary=city,
        centres=centres,
        config=config,
    )

    assert not candidates.empty
    assert candidates["required"].astype(bool).sum() == 1
    assert set(candidates["source"]).issuperset({"required", "grid_station"})


def test_grid_station_candidates_keep_connectivity_support_points():
    crs = pl.LOCAL_CRS
    centre = Point(0, 0)
    demand_points = [Point(1_000, 0), Point(2_000, 0), Point(3_000, 0), Point(5_000, 0)]
    demand = gpd.GeoDataFrame(
        {"population": [10.0, 10.0, 10.0, 1_000.0]},
        geometry=demand_points,
        crs=crs,
    )
    city = gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(-500, -800, 5_500, 800)], crs=crs)
    centres = gpd.GeoDataFrame(
        {"name": ["Stare Miasto"], "role": ["required_city_centre"]},
        geometry=[centre],
        crs=crs,
    )
    config = pl.MetroConfig(
        length_m=5_000.0,
        station_count=4,
        walk_radius_m=800.0,
        grid_station_step_m=1_000.0,
        grid_max_station_candidates=4,
        grid_connectivity_candidate_share=0.75,
        station_min_spacing_m=700.0,
        station_max_spacing_m=1_100.0,
    )

    candidates = pl.grid_station_candidates(
        demand,
        city_boundary=city,
        centres=centres,
        config=config,
    )

    non_required = candidates[~candidates["required"].astype(bool)]
    assert "connectivity_support" in set(non_required["grid_selection_role"])
    assert non_required["distance_to_required_centre_m"].min() <= config.station_max_spacing_m


def test_grid_station_route_selects_actual_station_count_without_anchors():
    crs = pl.LOCAL_CRS
    centre = Point(0, 0)
    station_points = [Point(-2_000, 0), Point(-1_000, 0), centre, Point(1_000, 0), Point(2_000, 0)]
    demand = gpd.GeoDataFrame(
        {"population": [100.0] * len(station_points)},
        geometry=station_points,
        crs=crs,
    )
    candidates = gpd.GeoDataFrame(
        {
            "candidate_id": ["C", "W2", "W1", "E1", "E2"],
            "name": ["centre", "west2", "west1", "east1", "east2"],
            "source": ["required", "grid_station", "grid_station", "grid_station", "grid_station"],
            "required": [True, False, False, False, False],
            "candidate_weight": [100.0] * 5,
        },
        geometry=[centre, Point(-2_000, 0), Point(-1_000, 0), Point(1_000, 0), Point(2_000, 0)],
        crs=crs,
    )
    config = pl.MetroConfig(
        length_m=4_000.0,
        station_count=5,
        walk_radius_m=500.0,
        station_min_spacing_m=900.0,
        station_max_spacing_m=1_100.0,
        turn_penalty_per_degree=0.0,
        corridor_detour_penalty_per_ratio=0.0,
        corridor_backtrack_penalty_per_km=0.0,
        grid_length_usage_bonus_per_km=0.0,
        grid_length_underuse_penalty_per_km=0.0,
    )

    result = pl.solve_grid_station_route(
        demand=demand,
        station_candidates=candidates,
        centre=centre,
        config=config,
    )

    assert len(result["stations"]) == 5
    assert result["line"]["algorithm"].iloc[0] == "grid_station_orienteering"
    assert result["line"].geometry.iloc[0].length == 4_000.0

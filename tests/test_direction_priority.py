import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
from shapely.geometry import Point

from wro_metro import planning as pl


def test_direction_octant_classifies_northeast():
    assert pl.direction_octant_from_delta(2_000, 2_000) == "northeast"
    assert pl.broad_direction_from_octant("northeast") == "north"


def test_direction_priority_can_select_northeast_anchor():
    crs = pl.LOCAL_CRS
    centre = Point(0, 0)
    east = Point(2_000, 0)
    northeast = Point(2_000, 2_000)
    demand = gpd.GeoDataFrame(
        {"population": [100.0, 100.0]},
        geometry=[east, northeast],
        crs=crs,
    )
    candidates = gpd.GeoDataFrame(
        {
            "candidate_id": ["C0", "C1", "C2"],
            "name": ["centre", "east", "northeast"],
            "source": ["forced_centre", "demand", "demand"],
            "required": [True, False, False],
            "direction_sector": ["centre", "east", "north"],
            "direction_octant": ["centre", "east", "northeast"],
            "distance_to_required_centre_m": [0.0, 2_000.0, 2_828.0],
        },
        geometry=[centre, east, northeast],
        crs=crs,
    )
    config = pl.MetroConfig(
        length_m=5_000.0,
        station_count=5,
        route_anchor_count=2,
        walk_radius_m=800.0,
        direction_priority_sectors=("northeast",),
        direction_priority_bonus=5_000.0,
        direction_priority_min_distance_m=1_000.0,
        direction_priority_seed_line_id=1,
        station_min_spacing_m=250.0,
        station_max_spacing_m=2_500.0,
        turn_penalty_per_degree=0.0,
        corridor_detour_penalty_per_ratio=0.0,
        corridor_backtrack_penalty_per_km=0.0,
    )

    result = pl.solve_orienteering_route(
        demand=demand,
        candidates=candidates,
        centre=centre,
        config=config,
    )

    selected_octants = set(result["anchors"]["direction_octant"].dropna())
    assert "northeast" in selected_octants
    assert bool(result["line"]["forced_direction_priority_anchor"].iloc[0])
    assert float(result["line"]["direction_priority_bonus"].iloc[0]) > 0.0

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

import wro_metro as wm


def test_safe_area_relocation_ignores_tiny_safe_sliver():
    crs = wm.LOCAL_CRS
    city = gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(0, 0, 4_000, 1_000)], crs=crs)
    forbidden = gpd.GeoDataFrame({"risk": ["flood"]}, geometry=[box(200, 0, 2_000, 1_000)], crs=crs)
    demand = gpd.GeoDataFrame({"population": [100.0]}, geometry=[Point(1_000, 500)], crs=crs)

    relocation_areas = wm.safe_relocation_areas(
        demand_areas=city,
        forbidden=forbidden,
        city_boundary=city,
        min_area_km2=0.5,
        flood_safety_buffer_m=50.0,
        interior_buffer_m=25.0,
    )

    relocated = wm.relocate_demand_to_safe_areas(
        demand,
        forbidden,
        relocation_areas=relocation_areas,
    )

    assert len(relocation_areas) == 1
    assert relocation_areas["area_km2"].iloc[0] >= 0.5
    assert relocated["was_relocated"].iloc[0]
    assert relocated["relocation_area_km2"].iloc[0] == pytest.approx(relocation_areas["area_km2"].iloc[0])
    assert relocated.geometry.iloc[0].x > 2_000
    assert not forbidden.geometry.iloc[0].buffer(50).covers(relocated.geometry.iloc[0])

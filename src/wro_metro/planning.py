"""Geospatial helpers for a first-pass Wroclaw metro planning notebook.

The module intentionally keeps the model explainable:
- demand is represented by weighted points,
- flood/forbidden zones remove or relocate demand,
- each metro line has the same length and station count as Warsaw M1 by default,
- candidate lines are radial lines forced through a chosen city centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import atan2, cos, degrees, hypot, pi, radians, sin
from pathlib import Path
from typing import Iterable, Mapping
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, substring, unary_union

WGS84 = 4326
LOCAL_CRS = 2177
WEB_MERCATOR = 3857
WARSAW_M1_LENGTH_M = 23_100.0
WARSAW_M1_STATIONS = 21


@dataclass(frozen=True)
class MetroConfig:
    """Planning knobs that are safe to tweak in the notebook."""

    length_m: float = WARSAW_M1_LENGTH_M
    station_count: int = WARSAW_M1_STATIONS
    walk_radius_m: float = 1_100.0
    cost_per_km_mln: float = 650.0
    flood_cost_multiplier: float = 2.0
    angle_step_deg: float = 2.0
    forbidden_penalty_per_km: float = 60_000.0
    max_forbidden_km: float | None = None
    forbidden_excess_penalty_per_km: float = 0.0
    outside_city_penalty_per_km: float = 500_000.0
    max_outside_city_km: float = 0.15
    outside_city_excess_penalty_per_km: float = 5_000_000.0
    # Legacy flat penalty kept for older notebook variants.
    geology_penalty: float = 1_000.0
    geology_penalty_per_km: float = 40_000.0
    high_geology_factor_threshold: float = 1.50
    high_geology_penalty_per_km: float = 120_000.0
    anchor_geology_penalty_per_excess: float = 35_000.0
    anchor_high_geology_penalty: float = 25_000.0
    station_geology_penalty_per_excess: float = 12_000.0
    station_high_geology_penalty: float = 8_000.0
    station_geology_score_factor: float = 1.40
    river_crossing_bonus_per_km: float = 600.0
    transfer_bonus_per_interchange: float = 35_000.0
    interchange_radius_m: float = 450.0
    line_overlap_penalty_per_km: float = 350_000.0
    parallel_line_buffer_m: float = 900.0
    candidate_existing_line_buffer_m: float = 900.0
    candidate_radial_overlap_limit_km: float = 1.50
    parallel_overlap_angle_deg: float = 35.0
    max_line_overlap_km: float = 0.75
    line_overlap_excess_penalty_per_km: float = 5_000_000.0
    arm_reuse_angle_deg: float = 35.0
    arm_reuse_penalty: float = 10_000_000.0
    residual_coverage_multiplier: float = 1.35
    residual_corridor_radius_m: float = 2_200.0
    residual_corridor_coverage_multiplier: float = 1.10
    relocation_search_radius_m: float = 3_000.0
    relocation_step_m: float = 100.0
    relocation_min_safe_area_km2: float = 0.35
    relocation_safe_area_buffer_m: float = 120.0
    relocation_safe_area_interior_buffer_m: float = 90.0
    relocation_study_area_buffer_m: float = 2_000.0
    station_risk_buffer_m: float = 260.0
    flood_zones_are_cost_only: bool = False
    candidate_catchment_radius_m: float = 1_600.0
    central_anchor_radius_m: float = 3_000.0
    min_central_anchor_candidates: int = 8
    max_central_anchor_share: float = 0.32
    min_regional_anchor_candidates: int = 8
    min_directional_anchor_candidates_per_sector: int = 5
    min_octant_anchor_candidates_per_sector: int = 2
    regional_anchor_weight_floor_fraction: float = 0.08
    direction_priority_sectors: tuple[str, ...] = ()
    direction_priority_bonus: float = 0.0
    direction_priority_min_distance_m: float = 2_500.0
    direction_priority_max_per_route: int = 2
    direction_priority_seed_line_id: int | None = None
    force_southern_anchor_line_id: int | None = None
    force_southern_anchor_min_distance_m: float = 2_500.0
    route_anchor_count: int = 6
    max_turn_angle_deg: float = 45.0
    hard_max_turn_angle_deg: float = 85.0
    turn_penalty_per_degree: float = 1_500.0
    minimum_curve_radius_m: float = 450.0
    curve_radius_penalty_per_m: float = 90.0
    corridor_detour_ratio_limit: float = 1.35
    corridor_detour_penalty_per_ratio: float = 90_000.0
    corridor_backtrack_penalty_per_km: float = 45_000.0
    corridor_backtrack_tolerance_m: float = 350.0
    adaptive_station_placement: bool = True
    station_min_spacing_m: float = 800.0
    station_max_spacing_m: float = 1_700.0
    station_candidate_step_m: float = 125.0
    station_flood_score_factor: float = 0.03
    station_water_buffer_m: float = 90.0
    station_water_score_factor: float = 0.02
    station_terminal_flex_m: float = 1_500.0
    station_anchor_bonus_radius_m: float = 260.0
    station_anchor_bonus: float = 7_500.0
    grid_station_step_m: float = 700.0
    grid_max_station_candidates: int = 420
    grid_length_usage_bonus_per_km: float = 3_000.0
    grid_length_underuse_penalty_per_km: float = 8_000.0
    grid_min_length_ratio: float = 0.82
    grid_use_cuda: bool = False
    grid_cuda_batch_size: int = 2048
    grid_connectivity_candidate_share: float = 0.35
    grid_include_demand_points: bool = True
    grid_station_gap_penalty_per_m: float = 30.0
    grid_existing_line_node_penalty: float = 60_000.0

    @property
    def station_spacing_m(self) -> float:
        if self.station_count <= 1:
            return 0.0
        return self.length_m / (self.station_count - 1)


def _gdf_from_lonlat(records: Iterable[Mapping], crs: int = LOCAL_CRS) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs=WGS84,
    )
    return gdf.to_crs(crs)


def _align_crs(gdf: gpd.GeoDataFrame | None, target_crs) -> gpd.GeoDataFrame:
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    if gdf.empty:
        return gdf.set_crs(target_crs, allow_override=True) if gdf.crs is None else gdf.to_crs(target_crs)
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set it before spatial operations.")
    return gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf.copy()


def demo_demand(crs: int = LOCAL_CRS) -> gpd.GeoDataFrame:
    """Small synthetic demand set for immediate notebook runs.

    Replace this with SIP demography or polling-place turnout data once raw data is
    downloaded. Values are rough proxy weights, not official population counts.
    """

    records = [
        {"name": "Rynek", "lon": 17.0325, "lat": 51.1107, "population": 42_000},
        {"name": "Dworzec Glowny", "lon": 17.0369, "lat": 51.0989, "population": 55_000},
        {"name": "Plac Grunwaldzki", "lon": 17.0618, "lat": 51.1115, "population": 64_000},
        {"name": "Nadodrze", "lon": 17.0344, "lat": 51.1245, "population": 46_000},
        {"name": "Popowice", "lon": 16.9970, "lat": 51.1220, "population": 49_000},
        {"name": "Kozanow", "lon": 16.9580, "lat": 51.1327, "population": 39_000},
        {"name": "Nowy Dwor", "lon": 16.9560, "lat": 51.1150, "population": 45_000},
        {"name": "Muchobor Wielki", "lon": 16.9690, "lat": 51.1000, "population": 36_000},
        {"name": "Lesnica", "lon": 16.8720, "lat": 51.1440, "population": 31_000},
        {"name": "Karłowice", "lon": 17.0500, "lat": 51.1400, "population": 33_000},
        {"name": "Psie Pole", "lon": 17.0990, "lat": 51.1470, "population": 38_000},
        {"name": "Biskupin", "lon": 17.1010, "lat": 51.1100, "population": 29_000},
        {"name": "Brochow", "lon": 17.0790, "lat": 51.0680, "population": 30_000},
        {"name": "Jagodno", "lon": 17.0660, "lat": 51.0530, "population": 35_000},
        {"name": "Gaj", "lon": 17.0420, "lat": 51.0730, "population": 48_000},
        {"name": "Tarnogaj", "lon": 17.0550, "lat": 51.0850, "population": 32_000},
        {"name": "Borek", "lon": 17.0070, "lat": 51.0810, "population": 40_000},
        {"name": "Klecina", "lon": 16.9760, "lat": 51.0710, "population": 28_000},
        {"name": "Ołtaszyn", "lon": 17.0300, "lat": 51.0430, "population": 26_000},
    ]
    return _gdf_from_lonlat(records, crs=crs)


def demo_centres(crs: int = LOCAL_CRS) -> gpd.GeoDataFrame:
    records = [
        {"name": "Rynek", "role": "required_city_centre", "lon": 17.0325, "lat": 51.1107},
        {"name": "Dworzec Glowny", "role": "rail_hub", "lon": 17.0369, "lat": 51.0989},
        {"name": "Plac Grunwaldzki", "role": "university_hospital_hub", "lon": 17.0618, "lat": 51.1115},
        {"name": "Plac Jana Pawla II", "role": "west_centre", "lon": 17.0193, "lat": 51.1112},
    ]
    return _gdf_from_lonlat(records, crs=crs)


def demo_flood_zones(crs: int = LOCAL_CRS) -> gpd.GeoDataFrame:
    """Approximate flood/river proxy zones used only for the demo workflow."""

    odra = LineString(
        [
            (16.89, 51.123),
            (16.96, 51.128),
            (17.015, 51.117),
            (17.055, 51.112),
            (17.11, 51.128),
        ]
    )
    widawa = LineString(
        [
            (16.93, 51.160),
            (17.00, 51.154),
            (17.08, 51.160),
            (17.13, 51.151),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"name": ["Odra flood proxy", "Widawa flood proxy"], "risk": ["demo", "demo"]},
        geometry=[odra, widawa],
        crs=WGS84,
    ).to_crs(crs)
    gdf["geometry"] = [geom.buffer(450) for geom in gdf.geometry]
    return gdf


def demo_geology(crs: int = LOCAL_CRS) -> gpd.GeoDataFrame:
    """Toy geology-cost zones. Replace with real geology/borehole layers."""

    records = [
        {"name": "west terrace", "cost_factor": 1.08, "geometry": box(16.82, 51.03, 16.98, 51.18)},
        {"name": "central river terrace", "cost_factor": 1.35, "geometry": box(16.98, 51.03, 17.06, 51.18)},
        {"name": "east terrace", "cost_factor": 1.12, "geometry": box(17.06, 51.03, 17.14, 51.18)},
    ]
    return gpd.GeoDataFrame(records, crs=WGS84).to_crs(crs)


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector file, including zipped shapefiles supported by GeoPandas."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return gpd.read_file(path)
    except Exception:
        return gpd.read_file(f"zip://{path}")


def shapefiles_in_zip(path: str | Path) -> list[str]:
    path = Path(path)
    with ZipFile(path) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".shp")]


def read_zipped_shapefile(path: str | Path, member: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    if member is None:
        members = shapefiles_in_zip(path)
        if len(members) != 1:
            raise ValueError(f"Expected exactly one shapefile in {path}, found {len(members)}.")
        member = members[0]
    return gpd.read_file(f"zip://{path}!{member}")


def wroclaw_demography_member(
    path: str | Path,
    year: int | None = 2025,
    unit: str = "REJSTAT",
) -> str:
    """Pick the requested or latest demography shapefile from the SIP archive."""

    unit = unit.upper()
    matches: list[tuple[int, str]] = []
    for member in shapefiles_in_zip(path):
        stem = Path(member).stem.upper()
        if not stem.startswith(f"{unit}_"):
            continue
        date_token = stem.split("_")[-1]
        if len(date_token) >= 4 and date_token[:4].isdigit():
            matches.append((int(date_token[:4]), member))

    if not matches:
        raise ValueError(f"No {unit} demography shapefiles found in {path}.")

    if year is not None:
        requested = [member for found_year, member in matches if found_year == year]
        if requested:
            return requested[0]

    return sorted(matches, key=lambda item: item[0])[-1][1]


def load_wroclaw_demography(
    data_dir: str | Path = "data/raw",
    year: int | None = 2025,
    unit: str = "REJSTAT",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    """Load official SIP Wroclaw demography from data/raw."""

    path = Path(data_dir) / "dem-rejurb-rejstat-shp.zip"
    member = wroclaw_demography_member(path, year=year, unit=unit)
    return read_zipped_shapefile(path, member).to_crs(target_crs)


def load_wroclaw_osiedla(
    data_dir: str | Path = "data/raw",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    path = Path(data_dir) / "granice-osiedli.zip"
    return read_zipped_shapefile(path).to_crs(target_crs)


def city_boundary_from_layer(layer: gpd.GeoDataFrame, name: str = "Wroclaw") -> gpd.GeoDataFrame:
    """Build a single city-area polygon from districts or other city polygons."""

    if layer is None or layer.empty:
        return gpd.GeoDataFrame({"name": []}, geometry=[], crs=getattr(layer, "crs", LOCAL_CRS))
    geoms = []
    for geom in layer.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms.append(geom if geom.is_valid else geom.buffer(0))
    if not geoms:
        return gpd.GeoDataFrame({"name": []}, geometry=[], crs=layer.crs)
    return gpd.GeoDataFrame({"name": [name]}, geometry=[unary_union(geoms)], crs=layer.crs)


def load_wroclaw_surface_water(
    data_dir: str | Path = "data/raw",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    path = Path(data_dir) / "wody-powierzchniowe.zip"
    return read_zipped_shapefile(path).to_crs(target_crs)


def load_forbidden_zones_from_raw(
    data_dir: str | Path = "data/raw",
    target_crs: int = LOCAL_CRS,
    water_buffer_m: float = 80.0,
    use_surface_water_proxy: bool = False,
) -> gpd.GeoDataFrame | None:
    """Load real flood/forbidden zones if present.

    Surface water is not a flood-risk substitute for metro routing. It can be
    optionally loaded as a weak proxy only for early experiments, but the notebook
    treats rivers separately as communication barriers/crossing opportunities.
    """

    def _finalize(zones: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
        zones = zones.to_crs(target_crs)
        if "source_layer" not in zones.columns:
            zones["source_layer"] = source_name
        if "risk" not in zones.columns:
            zones["risk"] = "flood_zone"
        return zones

    data_dir = Path(data_dir)
    for name in ["flood_zones.geojson", "flood_zones.gpkg", "flood_zones.shp", "flood_zones.zip"]:
        path = data_dir / name
        if path.exists():
            return _finalize(read_vector(path), name)

    folder = data_dir / "flood_zones"
    if folder.exists():
        shapefiles = sorted(folder.glob("*.shp"))
        if shapefiles:
            return _finalize(gpd.read_file(shapefiles[0]), f"flood_zones/{shapefiles[0].name}")

    if not use_surface_water_proxy:
        return None

    water_path = data_dir / "wody-powierzchniowe.zip"
    if not water_path.exists():
        return None
    water = load_wroclaw_surface_water(data_dir, target_crs=target_crs)
    proxy = water.copy()
    proxy["source_layer"] = "wody-powierzchniowe.zip"
    proxy["risk"] = "surface_water_proxy_not_flood_zone"
    proxy["geometry"] = proxy.geometry.buffer(water_buffer_m)
    return proxy


def load_water_crossing_layer(
    data_dir: str | Path = "data/raw",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame | None:
    """Load rivers/surface water as a crossing/barrier layer, not forbidden land."""

    path = Path(data_dir) / "wody-powierzchniowe.zip"
    if not path.exists():
        return None
    water = load_wroclaw_surface_water(data_dir, target_crs=target_crs)
    water["source_layer"] = "wody-powierzchniowe.zip"
    return water


def load_geology_cost_layer_from_raw(
    data_dir: str | Path = "data/raw",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame | None:
    """Load a real geology/cost multiplier layer if the user supplies one."""

    data_dir = Path(data_dir)
    candidates = [
        "geology.geojson",
        "geology.gpkg",
        "geology.shp",
        "geology.zip",
        "cost_zones.geojson",
        "cost_zones.gpkg",
        "cost_zones.shp",
        "cost_zones.zip",
    ]
    cost_columns = ["cost_factor", "cost", "factor", "multiplier", "mult", "cost_multip"]

    def _normalise(geology: gpd.GeoDataFrame, source_layer: str) -> gpd.GeoDataFrame:
        geology = geology.to_crs(target_crs)
        found = next((col for col in cost_columns if col in geology.columns), None)
        if found and found != "cost_factor":
            geology["cost_factor"] = pd.to_numeric(geology[found], errors="coerce").fillna(1.0)
        elif "cost_factor" not in geology.columns:
            geology["cost_factor"] = 1.0
        else:
            geology["cost_factor"] = pd.to_numeric(geology["cost_factor"], errors="coerce").fillna(1.0)
        geology["source_layer"] = source_layer
        return geology

    for name in candidates:
        path = data_dir / name
        if path.exists():
            try:
                return _normalise(gpd.read_file(path), name)
            except Exception:
                try:
                    return _normalise(gpd.read_file(f"zip://{path}"), name)
                except Exception:
                    continue

    folder = data_dir / "geology"
    if folder.exists() and folder.is_dir():
        shapefiles = sorted(folder.glob("*.shp"))
        if shapefiles:
            return _normalise(gpd.read_file(shapefiles[0]), f"geology/{shapefiles[0].name}")
    return None


def demand_areas_from_polygons(
    gdf: gpd.GeoDataFrame,
    weight_col: str | None = None,
    year: int | None = 2025,
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    """Prepare polygon demand areas for choropleth maps."""

    areas = _align_crs(gdf, target_crs)
    weight_col = weight_col or guess_weight_column(areas, year=year)
    areas = areas[areas.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    areas["population"] = pd.to_numeric(areas[weight_col], errors="coerce").fillna(0.0)
    areas["area_km2"] = areas.geometry.area / 1_000_000.0
    areas["population_density"] = areas["population"] / areas["area_km2"].replace(0, np.nan)
    return areas


def guess_weight_column(frame: pd.DataFrame, year: int | None = 2025) -> str:
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError("No numeric column found for demand weights.")

    tokens = ["population", "ludn", "mieszk", "ogolem", "razem", "votes", "glosy", "frekw"]
    scored: list[tuple[int, str]] = []
    for col in numeric_cols:
        lower = str(col).lower()
        score = sum(token in lower for token in tokens)
        if year is not None and str(year) in lower:
            score += 2
        scored.append((score, col))
    scored.sort(reverse=True)
    return scored[0][1]


def to_demand_points(
    gdf: gpd.GeoDataFrame,
    weight_col: str | None = None,
    year: int | None = 2025,
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    """Convert population polygons or point layers to weighted demand points."""

    gdf = _align_crs(gdf, target_crs)
    weight_col = weight_col or guess_weight_column(gdf, year=year)
    out = gdf.copy()
    out["population"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)

    area_mask = out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    point_geoms = out.geometry.copy()
    point_geoms.loc[area_mask] = out.loc[area_mask].representative_point()
    point_geoms.loc[~area_mask] = out.loc[~area_mask].geometry.centroid
    out = out.set_geometry(point_geoms)
    out = out[out["population"] > 0].copy()
    return out[["population", "geometry"] + [c for c in out.columns if c not in {"population", "geometry"}]]


def geocode_google_addresses(
    frame: pd.DataFrame,
    address_col: str,
    api_key: str,
    city_suffix: str = "Wroclaw, Poland",
    pause_s: float = 0.05,
) -> pd.DataFrame:
    """Geocode addresses with Google Maps Geocoding API.

    The function is deliberately explicit and should be used only with your own API
    key. Store the key outside the notebook, for example in GOOGLE_MAPS_API_KEY.
    """

    import time

    import requests

    if not api_key:
        raise ValueError("Missing Google Maps API key.")

    out = frame.copy()
    latitudes: list[float | None] = []
    longitudes: list[float | None] = []
    statuses: list[str] = []

    for raw_address in out[address_col].fillna("").astype(str):
        query = raw_address if city_suffix.lower() in raw_address.lower() else f"{raw_address}, {city_suffix}"
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": api_key, "region": "pl"},
            timeout=20,
        )
        payload = response.json()
        status = payload.get("status", "UNKNOWN")
        statuses.append(status)
        if status == "OK" and payload.get("results"):
            location = payload["results"][0]["geometry"]["location"]
            latitudes.append(location["lat"])
            longitudes.append(location["lng"])
        else:
            latitudes.append(None)
            longitudes.append(None)
        time.sleep(pause_s)

    out["lat"] = latitudes
    out["lon"] = longitudes
    out["geocode_status"] = statuses
    return out


def load_polling_places_csv(
    path: str | Path,
    votes_col: str = "votes",
    lon_col: str = "lon",
    lat_col: str = "lat",
    target_crs: int = LOCAL_CRS,
) -> gpd.GeoDataFrame:
    """Load polling-place turnout proxies from CSV with lon/lat and vote counts."""

    frame = pd.read_csv(path)
    missing = {votes_col, lon_col, lat_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in polling CSV: {sorted(missing)}")
    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame[lon_col], frame[lat_col]),
        crs=WGS84,
    ).to_crs(target_crs)
    gdf["population"] = pd.to_numeric(gdf[votes_col], errors="coerce").fillna(0.0)
    return gdf[gdf["population"] > 0].copy()


def forbidden_union(forbidden: gpd.GeoDataFrame | None):
    if forbidden is None:
        return None
    if not isinstance(forbidden, gpd.GeoDataFrame):
        return forbidden
    if forbidden.empty:
        return None
    return unary_union([geom for geom in forbidden.geometry if geom is not None and not geom.is_empty])


def direction_octant_from_delta(dx: float, dy: float, centre_tolerance_m: float = 1.0) -> str:
    """Classify a vector into an 8-way city arm around the required centre."""

    if hypot(float(dx), float(dy)) <= centre_tolerance_m:
        return "centre"
    angle = (degrees(atan2(float(dy), float(dx))) + 360.0) % 360.0
    sectors = [
        (22.5, "east"),
        (67.5, "northeast"),
        (112.5, "north"),
        (157.5, "northwest"),
        (202.5, "west"),
        (247.5, "southwest"),
        (292.5, "south"),
        (337.5, "southeast"),
        (360.0, "east"),
    ]
    for threshold, name in sectors:
        if angle < threshold:
            return name
    return "east"


def broad_direction_from_octant(octant: str) -> str:
    if octant in {"north", "northeast", "northwest"}:
        return "north"
    if octant in {"south", "southeast", "southwest"}:
        return "south"
    if octant == "west":
        return "west"
    if octant == "east":
        return "east"
    return "centre"


def _iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)


def nearest_free_point(
    point: Point,
    forbidden_geom,
    max_radius_m: float = 3_000.0,
    step_m: float = 100.0,
    angle_count: int = 64,
) -> tuple[Point, float]:
    """Move a point inside forbidden geometry to the nearest sampled free point."""

    if forbidden_geom is None or forbidden_geom.is_empty or not forbidden_geom.covers(point):
        return point, 0.0

    angles = np.linspace(0.0, 2.0 * pi, angle_count, endpoint=False)
    for radius in np.arange(step_m, max_radius_m + step_m, step_m):
        candidates = [
            Point(point.x + radius * cos(angle), point.y + radius * sin(angle))
            for angle in angles
        ]
        free = [candidate for candidate in candidates if not forbidden_geom.covers(candidate)]
        if free:
            best = min(free, key=point.distance)
            return best, point.distance(best)
    return point, float("nan")


def nearest_free_point_from_boundary(
    point: Point,
    forbidden_geom,
    step_m: float = 40.0,
    max_extra_m: float = 600.0,
) -> tuple[Point, float]:
    """Move a point outside a forbidden geometry using the nearest boundary."""

    if forbidden_geom is None or forbidden_geom.is_empty or not forbidden_geom.covers(point):
        return point, 0.0

    boundary_point = nearest_points(point, forbidden_geom.boundary)[1]
    vector = np.array([boundary_point.x - point.x, boundary_point.y - point.y], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return nearest_free_point(point, forbidden_geom, max_radius_m=max_extra_m, step_m=step_m, angle_count=24)

    unit = vector / norm
    for extra in np.arange(step_m, max_extra_m + step_m, step_m):
        candidate = Point(boundary_point.x + unit[0] * extra, boundary_point.y + unit[1] * extra)
        if not forbidden_geom.covers(candidate):
            return candidate, point.distance(candidate)

    return nearest_free_point(point, forbidden_geom, max_radius_m=max_extra_m, step_m=step_m, angle_count=24)


def relocate_demand_from_forbidden(
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    config: MetroConfig = MetroConfig(),
) -> gpd.GeoDataFrame:
    """Preserve demand weights, but move points from forbidden zones to free space."""

    demand = demand.copy()
    forbidden = _align_crs(forbidden, demand.crs)
    union = forbidden_union(forbidden)
    new_geometries: list[Point] = []
    offsets: list[float] = []
    for point in demand.geometry:
        new_point, offset = nearest_free_point(
            point,
            union,
            max_radius_m=config.relocation_search_radius_m,
            step_m=config.relocation_step_m,
        )
        new_geometries.append(new_point)
        offsets.append(offset)

    out = demand.copy()
    out["original_geometry"] = list(demand.geometry)
    out["relocated_m"] = offsets
    out["was_relocated"] = pd.Series(offsets, index=out.index).fillna(0.0) > 0
    return out.set_geometry(gpd.GeoSeries(new_geometries, index=out.index, crs=demand.crs))


def relocate_demand_to_safe_land(
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    config: MetroConfig = MetroConfig(),
) -> gpd.GeoDataFrame:
    """Move demand from flood-risk polygons to the nearest sampled safe land."""

    demand = demand.copy()
    forbidden = _align_crs(forbidden, demand.crs)
    union = forbidden_union(forbidden)
    new_geometries: list[Point] = []
    offsets: list[float] = []
    for point in demand.geometry:
        new_point, offset = nearest_free_point_from_boundary(
            point,
            union,
            step_m=config.relocation_step_m,
            max_extra_m=config.relocation_search_radius_m,
        )
        new_geometries.append(new_point)
        offsets.append(offset)

    out = demand.copy()
    out["original_geometry"] = list(demand.geometry)
    out["relocated_m"] = offsets
    out["was_relocated"] = pd.Series(offsets, index=out.index).fillna(0.0) > 0
    out["safe_land_relocation"] = out["was_relocated"]
    return out.set_geometry(gpd.GeoSeries(new_geometries, index=out.index, crs=demand.crs))


def safe_relocation_areas(
    demand_areas: gpd.GeoDataFrame | None,
    forbidden: gpd.GeoDataFrame | None,
    city_boundary: gpd.GeoDataFrame | None = None,
    avoid: gpd.GeoDataFrame | None = None,
    min_area_km2: float | None = None,
    flood_safety_buffer_m: float | None = None,
    interior_buffer_m: float | None = None,
    config: MetroConfig = MetroConfig(),
) -> gpd.GeoDataFrame:
    """Build stable safe-land polygons used as destinations for relocated demand.

    The destination layer is intentionally more conservative than a simple
    nearest-free-point search: it subtracts flood-risk land with a small buffer,
    optionally subtracts rivers/surface water, erodes the remainder inward, and
    keeps only sizeable polygons. That prevents demand from being moved onto
    narrow slivers along the edge of flood zones.
    """

    min_area_km2 = (
        float(config.relocation_min_safe_area_km2)
        if min_area_km2 is None
        else float(min_area_km2)
    )
    flood_safety_buffer_m = (
        float(config.relocation_safe_area_buffer_m)
        if flood_safety_buffer_m is None
        else float(flood_safety_buffer_m)
    )
    interior_buffer_m = (
        float(config.relocation_safe_area_interior_buffer_m)
        if interior_buffer_m is None
        else float(interior_buffer_m)
    )

    target_crs = LOCAL_CRS
    for layer in (city_boundary, demand_areas, forbidden, avoid):
        if isinstance(layer, gpd.GeoDataFrame) and layer.crs is not None and not layer.empty:
            target_crs = layer.crs
            break

    demand_areas = _align_crs(demand_areas, target_crs)
    city_boundary = _align_crs(city_boundary, target_crs)
    forbidden = _align_crs(forbidden, target_crs)
    avoid = _align_crs(avoid, target_crs)

    base_geom = forbidden_union(city_boundary) if city_boundary is not None and not city_boundary.empty else None
    if base_geom is None:
        base_geom = forbidden_union(demand_areas) if demand_areas is not None and not demand_areas.empty else None
    if base_geom is None or base_geom.is_empty:
        return gpd.GeoDataFrame(
            {
                "relocation_area_id": pd.Series(dtype="int64"),
                "area_km2": pd.Series(dtype="float64"),
                "meets_min_area": pd.Series(dtype="bool"),
            },
            geometry=[],
            crs=target_crs,
        )

    blocked_parts = []
    forbidden_geom = forbidden_union(forbidden)
    if forbidden_geom is not None and not forbidden_geom.is_empty:
        blocked_parts.append(forbidden_geom.buffer(max(0.0, float(flood_safety_buffer_m))))
    avoid_geom = forbidden_union(avoid)
    if avoid_geom is not None and not avoid_geom.is_empty:
        blocked_parts.append(avoid_geom.buffer(40.0))

    blocked_geom = unary_union(blocked_parts) if blocked_parts else None
    safe_geom = base_geom.difference(blocked_geom) if blocked_geom is not None and not blocked_geom.is_empty else base_geom
    if (safe_geom is None or safe_geom.is_empty) and forbidden_geom is not None and not forbidden_geom.is_empty:
        safe_geom = base_geom.difference(forbidden_geom)

    placement_geom = safe_geom
    if placement_geom is not None and not placement_geom.is_empty and interior_buffer_m > 0:
        eroded = placement_geom.buffer(-float(interior_buffer_m))
        if eroded is not None and not eroded.is_empty:
            placement_geom = eroded

    parts = [part for part in _iter_polygon_parts(placement_geom) if part is not None and not part.is_empty]
    min_area_m2 = max(0.0, float(min_area_km2)) * 1_000_000.0
    sizeable_parts = [part for part in parts if part.area >= min_area_m2]
    kept_parts = sizeable_parts if sizeable_parts else sorted(parts, key=lambda geom: geom.area, reverse=True)[:1]

    rows = []
    for idx, geom in enumerate(sorted(kept_parts, key=lambda item: item.area, reverse=True), start=1):
        rows.append(
            {
                "relocation_area_id": idx,
                "area_km2": float(geom.area / 1_000_000.0),
                "meets_min_area": bool(geom.area >= min_area_m2),
                "min_area_km2": float(min_area_km2),
                "flood_safety_buffer_m": float(flood_safety_buffer_m),
                "interior_buffer_m": float(interior_buffer_m),
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)


def _nearest_point_in_relocation_areas(point: Point, relocation_areas: gpd.GeoDataFrame) -> tuple[Point, float, object, float]:
    best: tuple[float, float, Point, object, float] | None = None
    for _, row in relocation_areas.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        candidate = point if geom.covers(point) else nearest_points(point, geom)[1]
        distance = float(point.distance(candidate))
        area_km2 = float(row.get("area_km2", geom.area / 1_000_000.0))
        area_id = row.get("relocation_area_id", None)
        score = (distance, -area_km2, candidate, area_id, area_km2)
        if best is None or score[:2] < best[:2]:
            best = score
    if best is None:
        return point, float("nan"), None, float("nan")
    distance, _, candidate, area_id, area_km2 = best
    return candidate, distance, area_id, area_km2


def relocate_demand_to_safe_areas(
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    relocation_areas: gpd.GeoDataFrame | None = None,
    demand_areas: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    avoid: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    min_area_km2: float | None = None,
    flood_safety_buffer_m: float | None = None,
    interior_buffer_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Move demand from flood-risk zones into larger, stable safe-land areas."""

    demand = demand.copy()
    forbidden = _align_crs(forbidden, demand.crs)
    union = forbidden_union(forbidden)
    if relocation_areas is None:
        relocation_areas = safe_relocation_areas(
            demand_areas=demand_areas,
            forbidden=forbidden,
            city_boundary=city_boundary,
            avoid=avoid,
            min_area_km2=min_area_km2,
            flood_safety_buffer_m=flood_safety_buffer_m,
            interior_buffer_m=interior_buffer_m,
            config=config,
        )
    relocation_areas = _align_crs(relocation_areas, demand.crs)

    new_geometries: list[Point] = []
    offsets: list[float] = []
    area_ids: list[object] = []
    area_sizes: list[float] = []
    methods: list[str] = []

    for point in demand.geometry:
        if union is None or union.is_empty or not union.covers(point):
            new_geometries.append(point)
            offsets.append(0.0)
            area_ids.append(None)
            area_sizes.append(float("nan"))
            methods.append("unchanged")
            continue

        if relocation_areas is not None and not relocation_areas.empty:
            new_point, offset, area_id, area_km2 = _nearest_point_in_relocation_areas(point, relocation_areas)
            if new_point is not None and not pd.isna(offset):
                new_geometries.append(new_point)
                offsets.append(offset)
                area_ids.append(area_id)
                area_sizes.append(area_km2)
                methods.append("safe_area")
                continue

        new_point, offset = nearest_free_point_from_boundary(
            point,
            union,
            step_m=config.relocation_step_m,
            max_extra_m=config.relocation_search_radius_m,
        )
        new_geometries.append(new_point)
        offsets.append(offset)
        area_ids.append(None)
        area_sizes.append(float("nan"))
        methods.append("nearest_free_fallback")

    out = demand.copy()
    out["original_geometry"] = list(demand.geometry)
    out["relocated_m"] = offsets
    out["was_relocated"] = pd.Series(offsets, index=out.index).fillna(0.0) > 0
    out["safe_land_relocation"] = out["was_relocated"]
    out["relocation_area_id"] = area_ids
    out["relocation_area_km2"] = area_sizes
    out["relocation_method"] = methods
    return out.set_geometry(gpd.GeoSeries(new_geometries, index=out.index, crs=demand.crs))


def demand_relocation_vectors(demand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return LineStrings from original demand points to relocated demand points."""

    if "original_geometry" not in demand.columns:
        return gpd.GeoDataFrame(
            {"relocated_m": pd.Series(dtype="float64")},
            geometry=[],
            crs=demand.crs,
        )
    rows = []
    for _, row in demand.iterrows():
        offset = float(row.get("relocated_m", 0.0) or 0.0)
        if offset <= 0:
            continue
        original = row.get("original_geometry")
        current = row.geometry
        if original is None or current is None or original.is_empty or current.is_empty:
            continue
        rows.append(
            {
                "relocated_m": offset,
                "population": float(row.get("population", 0.0) or 0.0),
                "relocation_area_id": row.get("relocation_area_id", None),
                "geometry": LineString([original, current]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=demand.crs)


def weighted_kmeans(
    demand: gpd.GeoDataFrame,
    k: int,
    weight_col: str = "population",
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Small weighted k-means implementation for regional demand centres."""

    if k <= 0:
        raise ValueError("k must be positive.")
    coords = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    k = min(k, len(coords))
    seeds = np.argsort(weights)[-k:]
    centres = coords[seeds].astype(float)
    labels = np.zeros(len(coords), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        new_centres = centres.copy()
        for cluster_id in range(k):
            mask = new_labels == cluster_id
            if mask.any() and weights[mask].sum() > 0:
                new_centres[cluster_id] = np.average(coords[mask], axis=0, weights=weights[mask])
        if np.array_equal(new_labels, labels) and np.allclose(new_centres, centres):
            break
        labels = new_labels
        centres = new_centres
    return centres, labels


def regional_centres_from_demand(
    demand: gpd.GeoDataFrame,
    k: int = 8,
    weight_col: str = "population",
) -> gpd.GeoDataFrame:
    centres, labels = weighted_kmeans(demand, k=k, weight_col=weight_col)
    rows = []
    weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    for cluster_id, xy in enumerate(centres):
        mask = labels == cluster_id
        rows.append(
            {
                "centre_id": cluster_id + 1,
                "demand_weight": float(weights[mask].sum()),
                "geometry": Point(float(xy[0]), float(xy[1])),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=demand.crs)


def regional_clusters_from_demand_areas(
    demand_areas: gpd.GeoDataFrame,
    k: int = 8,
    weight_col: str = "population",
) -> gpd.GeoDataFrame:
    """Cluster demand polygons into regional service areas."""

    areas = demand_areas.copy()
    if areas.empty:
        return gpd.GeoDataFrame(geometry=[], crs=demand_areas.crs)
    if weight_col not in areas.columns:
        raise ValueError(f"Missing weight column: {weight_col}")

    point_geoms = areas.geometry.representative_point()
    point_gdf = areas.copy().set_geometry(point_geoms)
    centres, labels = weighted_kmeans(point_gdf, k=k, weight_col=weight_col)
    areas["cluster_id"] = labels + 1

    value_cols = [col for col in ["population", "area_km2"] if col in areas.columns]
    agg = {col: "sum" for col in value_cols}
    clusters = areas.dissolve(by="cluster_id", aggfunc=agg).reset_index()
    if "area_km2" not in clusters.columns:
        clusters["area_km2"] = clusters.geometry.area / 1_000_000.0
    if "population" in clusters.columns:
        clusters["population_density"] = clusters["population"] / clusters["area_km2"].replace(0, np.nan)
    clusters["centre_x"] = [float(centres[int(cluster_id) - 1, 0]) for cluster_id in clusters["cluster_id"]]
    clusters["centre_y"] = [float(centres[int(cluster_id) - 1, 1]) for cluster_id in clusters["cluster_id"]]
    return gpd.GeoDataFrame(clusters, geometry="geometry", crs=areas.crs)


def regional_centres_from_clusters(
    clusters: gpd.GeoDataFrame,
    weight_col: str = "population",
) -> gpd.GeoDataFrame:
    """Convert regional cluster polygons to weighted centre points for optimisation."""

    if clusters is None or clusters.empty:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(clusters, "crs", LOCAL_CRS))

    rows = []
    for _, row in clusters.iterrows():
        if "centre_x" in row and "centre_y" in row and pd.notna(row["centre_x"]) and pd.notna(row["centre_y"]):
            point = Point(float(row["centre_x"]), float(row["centre_y"]))
        else:
            point = row.geometry.representative_point()
        rows.append(
            {
                "centre_id": int(row.get("cluster_id", len(rows) + 1)),
                "demand_weight": float(row.get(weight_col, row.get("population", 0.0))),
                "source": "regional_cluster",
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=clusters.crs)


def _pairwise_distance_matrix(
    left_xy: np.ndarray,
    right_xy: np.ndarray | None = None,
    *,
    use_cuda: bool = False,
    batch_size: int = 2048,
) -> tuple[np.ndarray, str]:
    """Return pairwise Euclidean distances, optionally accelerated with Torch CUDA."""

    left_xy = np.asarray(left_xy, dtype=np.float32)
    right_xy = left_xy if right_xy is None else np.asarray(right_xy, dtype=np.float32)
    if left_xy.size == 0 or right_xy.size == 0:
        return np.zeros((len(left_xy), len(right_xy)), dtype=float), "numpy"

    if use_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                chunks = []
                chunk_size = max(1, int(batch_size))
                with torch.no_grad():
                    right_tensor = torch.as_tensor(right_xy, dtype=torch.float32, device="cuda")
                    for start in range(0, len(left_xy), chunk_size):
                        left_tensor = torch.as_tensor(
                            left_xy[start : start + chunk_size],
                            dtype=torch.float32,
                            device="cuda",
                        )
                        chunks.append(torch.cdist(left_tensor, right_tensor).cpu().numpy())
                return np.vstack(chunks).astype(float, copy=False), "cuda"
        except Exception:
            pass

    deltas = left_xy[:, None, :] - right_xy[None, :, :]
    return np.sqrt((deltas**2).sum(axis=2)).astype(float, copy=False), "numpy"


def candidate_catchment_weights(
    candidate_points: list[Point],
    demand: gpd.GeoDataFrame,
    radius_m: float,
    weight_col: str = "population",
    use_cuda: bool = False,
    cuda_batch_size: int = 2048,
) -> np.ndarray:
    """Estimate station-anchor value from nearby demand, not only its own polygon."""

    if not candidate_points or demand.empty or radius_m <= 0:
        return np.zeros(len(candidate_points), dtype=float)

    candidate_xy = np.array([(point.x, point.y) for point in candidate_points], dtype=float)
    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    demand_weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    distances, _ = _pairwise_distance_matrix(
        candidate_xy,
        demand_xy,
        use_cuda=use_cuda,
        batch_size=cuda_batch_size,
    )
    coverage = np.clip(1.0 - distances / radius_m, 0.0, 1.0)
    return coverage @ demand_weights


def candidate_station_sites(
    demand: gpd.GeoDataFrame,
    regional_centres: gpd.GeoDataFrame | None = None,
    centres: gpd.GeoDataFrame | None = None,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    max_candidates: int = 80,
    weight_col: str = "population",
    min_separation_m: float = 250.0,
    catchment_radius_m: float | None = None,
    risk_buffer_m: float | None = None,
    central_anchor_radius_m: float | None = None,
    min_central_anchor_candidates: int | None = None,
    max_central_anchor_share: float | None = None,
    min_regional_anchor_candidates: int | None = None,
    min_directional_anchor_candidates_per_sector: int | None = None,
) -> gpd.GeoDataFrame:
    """Create a compact set of candidate station anchors.

    This is the search space for the NP-hard part of the model. Candidates can be
    demand points, regional demand centres, and forced city centres/hubs.
    In flood-safe mode, flood-risk areas move anchor candidates away from risky
    polygons and a small buffer around them. In cost-only mode, MZP is kept as a
    construction-cost layer and anchors may remain there. Geology reduces the
    score of anchors placed on difficult or high-risk ground.
    """

    demand = demand.copy()
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    water_crossings = _align_crs(water_crossings, demand.crs)
    forbidden_geom = forbidden_union(forbidden)
    water_geom = forbidden_union(water_crossings)
    risk_buffer_m = config.station_risk_buffer_m if risk_buffer_m is None else risk_buffer_m
    catchment_radius_m = (
        config.candidate_catchment_radius_m if catchment_radius_m is None else catchment_radius_m
    )
    central_anchor_radius_m = (
        config.central_anchor_radius_m if central_anchor_radius_m is None else central_anchor_radius_m
    )
    min_central_anchor_candidates = (
        config.min_central_anchor_candidates
        if min_central_anchor_candidates is None
        else min_central_anchor_candidates
    )
    max_central_anchor_share = (
        config.max_central_anchor_share if max_central_anchor_share is None else max_central_anchor_share
    )
    max_central_anchor_candidates = max(
        min_central_anchor_candidates,
        int(round(max_candidates * max_central_anchor_share)),
    )
    min_regional_anchor_candidates = (
        config.min_regional_anchor_candidates
        if min_regional_anchor_candidates is None
        else min_regional_anchor_candidates
    )
    min_directional_anchor_candidates_per_sector = (
        config.min_directional_anchor_candidates_per_sector
        if min_directional_anchor_candidates_per_sector is None
        else min_directional_anchor_candidates_per_sector
    )
    min_octant_anchor_candidates_per_sector = config.min_octant_anchor_candidates_per_sector

    rows = []
    for idx, row in demand.iterrows():
        rows.append(
            {
                "source": "demand",
                "source_id": str(idx),
                "name": row.get("name", f"demand_{idx}"),
                "base_candidate_weight": float(row.get(weight_col, 0.0)),
                "required": False,
                "geometry": row.geometry,
            }
        )

    if regional_centres is not None and not regional_centres.empty:
        regional_centres = _align_crs(regional_centres, demand.crs)
        for idx, row in regional_centres.iterrows():
            rows.append(
                {
                    "source": "regional_centre",
                    "source_id": str(row.get("centre_id", idx)),
                    "name": row.get("name", f"regional_{idx}"),
                    "base_candidate_weight": float(row.get("demand_weight", 0.0)),
                    "required": False,
                    "geometry": row.geometry,
                }
            )

    if centres is not None and not centres.empty:
        centres = _align_crs(centres, demand.crs)
        for idx, row in centres.iterrows():
            rows.append(
                {
                    "source": "forced_centre",
                    "source_id": str(idx),
                    "name": row.get("name", f"centre_{idx}"),
                    "base_candidate_weight": float(row.get("candidate_weight", 0.0)),
                    "required": bool(row.get("role", "") == "required_city_centre"),
                    "geometry": row.geometry,
                }
            )

    candidates = gpd.GeoDataFrame(rows, geometry="geometry", crs=demand.crs)
    if candidates.empty:
        return candidates
    candidates["_candidate_uid"] = np.arange(len(candidates))

    original_geometries = list(candidates.geometry)
    safe_geometries: list[Point] = []
    relocated: list[float] = []
    in_flood_zone: list[bool] = []
    in_flood_buffer: list[bool] = []
    distance_to_flood: list[float] = []
    in_water_zone: list[bool] = []
    in_water_buffer: list[bool] = []
    distance_to_water: list[float] = []

    if (
        forbidden_geom is not None
        and not forbidden_geom.is_empty
        or water_geom is not None
        and not water_geom.is_empty
    ):
        avoidance_parts = []
        flood_avoidance_geom = None
        water_avoidance_geom = None
        if (
            forbidden_geom is not None
            and not forbidden_geom.is_empty
            and not config.flood_zones_are_cost_only
        ):
            flood_avoidance_geom = forbidden_geom.buffer(float(risk_buffer_m))
            avoidance_parts.append(flood_avoidance_geom)
        if water_geom is not None and not water_geom.is_empty:
            water_avoidance_geom = water_geom.buffer(float(config.station_water_buffer_m))
            avoidance_parts.append(water_avoidance_geom)
        avoidance_geom = unary_union(avoidance_parts)
        for point in original_geometries:
            inside_zone = bool(forbidden_geom.covers(point)) if forbidden_geom is not None and not forbidden_geom.is_empty else False
            inside_water = bool(water_geom.covers(point)) if water_geom is not None and not water_geom.is_empty else False
            inside_buffer = bool(flood_avoidance_geom.covers(point)) if flood_avoidance_geom is not None else False
            inside_water_buffer = bool(water_avoidance_geom.covers(point)) if water_avoidance_geom is not None else False
            distance_to_flood.append(
                0.0 if inside_zone else float(point.distance(forbidden_geom))
                if forbidden_geom is not None and not forbidden_geom.is_empty
                else float("inf")
            )
            distance_to_water.append(
                0.0 if inside_water else float(point.distance(water_geom))
                if water_geom is not None and not water_geom.is_empty
                else float("inf")
            )
            if avoidance_geom is not None and not avoidance_geom.is_empty:
                new_point, offset = nearest_free_point_from_boundary(
                    point,
                    avoidance_geom,
                    step_m=config.relocation_step_m,
                    max_extra_m=config.relocation_search_radius_m,
                )
            else:
                new_point, offset = point, 0.0
            safe_geometries.append(new_point)
            relocated.append(offset)
            in_flood_zone.append(inside_zone)
            in_flood_buffer.append(inside_buffer)
            in_water_zone.append(inside_water)
            in_water_buffer.append(inside_water_buffer)
    else:
        safe_geometries = original_geometries
        relocated = [0.0] * len(candidates)
        in_flood_zone = [False] * len(candidates)
        in_flood_buffer = [False] * len(candidates)
        distance_to_flood = [float("inf")] * len(candidates)
        in_water_zone = [False] * len(candidates)
        in_water_buffer = [False] * len(candidates)
        distance_to_water = [float("inf")] * len(candidates)

    candidates["original_geometry"] = original_geometries
    candidates = candidates.set_geometry(gpd.GeoSeries(safe_geometries, index=candidates.index, crs=demand.crs))
    candidates["anchor_relocated_m"] = relocated
    candidates["in_flood_zone"] = in_flood_zone
    candidates["in_flood_buffer"] = in_flood_buffer
    candidates["distance_to_flood_m"] = distance_to_flood
    candidates["in_water_zone"] = in_water_zone
    candidates["in_water_buffer"] = in_water_buffer
    candidates["distance_to_water_m"] = distance_to_water

    catchment = candidate_catchment_weights(list(candidates.geometry), demand, catchment_radius_m, weight_col=weight_col)
    candidates["catchment_weight"] = catchment
    candidates["base_candidate_weight"] = pd.to_numeric(candidates["base_candidate_weight"], errors="coerce").fillna(0.0)
    candidates["risk_safety_factor"] = 1.0
    candidates.loc[candidates["anchor_relocated_m"].fillna(0.0) > 0.0, "risk_safety_factor"] = 0.88
    still_risky = candidates.geometry.apply(lambda geom: bool(forbidden_geom.covers(geom)) if forbidden_geom is not None and not forbidden_geom.is_empty else False)
    if not config.flood_zones_are_cost_only:
        candidates.loc[still_risky, "risk_safety_factor"] = 0.20

    regional_floor = candidates["base_candidate_weight"] * config.regional_anchor_weight_floor_fraction
    source_weight = candidates["base_candidate_weight"].where(candidates["source"].ne("regional_centre"), regional_floor)
    candidates["candidate_weight_raw"] = np.maximum(candidates["catchment_weight"], source_weight)
    candidates["geology_factor"] = [
        geology_factor_at_point(point, geology) for point in candidates.geometry
    ]
    candidates["high_geology"] = candidates["geology_factor"] >= config.high_geology_factor_threshold
    candidates["geology_excess_factor"] = np.maximum(0.0, candidates["geology_factor"] - 1.0)
    candidates["anchor_geology_penalty"] = (
        candidates["geology_excess_factor"] * config.anchor_geology_penalty_per_excess
        + candidates["high_geology"].astype(float) * config.anchor_high_geology_penalty
    )
    candidates.loc[candidates["required"], "anchor_geology_penalty"] = 0.0
    geology_weight_factor = np.maximum(
        0.15,
        1.0 - candidates["geology_excess_factor"] * config.station_geology_score_factor,
    )
    geology_weight_factor = np.where(candidates["high_geology"], geology_weight_factor * 0.35, geology_weight_factor)
    candidates["geology_weight_factor"] = geology_weight_factor
    candidates["candidate_weight"] = (
        candidates["candidate_weight_raw"]
        * candidates["risk_safety_factor"]
        * candidates["geology_weight_factor"]
        - candidates["anchor_geology_penalty"]
    )
    candidates["candidate_weight"] = candidates["candidate_weight"].clip(lower=0.0)

    required = candidates[candidates["required"]]
    if not required.empty:
        required_centre = required.geometry.iloc[0]
        candidates["distance_to_required_centre_m"] = candidates.geometry.distance(required_centre)
        candidates["near_required_centre"] = candidates["distance_to_required_centre_m"] <= central_anchor_radius_m
        dx = candidates.geometry.x - required_centre.x
        dy = candidates.geometry.y - required_centre.y
        candidates["direction_octant"] = [
            direction_octant_from_delta(float(delta_x), float(delta_y))
            for delta_x, delta_y in zip(dx, dy)
        ]
        candidates["direction_sector"] = [
            broad_direction_from_octant(octant)
            for octant in candidates["direction_octant"]
        ]
    else:
        candidates["distance_to_required_centre_m"] = np.nan
        candidates["near_required_centre"] = False
        candidates["direction_sector"] = "unknown"
        candidates["direction_octant"] = "unknown"

    candidates = candidates.sort_values(["required", "candidate_weight"], ascending=[False, False])

    kept_rows: list[pd.Series] = []
    kept_points: list[Point] = []

    def keep_if_possible(row: pd.Series, force: bool = False) -> bool:
        point = row.geometry
        if any(point.distance(existing) < min_separation_m for existing in kept_points):
            if not force and not row["required"]:
                return False
        kept_rows.append(row)
        kept_points.append(point)
        return True

    for _, row in candidates[candidates["required"]].iterrows():
        keep_if_possible(row, force=True)

    regional_kept = 0
    if min_regional_anchor_candidates > 0:
        regional_pool = candidates[
            candidates["source"].eq("regional_centre")
            & ~candidates["required"]
        ].sort_values("candidate_weight", ascending=False)
        for _, row in regional_pool.iterrows():
            if keep_if_possible(row):
                regional_kept += 1
            if regional_kept >= min_regional_anchor_candidates:
                break

    if min_directional_anchor_candidates_per_sector > 0 and "direction_sector" in candidates.columns:
        for sector in ["south", "north", "east", "west"]:
            sector_kept = sum(
                (not item["required"]) and item.get("direction_sector") == sector
                for item in kept_rows
            )
            if sector_kept >= min_directional_anchor_candidates_per_sector:
                continue
            sector_pool = candidates[
                candidates["direction_sector"].eq(sector)
                & ~candidates["required"]
            ].sort_values("candidate_weight", ascending=False)
            for _, row in sector_pool.iterrows():
                if row["_candidate_uid"] in {item["_candidate_uid"] for item in kept_rows}:
                    continue
                if keep_if_possible(row):
                    sector_kept += 1
                if sector_kept >= min_directional_anchor_candidates_per_sector:
                    break

    if min_octant_anchor_candidates_per_sector > 0 and "direction_octant" in candidates.columns:
        for sector in ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]:
            sector_kept = sum(
                (not item["required"]) and item.get("direction_octant") == sector
                for item in kept_rows
            )
            if sector_kept >= min_octant_anchor_candidates_per_sector:
                continue
            sector_pool = candidates[
                candidates["direction_octant"].eq(sector)
                & ~candidates["required"]
            ].sort_values("candidate_weight", ascending=False)
            for _, row in sector_pool.iterrows():
                if row["_candidate_uid"] in {item["_candidate_uid"] for item in kept_rows}:
                    continue
                if keep_if_possible(row):
                    sector_kept += 1
                if sector_kept >= min_octant_anchor_candidates_per_sector:
                    break

    central_kept = 0
    if min_central_anchor_candidates > 0 and candidates["near_required_centre"].any():
        central_pool = candidates[
            candidates["near_required_centre"]
            & candidates["source"].eq("demand")
            & ~candidates["required"]
        ].sort_values("candidate_weight", ascending=False)
        for _, row in central_pool.iterrows():
            if keep_if_possible(row):
                central_kept += 1
            if central_kept >= min_central_anchor_candidates:
                break

    for _, row in candidates[~candidates["required"]].iterrows():
        if len(kept_rows) >= max_candidates + len(candidates[candidates["required"]]):
            break
        if row["_candidate_uid"] in {item["_candidate_uid"] for item in kept_rows}:
            continue
        is_central_demand = bool(row.get("near_required_centre", False)) and row.get("source") == "demand"
        if is_central_demand and central_kept >= max_central_anchor_candidates:
            continue
        if not keep_if_possible(row):
            continue
        if is_central_demand:
            central_kept += 1
        optional_count = sum(not item["required"] for item in kept_rows)
        if optional_count >= max_candidates:
            break

    out = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=demand.crs).reset_index(drop=True)
    out["candidate_id"] = [f"C{i:03d}" for i in range(1, len(out) + 1)]
    return out


def route_length_m(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(points[idx].distance(points[idx + 1]) for idx in range(len(points) - 1)))


def _empty_turn_metrics() -> dict:
    return {
        "turn_penalty": 0.0,
        "max_turn_angle_deg": 0.0,
        "mean_turn_angle_deg": 0.0,
        "sharp_turn_count": 0,
        "curve_radius_violation_m": 0.0,
    }


def _empty_corridor_shape_metrics() -> dict:
    return {
        "corridor_detour_ratio": 1.0,
        "corridor_backtrack_km": 0.0,
        "corridor_shape_penalty": 0.0,
    }


def _turn_metrics_from_xy(xy: np.ndarray, config: MetroConfig) -> dict:
    """Approximate alignment feasibility from polyline deflection angles."""

    if len(xy) < 3:
        return _empty_turn_metrics()

    incoming = xy[1:-1] - xy[:-2]
    outgoing = xy[2:] - xy[1:-1]
    incoming_lengths = np.linalg.norm(incoming, axis=1)
    outgoing_lengths = np.linalg.norm(outgoing, axis=1)
    valid = (incoming_lengths > 0.0) & (outgoing_lengths > 0.0)
    if not np.any(valid):
        return _empty_turn_metrics()

    incoming = incoming[valid]
    outgoing = outgoing[valid]
    incoming_lengths = incoming_lengths[valid]
    outgoing_lengths = outgoing_lengths[valid]
    cosines = np.sum(incoming * outgoing, axis=1) / (incoming_lengths * outgoing_lengths)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    excess_angles = np.maximum(0.0, angles - config.max_turn_angle_deg)
    angle_penalty = float(excess_angles.sum() * config.turn_penalty_per_degree)

    radius_violation = 0.0
    if config.minimum_curve_radius_m > 0.0:
        required_tangent = config.minimum_curve_radius_m * np.tan(np.radians(angles) / 2.0)
        available_tangent = 0.5 * np.minimum(incoming_lengths, outgoing_lengths)
        radius_violation = float(np.maximum(0.0, required_tangent - available_tangent).sum())

    radius_penalty = radius_violation * config.curve_radius_penalty_per_m
    return {
        "turn_penalty": angle_penalty + radius_penalty,
        "max_turn_angle_deg": float(angles.max()),
        "mean_turn_angle_deg": float(angles.mean()),
        "sharp_turn_count": int((angles > config.max_turn_angle_deg).sum()),
        "curve_radius_violation_m": radius_violation,
    }


def route_turn_metrics(points: list[Point], config: MetroConfig) -> dict:
    if len(points) < 3:
        return _empty_turn_metrics()
    xy = np.array([(point.x, point.y) for point in points], dtype=float)
    return _turn_metrics_from_xy(xy, config)


def _corridor_shape_metrics_from_xy(xy: np.ndarray, config: MetroConfig) -> dict:
    """Penalize TSP-like detours and backtracking along the line's main axis."""

    if len(xy) < 3:
        return _empty_corridor_shape_metrics()

    segments = xy[1:] - xy[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    route_length = float(segment_lengths.sum())
    axis = xy[-1] - xy[0]
    end_to_end = float(np.linalg.norm(axis))
    if route_length <= 0.0 or end_to_end <= 0.0:
        return _empty_corridor_shape_metrics()

    detour_ratio = route_length / end_to_end
    detour_excess = max(0.0, detour_ratio - config.corridor_detour_ratio_limit)
    detour_penalty = detour_excess * config.corridor_detour_penalty_per_ratio

    unit_axis = axis / end_to_end
    projections = (xy - xy[0]) @ unit_axis
    projection_steps = np.diff(projections)
    backtrack_m = float(np.maximum(0.0, -projection_steps - config.corridor_backtrack_tolerance_m).sum())
    backtrack_km = backtrack_m / 1_000.0
    backtrack_penalty = backtrack_km * config.corridor_backtrack_penalty_per_km

    return {
        "corridor_detour_ratio": detour_ratio,
        "corridor_backtrack_km": backtrack_km,
        "corridor_shape_penalty": detour_penalty + backtrack_penalty,
    }


def corridor_shape_metrics(points: list[Point], config: MetroConfig) -> dict:
    if len(points) < 3:
        return _empty_corridor_shape_metrics()
    xy = np.array([(point.x, point.y) for point in points], dtype=float)
    return _corridor_shape_metrics_from_xy(xy, config)


def route_polyline(points: list[Point]) -> LineString:
    if not points:
        raise ValueError("Route must contain at least one point.")
    coords = [(point.x, point.y) for point in points]
    if len(coords) == 1:
        coords = [coords[0], coords[0]]
    return LineString(coords)


def _extension_capacity_for_unit_m(endpoint: Point, unit: np.ndarray, max_extra_m: float, city_area) -> float:
    def extension_segment(extra_m: float) -> LineString:
        return LineString(
            [
                (endpoint.x, endpoint.y),
                (endpoint.x + unit[0] * extra_m, endpoint.y + unit[1] * extra_m),
            ]
        )

    if extension_segment(max_extra_m).difference(city_area).length <= 1.0:
        return max_extra_m

    low = 0.0
    high = max_extra_m
    for _ in range(24):
        mid = (low + high) / 2.0
        if extension_segment(mid).difference(city_area).length <= 1.0:
            low = mid
        else:
            high = mid
    return low


def _endpoint_extension_choice(
    endpoint: Point,
    neighbour: Point,
    max_extra_m: float,
    city_area,
    flexible_terminal: bool = False,
) -> tuple[float, np.ndarray]:
    """Choose a terminal extension direction that keeps as much length as possible inside the city."""

    vector = np.array([endpoint.x - neighbour.x, endpoint.y - neighbour.y], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return 0.0, np.array([0.0, 0.0], dtype=float)
    base_unit = vector / norm
    if city_area is None or city_area.is_empty or max_extra_m <= 0.0:
        return max(0.0, max_extra_m), base_unit
    if not flexible_terminal:
        return _extension_capacity_for_unit_m(endpoint, base_unit, max_extra_m, city_area), base_unit

    best_capacity = -1.0
    best_unit = base_unit
    best_rotation = 180.0
    for angle in [0, -15, 15, -30, 30, -45, 45]:
        theta = radians(float(angle))
        rotation = np.array(
            [
                [cos(theta), -sin(theta)],
                [sin(theta), cos(theta)],
            ],
            dtype=float,
        )
        unit = rotation @ base_unit
        capacity = _extension_capacity_for_unit_m(endpoint, unit, max_extra_m, city_area)
        if (capacity, -abs(angle)) > (best_capacity, -best_rotation):
            best_capacity = capacity
            best_unit = unit
            best_rotation = abs(angle)
    return max(0.0, best_capacity), best_unit


def extend_route_points_to_length(
    points: list[Point],
    target_length_m: float,
    city_area=None,
    flexible_terminal: bool = False,
) -> list[Point]:
    """Extend open-route endpoints so the corridor uses the full length budget."""

    if not points:
        raise ValueError("Route must contain at least one point.")

    current_length = route_length_m(points)
    if current_length >= target_length_m:
        return list(points)

    deficit = target_length_m - current_length
    if len(points) == 1 or current_length == 0:
        centre = points[0]
        half = target_length_m / 2.0
        if city_area is not None and not city_area.is_empty:
            best_points = None
            best_outside_m = float("inf")
            for angle in np.arange(0.0, 180.0, 5.0):
                unit = np.array([cos(radians(float(angle))), sin(radians(float(angle)))], dtype=float)
                trial_points = [
                    Point(centre.x - unit[0] * half, centre.y - unit[1] * half),
                    centre,
                    Point(centre.x + unit[0] * half, centre.y + unit[1] * half),
                ]
                trial_line = route_polyline(trial_points)
                outside_m = float(trial_line.difference(city_area).length)
                if outside_m < best_outside_m:
                    best_outside_m = outside_m
                    best_points = trial_points
            if best_points is not None:
                return best_points
        return [Point(centre.x - half, centre.y), centre, Point(centre.x + half, centre.y)]

    first, second = points[0], points[1]
    last, before_last = points[-1], points[-2]

    def extend_endpoint(endpoint: Point, unit: np.ndarray, extra_m: float) -> Point:
        if float(np.linalg.norm(unit)) == 0.0:
            return endpoint
        return Point(endpoint.x + unit[0] * extra_m, endpoint.y + unit[1] * extra_m)

    first_extra = deficit / 2.0
    last_extra = deficit / 2.0
    first_capacity, first_unit = _endpoint_extension_choice(
        first,
        second,
        deficit,
        city_area,
        flexible_terminal=flexible_terminal,
    )
    last_capacity, last_unit = _endpoint_extension_choice(
        last,
        before_last,
        deficit,
        city_area,
        flexible_terminal=flexible_terminal,
    )
    if city_area is not None and not city_area.is_empty:
        first_extra = min(first_extra, first_capacity)
        last_extra = min(last_extra, last_capacity)
        remaining = deficit - first_extra - last_extra
        for side in sorted(
            [
                ("first", first_capacity - first_extra),
                ("last", last_capacity - last_extra),
            ],
            key=lambda item: item[1],
            reverse=True,
        ):
            if remaining <= 0.0:
                break
            extra = min(remaining, max(0.0, side[1]))
            if side[0] == "first":
                first_extra += extra
            else:
                last_extra += extra
            remaining -= extra
        if remaining > 0.0:
            first_extra += remaining / 2.0
            last_extra += remaining - remaining / 2.0

    extended = []
    if first_extra > 1e-6:
        extended.append(extend_endpoint(first, first_unit, first_extra))
    extended.extend(points)
    if last_extra > 1e-6:
        extended.append(extend_endpoint(last, last_unit, last_extra))
    return extended


def best_single_anchor_corridor_points(
    centre: Point,
    config: MetroConfig,
    city_area=None,
    existing_lines: gpd.GeoDataFrame | None = None,
) -> list[Point]:
    """Fallback full-length line through one forced centre with minimal spatial conflicts."""

    half = config.length_m / 2.0
    best_points: list[Point] | None = None
    best_key: tuple[float, float, float, float] | None = None
    step = max(1.0, float(config.angle_step_deg))
    for angle in np.arange(0.0, 180.0, step):
        unit = np.array([cos(radians(float(angle))), sin(radians(float(angle)))], dtype=float)
        points = [
            Point(centre.x - unit[0] * half, centre.y - unit[1] * half),
            centre,
            Point(centre.x + unit[0] * half, centre.y + unit[1] * half),
        ]
        line = route_polyline(points)
        outside_km = float(line.difference(city_area).length / 1_000.0) if city_area is not None and not city_area.is_empty else 0.0
        overlap_km = line_overlap_km(points, existing_lines, config)
        penalties = spatial_limit_penalties(outside_km, overlap_km, config)
        key = (
            penalties["outside_city_excess_km"],
            penalties["line_overlap_excess_km"],
            outside_km,
            overlap_km,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_points = points
    if best_points is None:
        return extend_route_points_to_length([centre], config.length_m, city_area)
    return best_points


def route_forbidden_km(points: list[Point], forbidden: gpd.GeoDataFrame | None) -> float:
    union = forbidden_union(forbidden)
    if union is None or union.is_empty or len(points) < 2:
        return 0.0
    return float(route_polyline(points).intersection(union).length / 1_000.0)


def line_outside_city_km(line: LineString, city_boundary: gpd.GeoDataFrame | None) -> float:
    city_area = forbidden_union(city_boundary)
    if city_area is None or city_area.is_empty or line.is_empty:
        return 0.0
    return float(line.difference(city_area).length / 1_000.0)


def route_outside_city_km(points: list[Point], city_boundary: gpd.GeoDataFrame | None) -> float:
    if len(points) < 2:
        return 0.0
    return line_outside_city_km(route_polyline(points), city_boundary)


def route_geology_factor(points: list[Point], geology: gpd.GeoDataFrame | None) -> float:
    if len(points) < 2:
        return 1.0
    return geology_multiplier_for_line(route_polyline(points), geology)


def route_geology_excess_km(points: list[Point], geology: gpd.GeoDataFrame | None) -> float:
    if len(points) < 2:
        return 0.0
    return geology_excess_km_for_line(route_polyline(points), geology)


def route_high_geology_km(points: list[Point], geology: gpd.GeoDataFrame | None, config: MetroConfig) -> float:
    if len(points) < 2:
        return 0.0
    return geology_high_factor_km_for_line(
        route_polyline(points),
        geology,
        threshold=config.high_geology_factor_threshold,
    )


def route_geology_point_penalty(
    points: list[Point],
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    penalty_per_excess: float,
    high_penalty: float,
) -> float:
    return geology_point_penalty(
        points,
        geology,
        config,
        penalty_per_excess=penalty_per_excess,
        high_penalty=high_penalty,
    )


def water_crossing_km(points: list[Point], water_crossings: gpd.GeoDataFrame | None) -> float:
    union = forbidden_union(water_crossings)
    if union is None or union.is_empty or len(points) < 2:
        return 0.0
    return float(route_polyline(points).intersection(union).length / 1_000.0)


def _iter_line_parts(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
        return
    if geom.geom_type == "MultiLineString":
        yield from geom.geoms


def _line_segments_with_angles(geometries: Iterable) -> list[tuple[LineString, float]]:
    segments: list[tuple[LineString, float]] = []
    for geom in geometries:
        for line in _iter_line_parts(geom) or []:
            coords = list(line.coords)
            for start, end in zip(coords[:-1], coords[1:]):
                segment = LineString([start, end])
                if segment.length <= 0.0:
                    continue
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                segments.append((segment, degrees(atan2(dy, dx))))
    return segments


def _acute_angle_difference_deg(left: float, right: float) -> float:
    diff = abs((left - right + 180.0) % 360.0 - 180.0)
    return min(diff, 180.0 - diff)


def _angle_from_xy(origin: np.ndarray, target: np.ndarray) -> float | None:
    vector = target - origin
    if float(np.linalg.norm(vector)) <= 0.0:
        return None
    return degrees(atan2(float(vector[1]), float(vector[0])))


def _existing_line_arm_angles(geometries: Iterable, centre_xy: np.ndarray, min_distance_m: float = 1_000.0) -> list[float]:
    angles: list[float] = []
    for geom in geometries:
        for line in _iter_line_parts(geom) or []:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for coord in (coords[0], coords[-1]):
                target = np.array([coord[0], coord[1]], dtype=float)
                if float(np.linalg.norm(target - centre_xy)) < min_distance_m:
                    continue
                angle = _angle_from_xy(centre_xy, target)
                if angle is not None:
                    angles.append(angle)
    return angles


def _arm_reuse_penalty(order: list[int], node_xy: np.ndarray, existing_arm_angles: list[float], config: MetroConfig) -> float:
    if len(order) < 2 or not existing_arm_angles or config.arm_reuse_angle_deg <= 0.0:
        return 0.0
    centre_xy = node_xy[0]
    endpoint_indices = [order[0], order[-1]]
    penalty = 0.0
    for node_index in endpoint_indices:
        if node_index == 0:
            continue
        angle = _angle_from_xy(centre_xy, node_xy[node_index])
        if angle is None:
            continue
        nearest = min(_acute_angle_difference_deg(angle, existing_angle) for existing_angle in existing_arm_angles)
        if nearest < config.arm_reuse_angle_deg:
            penalty += (1.0 - nearest / config.arm_reuse_angle_deg) * config.arm_reuse_penalty
    return penalty


def _parallel_overlap_length_m(
    line: LineString,
    overlap_zone,
    existing_segments: list[tuple[LineString, float]],
    config: MetroConfig,
) -> float:
    """Measure only near-parallel overlap, not ordinary line crossings."""

    if line.is_empty or overlap_zone is None or overlap_zone.is_empty or not existing_segments:
        return 0.0
    total = 0.0
    for segment, angle in _line_segments_with_angles([line]):
        inside_length = float(segment.intersection(overlap_zone).length)
        if inside_length <= 0.0:
            continue
        parallel = any(
            segment.distance(previous_segment) <= config.parallel_line_buffer_m
            and _acute_angle_difference_deg(angle, previous_angle) <= config.parallel_overlap_angle_deg
            for previous_segment, previous_angle in existing_segments
        )
        if parallel:
            total += inside_length
    return total


def line_overlap_km(
    points: list[Point],
    existing_lines: gpd.GeoDataFrame | None,
    config: MetroConfig,
) -> float:
    """Length of a route that runs near-parallel to an existing metro corridor."""

    if existing_lines is None or existing_lines.empty or len(points) < 2:
        return 0.0
    corridor = route_polyline(points)
    existing_geoms = [geom for geom in existing_lines.geometry if geom is not None and not geom.is_empty]
    existing_union = unary_union(existing_geoms)
    if existing_union.is_empty:
        return 0.0
    overlap_zone = existing_union.buffer(config.parallel_line_buffer_m)
    existing_segments = _line_segments_with_angles(existing_geoms)
    return float(_parallel_overlap_length_m(corridor, overlap_zone, existing_segments, config) / 1_000.0)


def line_overlap_buffer_km(
    points: list[Point],
    existing_lines: gpd.GeoDataFrame | None,
    config: MetroConfig,
) -> float:
    """Raw buffer overlap, useful for diagnostics when crossings are included."""

    if existing_lines is None or existing_lines.empty or len(points) < 2:
        return 0.0
    existing_geoms = [geom for geom in existing_lines.geometry if geom is not None and not geom.is_empty]
    existing_union = unary_union(existing_geoms)
    if existing_union.is_empty:
        return 0.0
    overlap_zone = existing_union.buffer(config.parallel_line_buffer_m)
    return float(route_polyline(points).intersection(overlap_zone).length / 1_000.0)


def spatial_limit_penalties(outside_city_km: float, line_overlap_km_value: float, config: MetroConfig) -> dict:
    outside_excess = max(0.0, outside_city_km - config.max_outside_city_km)
    overlap_excess = max(0.0, line_overlap_km_value - config.max_line_overlap_km)
    return {
        "outside_city_excess_km": outside_excess,
        "outside_city_excess_penalty": outside_excess * config.outside_city_excess_penalty_per_km,
        "line_overlap_excess_km": overlap_excess,
        "line_overlap_excess_penalty": overlap_excess * config.line_overlap_excess_penalty_per_km,
    }


def forbidden_limit_penalties(forbidden_km: float, config: MetroConfig) -> dict:
    if config.max_forbidden_km is None:
        excess = 0.0
    else:
        excess = max(0.0, forbidden_km - float(config.max_forbidden_km))
    return {
        "forbidden_excess_km": excess,
        "forbidden_excess_penalty": excess * config.forbidden_excess_penalty_per_km,
    }


def transfer_interchange_score(
    points: list[Point],
    transfer_points: gpd.GeoDataFrame | None,
    transfer_lines: gpd.GeoDataFrame | None,
    config: MetroConfig,
) -> tuple[float, int]:
    """Score useful interchanges with an existing metro line."""

    transfer_count = 0
    if transfer_points is not None and not transfer_points.empty and points:
        transfer_geoms = list(transfer_points.geometry)
        near_existing = 0
        for point in points:
            if any(point.distance(existing) <= config.interchange_radius_m for existing in transfer_geoms):
                near_existing += 1
        transfer_count += min(2, near_existing)

    if transfer_lines is not None and not transfer_lines.empty and len(points) >= 2:
        line = route_polyline(points)
        for previous_line in transfer_lines.geometry:
            if line.crosses(previous_line) or line.intersects(previous_line):
                transfer_count += 1
        transfer_count = min(3, transfer_count)

    return float(transfer_count * config.transfer_bonus_per_interchange), transfer_count


def route_score_from_points(
    points: list[Point],
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    weight_col: str = "population",
    water_crossings: gpd.GeoDataFrame | None = None,
    transfer_points: gpd.GeoDataFrame | None = None,
    transfer_lines: gpd.GeoDataFrame | None = None,
    existing_lines: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
) -> dict:
    """Score an ordered station/anchor sequence.

    This objective combines maximum coverage, a length budget, and route penalties.
    The selection/order problem is a prize-collecting TSP / orienteering variant.
    """

    if not points:
        return {
            "score": 0.0,
            "served_weight": 0.0,
            "served_share": 0.0,
            "route_length_m": 0.0,
            "forbidden_km": 0.0,
            "forbidden_excess_km": 0.0,
            "forbidden_excess_penalty": 0.0,
            "geology_factor": 1.0,
            "geology_excess_km": 0.0,
            "high_geology_km": 0.0,
            "station_geology_penalty": 0.0,
            "water_crossing_km": 0.0,
            "transfer_score": 0.0,
            "transfer_count": 0,
            "line_overlap_km": 0.0,
            "line_overlap_excess_km": 0.0,
            "line_overlap_excess_penalty": 0.0,
            "arm_reuse_penalty": 0.0,
            "direction_priority_bonus": 0.0,
            "direction_priority_sector_count": 0,
            "outside_city_km": 0.0,
            "outside_city_penalty": 0.0,
            "outside_city_excess_km": 0.0,
            "outside_city_excess_penalty": 0.0,
            **_empty_turn_metrics(),
            **_empty_corridor_shape_metrics(),
        }

    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    station_xy = np.array([(point.x, point.y) for point in points], dtype=float)
    distances = np.sqrt(((demand_xy[:, None, :] - station_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    coverage_score = np.clip(1.0 - distances / config.walk_radius_m, 0.0, 1.0)
    weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    total_weight = float(weights.sum())
    served_weight = float(np.sum(weights * coverage_score))
    forbidden_km = route_forbidden_km(points, forbidden)
    forbidden_limit = forbidden_limit_penalties(forbidden_km, config)
    route_km = route_length_m(points) / 1_000.0
    geology_excess_km = route_geology_excess_km(points, geology)
    geology_factor = 1.0 + geology_excess_km / route_km if route_km else 1.0
    high_geology_km = route_high_geology_km(points, geology, config)
    station_geology_penalty = route_geology_point_penalty(
        points,
        geology,
        config,
        penalty_per_excess=config.station_geology_penalty_per_excess,
        high_penalty=config.station_high_geology_penalty,
    )
    water_km = water_crossing_km(points, water_crossings)
    transfer_score, transfer_count = transfer_interchange_score(points, transfer_points, transfer_lines, config)
    overlap_km = line_overlap_km(points, existing_lines if existing_lines is not None else transfer_lines, config)
    outside_city_km = route_outside_city_km(points, city_boundary)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km
    limit_penalties = spatial_limit_penalties(outside_city_km, overlap_km, config)
    turn_metrics = route_turn_metrics(points, config)
    shape_metrics = corridor_shape_metrics(points, config)
    score = served_weight
    score -= forbidden_km * config.forbidden_penalty_per_km
    score -= forbidden_limit["forbidden_excess_penalty"]
    score -= outside_city_penalty
    score -= geology_excess_km * config.geology_penalty_per_km
    score -= high_geology_km * config.high_geology_penalty_per_km
    score -= station_geology_penalty
    score -= overlap_km * config.line_overlap_penalty_per_km
    score -= limit_penalties["outside_city_excess_penalty"]
    score -= limit_penalties["line_overlap_excess_penalty"]
    score -= turn_metrics["turn_penalty"]
    score -= shape_metrics["corridor_shape_penalty"]
    score += water_km * config.river_crossing_bonus_per_km
    score += transfer_score
    return {
        "score": score,
        "served_weight": served_weight,
        "served_share": served_weight / total_weight if total_weight else 0.0,
        "route_length_m": route_length_m(points),
        "forbidden_km": forbidden_km,
        "forbidden_excess_km": forbidden_limit["forbidden_excess_km"],
        "forbidden_excess_penalty": forbidden_limit["forbidden_excess_penalty"],
        "geology_factor": geology_factor,
        "geology_excess_km": geology_excess_km,
        "high_geology_km": high_geology_km,
        "station_geology_penalty": station_geology_penalty,
        "water_crossing_km": water_km,
        "transfer_score": transfer_score,
        "transfer_count": transfer_count,
        "line_overlap_km": overlap_km,
        "line_overlap_excess_km": limit_penalties["line_overlap_excess_km"],
        "line_overlap_excess_penalty": limit_penalties["line_overlap_excess_penalty"],
        "outside_city_km": outside_city_km,
        "outside_city_penalty": outside_city_penalty,
        "outside_city_excess_km": limit_penalties["outside_city_excess_km"],
        "outside_city_excess_penalty": limit_penalties["outside_city_excess_penalty"],
        **turn_metrics,
        **shape_metrics,
    }


def construction_cost_mln(
    length_km: float,
    forbidden_km: float,
    geology_factor: float,
    config: MetroConfig,
) -> tuple[float, float, float]:
    """Estimate line construction cost with flood-risk segments priced higher."""

    base_cost = length_km * config.cost_per_km_mln * geology_factor
    flood_extra = max(0.0, forbidden_km) * config.cost_per_km_mln * max(0.0, config.flood_cost_multiplier - 1.0)
    return base_cost + flood_extra, base_cost, flood_extra


def station_candidate_scores_along_line(
    line: LineString,
    distances: np.ndarray,
    demand: gpd.GeoDataFrame,
    forbidden_geom,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    water_geom=None,
    weight_col: str = "population",
    route_anchor_points: list[Point] | None = None,
) -> np.ndarray:
    """Score possible station locations along a corridor by nearby demand and risk."""

    points = [line.interpolate(float(distance)) for distance in distances]
    scores = candidate_catchment_weights(points, demand, config.walk_radius_m, weight_col=weight_col)

    if route_anchor_points and config.station_anchor_bonus > 0.0 and config.station_anchor_bonus_radius_m > 0.0:
        anchor_distances = np.array([line.project(point) for point in route_anchor_points], dtype=float)
        if len(anchor_distances):
            nearest_along_line = np.abs(distances[:, None] - anchor_distances[None, :]).min(axis=1)
            anchor_bonus = np.clip(1.0 - nearest_along_line / config.station_anchor_bonus_radius_m, 0.0, 1.0)
            scores = scores + anchor_bonus * config.station_anchor_bonus

    if (
        forbidden_geom is not None
        and not forbidden_geom.is_empty
        and not config.flood_zones_are_cost_only
    ):
        risk_zone = forbidden_geom.buffer(config.station_risk_buffer_m)
        risk_factor = np.array(
            [
                config.station_flood_score_factor if risk_zone.covers(point) else 1.0
                for point in points
            ],
            dtype=float,
        )
        scores = scores * risk_factor
        in_flood_zone = np.array([forbidden_geom.covers(point) for point in points], dtype=bool)
        scores = np.where(in_flood_zone, -1_000_000_000.0, scores)
    if water_geom is not None and not water_geom.is_empty:
        water_zone = water_geom.buffer(config.station_water_buffer_m)
        water_factor = np.array(
            [
                config.station_water_score_factor if water_zone.covers(point) else 1.0
                for point in points
            ],
            dtype=float,
        )
        scores = scores * water_factor
        in_water_zone = np.array([water_geom.covers(point) for point in points], dtype=bool)
        scores = np.where(in_water_zone, -1_000_000_000.0, scores)
    if geology is not None and not geology.empty:
        geology_factors = np.array([geology_factor_at_point(point, geology) for point in points], dtype=float)
        geology_excess = np.maximum(0.0, geology_factors - 1.0)
        geology_factor = np.maximum(0.15, 1.0 - geology_excess * config.station_geology_score_factor)
        geology_factor = np.where(
            geology_factors >= config.high_geology_factor_threshold,
            geology_factor * 0.25,
            geology_factor,
        )
        scores = scores * geology_factor
    return scores


def _station_point_is_unsafe(point: Point, forbidden_geom, water_geom) -> bool:
    if forbidden_geom is not None and not forbidden_geom.is_empty and forbidden_geom.covers(point):
        return True
    if water_geom is not None and not water_geom.is_empty and water_geom.covers(point):
        return True
    return False


def repair_station_distances_along_line(
    line: LineString,
    chosen_distances: np.ndarray,
    candidate_distances: np.ndarray,
    forbidden_geom,
    water_geom,
    config: MetroConfig,
) -> tuple[np.ndarray, bool]:
    """Move stations out of actual flood/water polygons while preserving spacing."""

    repaired = np.array(chosen_distances, dtype=float).copy()
    changed = False
    for _ in range(3):
        pass_changed = False
        for idx, distance in enumerate(repaired):
            point = line.interpolate(float(distance))
            if not _station_point_is_unsafe(point, forbidden_geom, water_geom):
                continue

            lower = 0.0
            upper = line.length
            if idx > 0:
                lower = max(lower, repaired[idx - 1] + config.station_min_spacing_m)
                upper = min(upper, repaired[idx - 1] + config.station_max_spacing_m)
            if idx < len(repaired) - 1:
                lower = max(lower, repaired[idx + 1] - config.station_max_spacing_m)
                upper = min(upper, repaired[idx + 1] - config.station_min_spacing_m)
            if lower > upper:
                continue

            feasible = candidate_distances[(candidate_distances >= lower) & (candidate_distances <= upper)]
            safe = [
                value
                for value in feasible
                if not _station_point_is_unsafe(line.interpolate(float(value)), forbidden_geom, water_geom)
            ]
            if not safe:
                continue
            new_distance = min(safe, key=lambda value: abs(float(value) - float(distance)))
            if abs(float(new_distance) - float(distance)) > 1e-6:
                repaired[idx] = float(new_distance)
                pass_changed = True
                changed = True
        if not pass_changed:
            break
    return np.sort(repaired), changed


def optimise_station_distances_along_line(
    line: LineString,
    demand: gpd.GeoDataFrame,
    forbidden_geom,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    water_geom=None,
    weight_col: str = "population",
    route_anchor_points: list[Point] | None = None,
) -> tuple[np.ndarray, str]:
    """Choose station positions along the final corridor with spacing constraints.

    Terminal stations may move within a short window from the corridor ends, so
    they can avoid water/flood-risk points while keeping the corridor length.
    Interior stations are selected by dynamic programming from a dense 1D grid.
    """

    if not config.adaptive_station_placement or config.station_count <= 2 or line.length <= 0:
        return np.linspace(0.0, line.length, config.station_count), "uniform"

    grid = np.arange(0.0, line.length + config.station_candidate_step_m, config.station_candidate_step_m)
    grid = grid[(grid > 0.0) & (grid < line.length)]
    distances = np.concatenate(([0.0], grid, [line.length]))
    distances = np.unique(np.round(distances, 6))
    last_index = len(distances) - 1
    station_count = config.station_count

    scores = station_candidate_scores_along_line(
        line,
        distances,
        demand,
        forbidden_geom,
        geology,
        config,
        water_geom=water_geom,
        weight_col=weight_col,
        route_anchor_points=route_anchor_points,
    )
    dp = np.full((station_count, len(distances)), -np.inf)
    prev = np.full((station_count, len(distances)), -1, dtype=int)
    start_window = max(0.0, min(float(config.station_terminal_flex_m), line.length / 3.0))
    end_window_start = line.length - start_window
    start_indices = np.where(distances <= start_window)[0]
    end_indices = np.where(distances >= end_window_start)[0]
    if len(start_indices) == 0:
        start_indices = np.array([0])
    if len(end_indices) == 0:
        end_indices = np.array([last_index])
    dp[0, start_indices] = scores[start_indices]

    for station_idx in range(1, station_count):
        for current in range(1, len(distances)):
            spacing = distances[current] - distances[:current]
            feasible = np.where(
                (spacing >= config.station_min_spacing_m)
                & (spacing <= config.station_max_spacing_m)
                & np.isfinite(dp[station_idx - 1, :current])
            )[0]
            if len(feasible) == 0:
                continue
            best_prev = feasible[np.argmax(dp[station_idx - 1, feasible])]
            dp[station_idx, current] = dp[station_idx - 1, best_prev] + scores[current]
            prev[station_idx, current] = best_prev

    feasible_end_indices = end_indices[np.isfinite(dp[station_count - 1, end_indices])]
    if len(feasible_end_indices) == 0:
        return np.linspace(0.0, line.length, station_count), "uniform_fallback"

    current = int(feasible_end_indices[np.argmax(dp[station_count - 1, feasible_end_indices])])
    chosen = [current]
    for station_idx in range(station_count - 1, 0, -1):
        current = int(prev[station_idx, current])
        if current < 0:
            return np.linspace(0.0, line.length, station_count), "uniform_fallback"
        chosen.append(current)
    chosen = list(reversed(chosen))
    chosen_distances = distances[chosen]
    station_forbidden_geom = None if config.flood_zones_are_cost_only else forbidden_geom
    repaired_distances, repaired = repair_station_distances_along_line(
        line,
        chosen_distances,
        distances,
        station_forbidden_geom,
        water_geom,
        config,
    )
    return repaired_distances, "adaptive_dp_repaired" if repaired else "adaptive_dp"


def stations_for_corridor(
    corridor: LineString,
    demand: gpd.GeoDataFrame,
    forbidden_geom,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    line_id: int,
    weight_col: str = "population",
    water_geom=None,
    route_anchor_points: list[Point] | None = None,
) -> gpd.GeoDataFrame:
    distances, placement = optimise_station_distances_along_line(
        corridor,
        demand,
        forbidden_geom,
        geology,
        config,
        water_geom=water_geom,
        weight_col=weight_col,
        route_anchor_points=route_anchor_points,
    )
    previous_spacing = np.concatenate(([np.nan], np.diff(distances)))
    next_spacing = np.concatenate((np.diff(distances), [np.nan]))
    station_points = [corridor.interpolate(float(distance)) for distance in distances]
    geology_factors = [geology_factor_at_point(point, geology) for point in station_points]
    flood_station_geom = forbidden_geom if forbidden_geom is not None and not forbidden_geom.is_empty else None
    water_station_geom = water_geom if water_geom is not None and not water_geom.is_empty else None
    flood_station_zone = (
        forbidden_geom.buffer(config.station_risk_buffer_m)
        if forbidden_geom is not None and not forbidden_geom.is_empty
        else None
    )
    water_station_zone = (
        water_geom.buffer(config.station_water_buffer_m)
        if water_geom is not None and not water_geom.is_empty
        else None
    )
    if route_anchor_points:
        nearest_anchor_m = [min(point.distance(anchor) for anchor in route_anchor_points) for point in station_points]
    else:
        nearest_anchor_m = [np.nan] * len(station_points)
    return gpd.GeoDataFrame(
        {
            "line_id": line_id,
            "station_id": np.arange(1, config.station_count + 1),
            "station_code": [f"L{line_id:02d}-S{sid:02d}" for sid in range(1, config.station_count + 1)],
            "distance_m": distances,
            "spacing_from_previous_m": previous_spacing,
            "spacing_to_next_m": next_spacing,
            "station_placement": placement,
            "geology_factor": geology_factors,
            "high_geology": [
                bool(factor >= config.high_geology_factor_threshold)
                for factor in geology_factors
            ],
            "in_station_flood_zone": [
                bool(flood_station_geom.covers(point)) if flood_station_geom is not None else False
                for point in station_points
            ],
            "in_station_water_zone": [
                bool(water_station_geom.covers(point)) if water_station_geom is not None else False
                for point in station_points
            ],
            "in_station_flood_buffer": [
                bool(flood_station_zone.covers(point)) if flood_station_zone is not None else False
                for point in station_points
            ],
            "in_station_water_buffer": [
                bool(water_station_zone.covers(point)) if water_station_zone is not None else False
                for point in station_points
            ],
            "nearest_route_anchor_m": nearest_anchor_m,
            "near_route_anchor": [
                bool(distance <= config.station_anchor_bonus_radius_m) if np.isfinite(distance) else False
                for distance in nearest_anchor_m
            ],
        },
        geometry=station_points,
        crs=demand.crs,
    )


def grid_station_candidates(
    demand: gpd.GeoDataFrame,
    city_boundary: gpd.GeoDataFrame | None = None,
    centres: gpd.GeoDataFrame | None = None,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
    grid_step_m: float | None = None,
    max_candidates: int | None = None,
) -> gpd.GeoDataFrame:
    """Create station-level candidates from a regular city grid.

    Unlike ``candidate_station_sites``, this does not compress population into
    regional anchors. Every grid point is scored directly against the full demand
    layer, so the route optimiser chooses station locations from a demand
    surface rather than from hand-picked city arms.
    """

    demand = demand.copy()
    city_boundary = _align_crs(city_boundary, demand.crs)
    centres = _align_crs(centres, demand.crs)
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    water_crossings = _align_crs(water_crossings, demand.crs)
    grid_step_m = float(config.grid_station_step_m if grid_step_m is None else grid_step_m)
    max_candidates = int(config.grid_max_station_candidates if max_candidates is None else max_candidates)

    city_area = forbidden_union(city_boundary)
    if city_area is None or city_area.is_empty:
        minx, miny, maxx, maxy = demand.total_bounds
        city_area = box(minx, miny, maxx, maxy).buffer(config.relocation_study_area_buffer_m)

    minx, miny, maxx, maxy = city_area.bounds
    xs = np.arange(minx, maxx + grid_step_m, grid_step_m)
    ys = np.arange(miny, maxy + grid_step_m, grid_step_m)

    forbidden_geom = forbidden_union(forbidden)
    water_geom = forbidden_union(water_crossings)
    flood_avoidance_geom = (
        forbidden_geom.buffer(float(config.station_risk_buffer_m))
        if forbidden_geom is not None
        and not forbidden_geom.is_empty
        and not config.flood_zones_are_cost_only
        else None
    )
    water_avoidance_geom = (
        water_geom.buffer(float(config.station_water_buffer_m))
        if water_geom is not None and not water_geom.is_empty
        else None
    )

    points: list[Point] = []
    point_sources: list[str] = []
    for x in xs:
        for y in ys:
            point = Point(float(x), float(y))
            if not city_area.covers(point):
                continue
            if flood_avoidance_geom is not None and flood_avoidance_geom.covers(point):
                continue
            if water_avoidance_geom is not None and water_avoidance_geom.covers(point):
                continue
            points.append(point)
            point_sources.append("grid_station")

    if config.grid_include_demand_points:
        for point, raw_weight in zip(demand.geometry, pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0)):
            if raw_weight <= 0.0:
                continue
            if not city_area.covers(point):
                continue
            if flood_avoidance_geom is not None and flood_avoidance_geom.covers(point):
                continue
            if water_avoidance_geom is not None and water_avoidance_geom.covers(point):
                continue
            points.append(point)
            point_sources.append("demand_point")

    if not points:
        return gpd.GeoDataFrame(geometry=[], crs=demand.crs)

    weights = candidate_catchment_weights(
        points,
        demand,
        config.walk_radius_m,
        weight_col=weight_col,
        use_cuda=config.grid_use_cuda,
        cuda_batch_size=config.grid_cuda_batch_size,
    )
    rows = []
    required_point = None
    if centres is not None and not centres.empty:
        required = centres[centres["role"].eq("required_city_centre")] if "role" in centres.columns else centres
        if not required.empty:
            required_point = required.geometry.iloc[0]

    for idx, (point, point_source, catchment) in enumerate(zip(points, point_sources, weights), start=1):
        geology_factor = geology_factor_at_point(point, geology)
        geology_excess = max(0.0, geology_factor - 1.0)
        high_geology = bool(geology_factor >= config.high_geology_factor_threshold)
        geology_weight_factor = max(0.15, 1.0 - geology_excess * config.station_geology_score_factor)
        if high_geology:
            geology_weight_factor *= 0.25
        in_flood = bool(forbidden_geom.covers(point)) if forbidden_geom is not None and not forbidden_geom.is_empty else False
        in_water = bool(water_geom.covers(point)) if water_geom is not None and not water_geom.is_empty else False
        candidate_weight = float(catchment) * geology_weight_factor
        if in_water:
            candidate_weight *= config.station_water_score_factor
        if in_flood and not config.flood_zones_are_cost_only:
            candidate_weight = 0.0
        dx = point.x - required_point.x if required_point is not None else 0.0
        dy = point.y - required_point.y if required_point is not None else 0.0
        octant = direction_octant_from_delta(dx, dy) if required_point is not None else "unknown"
        rows.append(
            {
                "candidate_id": f"{'D' if point_source == 'demand_point' else 'G'}{idx:04d}",
                "source": point_source,
                "name": f"{point_source}_{idx:04d}",
                "required": False,
                "catchment_weight": float(catchment),
                "candidate_weight": candidate_weight,
                "geology_factor": geology_factor,
                "high_geology": high_geology,
                "geology_weight_factor": geology_weight_factor,
                "in_flood_zone": in_flood,
                "in_water_zone": in_water,
                "direction_octant": octant,
                "direction_sector": broad_direction_from_octant(octant),
                "distance_to_required_centre_m": point.distance(required_point) if required_point is not None else np.nan,
                "geometry": point,
            }
        )

    candidates = gpd.GeoDataFrame(rows, geometry="geometry", crs=demand.crs)
    candidates = candidates[candidates["catchment_weight"] > 0.0].copy()
    if candidates.empty:
        return candidates

    ranked = candidates.sort_values("candidate_weight", ascending=False)
    if required_point is not None and len(ranked) > max_candidates:
        support_count = min(
            max(0, max_candidates - 1),
            max(int(round(max_candidates * config.grid_connectivity_candidate_share)), config.station_count * 2),
        )
        demand_count = max(1, max_candidates - support_count)
        demand_ranked = ranked.head(demand_count).copy()
        demand_ranked["grid_selection_role"] = "demand_rank"
        support_pool = candidates.drop(index=demand_ranked.index)
        support = _grid_connectivity_support_candidates(
            support_pool,
            required_point,
            config,
            support_count,
        )
        candidates = pd.concat([demand_ranked, support], axis=0)
        candidates = candidates[~candidates.index.duplicated(keep="first")]
        if len(candidates) < max_candidates:
            fill = ranked.loc[~ranked.index.isin(candidates.index)].head(max_candidates - len(candidates)).copy()
            fill["grid_selection_role"] = "demand_rank_fill"
            candidates = pd.concat([candidates, fill], axis=0)
        candidates = candidates.head(max_candidates).copy()
    else:
        candidates = ranked.head(max_candidates).copy()
        candidates["grid_selection_role"] = "demand_rank"

    candidates = candidates.sort_values("candidate_weight", ascending=False).copy()

    if required_point is not None:
        required_weight = float(
            candidate_catchment_weights(
                [required_point],
                demand,
                config.walk_radius_m,
                weight_col=weight_col,
                use_cuda=config.grid_use_cuda,
                cuda_batch_size=config.grid_cuda_batch_size,
            )[0]
        )
        required_row = gpd.GeoDataFrame(
            [
                {
                    "candidate_id": "GRID-CENTRE",
                    "source": "required",
                    "name": "Stare Miasto",
                    "required": True,
                    "catchment_weight": required_weight,
                    "candidate_weight": required_weight,
                    "geology_factor": geology_factor_at_point(required_point, geology),
                    "high_geology": False,
                    "geology_weight_factor": 1.0,
                    "grid_selection_role": "required",
                    "in_flood_zone": bool(forbidden_geom.covers(required_point)) if forbidden_geom is not None and not forbidden_geom.is_empty else False,
                    "in_water_zone": bool(water_geom.covers(required_point)) if water_geom is not None and not water_geom.is_empty else False,
                    "direction_octant": "centre",
                    "direction_sector": "centre",
                    "distance_to_required_centre_m": 0.0,
                    "geometry": required_point,
                }
            ],
            geometry="geometry",
            crs=demand.crs,
        )
        candidates = pd.concat([required_row, candidates], ignore_index=True)

    candidates["grid_candidate_rank"] = np.arange(1, len(candidates) + 1)
    return gpd.GeoDataFrame(candidates, geometry="geometry", crs=demand.crs)


def _grid_station_spacing_ok(
    order: list[int],
    distance_matrix: np.ndarray,
    config: MetroConfig,
    enforce_max_spacing: bool = False,
) -> bool:
    if len(order) <= 1:
        return True
    left = np.array(order[:-1], dtype=int)
    right = np.array(order[1:], dtype=int)
    spacing = distance_matrix[left, right]
    if not np.all(spacing >= config.station_min_spacing_m):
        return False
    if enforce_max_spacing and not np.all(spacing <= config.station_max_spacing_m):
        return False
    return True


def _grid_connectivity_support_candidates(
    candidates: gpd.GeoDataFrame,
    centre: Point,
    config: MetroConfig,
    limit: int,
) -> gpd.GeoDataFrame:
    """Keep lower-ranked grid points that make station-to-station routing feasible."""

    if limit <= 0 or candidates.empty:
        return candidates.iloc[0:0].copy()

    remaining = candidates.copy()
    selected_labels = []
    reached_xy = np.array([[centre.x, centre.y]], dtype=float)
    max_spacing = max(float(config.station_max_spacing_m), float(config.grid_station_step_m) * 1.5)
    min_spacing = max(0.0, min(float(config.station_min_spacing_m) * 0.70, float(config.grid_station_step_m) * 0.90))

    for _ in range(int(limit)):
        if remaining.empty:
            break

        xy = np.array([(point.x, point.y) for point in remaining.geometry], dtype=float)
        distances_to_reached = np.sqrt(((xy[:, None, :] - reached_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        reachable = (distances_to_reached <= max_spacing) & (distances_to_reached >= min_spacing)
        if not reachable.any():
            reachable = distances_to_reached <= max_spacing * 1.5
        if not reachable.any():
            break

        candidate_weight = pd.to_numeric(remaining["candidate_weight"], errors="coerce").fillna(0.0).to_numpy()
        catchment_weight = pd.to_numeric(remaining["catchment_weight"], errors="coerce").fillna(0.0).to_numpy()
        centre_distance = np.sqrt((xy[:, 0] - centre.x) ** 2 + (xy[:, 1] - centre.y) ** 2)
        frontier_bonus = np.clip(centre_distance / max(config.length_m * 0.5, 1.0), 0.0, 1.0)
        score = candidate_weight + 0.10 * catchment_weight + 2_500.0 * frontier_bonus - 0.10 * distances_to_reached
        score[~reachable] = -np.inf
        if not np.isfinite(score).any():
            break

        best_position = int(np.nanargmax(score))
        best_label = remaining.index[best_position]
        selected_labels.append(best_label)
        reached_xy = np.vstack([reached_xy, xy[best_position]])
        remaining = remaining.drop(index=best_label)

    out = candidates.loc[selected_labels].copy()
    if not out.empty:
        out["grid_selection_role"] = "connectivity_support"
    return out


def _grid_route_metrics(
    order: list[int],
    nodes: list[dict],
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    water_crossings: gpd.GeoDataFrame | None,
    transfer_points: gpd.GeoDataFrame | None,
    transfer_lines: gpd.GeoDataFrame | None,
    city_boundary: gpd.GeoDataFrame | None,
    config: MetroConfig,
    weight_col: str,
) -> dict:
    points = [nodes[index]["geometry"] for index in order]
    metrics = route_score_from_points(
        points,
        demand,
        forbidden,
        geology,
        config,
        weight_col=weight_col,
        water_crossings=water_crossings,
        transfer_points=transfer_points,
        transfer_lines=transfer_lines,
        existing_lines=transfer_lines,
        city_boundary=city_boundary,
    )
    route_km = metrics["route_length_m"] / 1_000.0
    target_km = config.length_m / 1_000.0
    min_target_km = target_km * config.grid_min_length_ratio
    spacing = (
        np.array([points[index].distance(points[index + 1]) for index in range(len(points) - 1)], dtype=float)
        if len(points) >= 2
        else np.array([], dtype=float)
    )
    station_gap_excess_m = float(np.maximum(0.0, spacing - config.station_max_spacing_m).sum()) if len(spacing) else 0.0
    station_gap_penalty = station_gap_excess_m * config.grid_station_gap_penalty_per_m
    length_usage_bonus = min(route_km, target_km) * config.grid_length_usage_bonus_per_km
    length_underuse_penalty = max(0.0, min_target_km - route_km) * config.grid_length_underuse_penalty_per_km
    metrics["max_station_gap_m"] = float(spacing.max()) if len(spacing) else 0.0
    metrics["station_gap_excess_m"] = station_gap_excess_m
    metrics["station_gap_penalty"] = station_gap_penalty
    metrics["length_usage_bonus"] = length_usage_bonus
    metrics["length_underuse_penalty"] = length_underuse_penalty
    metrics["score"] += length_usage_bonus - length_underuse_penalty - station_gap_penalty
    return metrics


def solve_grid_station_route(
    demand: gpd.GeoDataFrame,
    station_candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    transfer_points: gpd.GeoDataFrame | None = None,
    transfer_lines: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
    line_id: int = 1,
    force_station_count: bool = True,
) -> dict:
    """Plan a line by selecting actual station candidates from a grid."""

    demand = demand.copy()
    station_candidates = _align_crs(station_candidates, demand.crs).copy()
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    water_crossings = _align_crs(water_crossings, demand.crs)
    transfer_points = _align_crs(transfer_points, demand.crs)
    transfer_lines = _align_crs(transfer_lines, demand.crs)
    city_boundary = _align_crs(city_boundary, demand.crs)

    if "candidate_id" not in station_candidates.columns:
        station_candidates["candidate_id"] = [f"G{i:04d}" for i in range(1, len(station_candidates) + 1)]
    if "required" in station_candidates.columns:
        required_candidates = station_candidates[station_candidates["required"].astype(bool)]
    else:
        required_candidates = station_candidates.iloc[0:0]

    nodes: list[dict] = []
    if not required_candidates.empty:
        centre_row = required_candidates.iloc[0]
        nodes.append(_node_from_candidate_row(centre_row, geometry=centre_row.geometry, candidate_id="GRID-CENTRE"))
        candidate_iter = station_candidates[~station_candidates.index.isin(required_candidates.index)]
    else:
        nodes.append({"candidate_id": "GRID-CENTRE", "name": "Stare Miasto", "source": "required", "geometry": centre})
        candidate_iter = station_candidates

    for _, row in candidate_iter.iterrows():
        nodes.append(_node_from_candidate_row(row))

    if len(nodes) <= 1:
        raise ValueError("Grid route needs at least one non-centre station candidate.")

    node_xy = np.array([(node["geometry"].x, node["geometry"].y) for node in nodes], dtype=float)
    distance_matrix, node_distance_backend = _pairwise_distance_matrix(
        node_xy,
        node_xy,
        use_cuda=config.grid_use_cuda,
        batch_size=config.grid_cuda_batch_size,
    )
    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    demand_weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    node_to_demand, demand_distance_backend = _pairwise_distance_matrix(
        node_xy,
        demand_xy,
        use_cuda=config.grid_use_cuda,
        batch_size=config.grid_cuda_batch_size,
    )
    coverage_matrix = np.clip(1.0 - node_to_demand / config.walk_radius_m, 0.0, 1.0)
    total_weight = float(demand_weights.sum())
    node_geology_penalty = np.zeros(len(nodes), dtype=float)
    for index, node in enumerate(nodes):
        factor = float(node.get("geology_factor", np.nan))
        if pd.isna(factor):
            factor = geology_factor_at_point(node["geometry"], geology)
        node_geology_penalty[index] = (
            max(0.0, factor - 1.0) * config.station_geology_penalty_per_excess
            + (
                config.station_high_geology_penalty
                if factor >= config.high_geology_factor_threshold
                else 0.0
            )
        )
    transfer_point_coverage = np.zeros(len(nodes), dtype=float)
    if transfer_points is not None and not transfer_points.empty:
        transfer_xy = np.array([(point.x, point.y) for point in transfer_points.geometry], dtype=float)
        if len(transfer_xy):
            transfer_distances, _ = _pairwise_distance_matrix(
                node_xy,
                transfer_xy,
                use_cuda=config.grid_use_cuda,
                batch_size=config.grid_cuda_batch_size,
            )
            transfer_point_coverage = (transfer_distances.min(axis=1) <= config.interchange_radius_m).astype(float)
    existing_line_node_penalty = np.zeros(len(nodes), dtype=float)
    existing_line_geom = forbidden_union(transfer_lines)
    existing_overlap_zone = None
    existing_segments: list[tuple[LineString, float]] = []
    if existing_line_geom is not None and not existing_line_geom.is_empty:
        buffer_m = max(1.0, float(config.candidate_existing_line_buffer_m))
        for index, node in enumerate(nodes):
            distance_m = float(node["geometry"].distance(existing_line_geom))
            if distance_m <= buffer_m:
                existing_line_node_penalty[index] = (
                    (1.0 - distance_m / buffer_m) * config.grid_existing_line_node_penalty
                )
        existing_geoms = [geom for geom in transfer_lines.geometry if geom is not None and not geom.is_empty]
        existing_overlap_zone = existing_line_geom.buffer(config.parallel_line_buffer_m)
        existing_segments = _line_segments_with_angles(existing_geoms)
    segment_overlap_km = np.zeros((len(nodes), len(nodes)), dtype=float)
    if existing_overlap_zone is not None and not existing_overlap_zone.is_empty and existing_segments:
        for left_index in range(len(nodes)):
            left_point = nodes[left_index]["geometry"]
            for right_index in range(left_index + 1, len(nodes)):
                right_point = nodes[right_index]["geometry"]
                if left_point.equals(right_point):
                    continue
                segment = LineString([left_point, right_point])
                overlap_km = _parallel_overlap_length_m(
                    segment,
                    existing_overlap_zone,
                    existing_segments,
                    config,
                ) / 1_000.0
                segment_overlap_km[left_index, right_index] = overlap_km
                segment_overlap_km[right_index, left_index] = overlap_km

    def fast_grid_metrics(order: list[int]) -> dict:
        order_array = np.array(order, dtype=int)
        if len(order) >= 2:
            left = np.array(order[:-1], dtype=int)
            right = np.array(order[1:], dtype=int)
            spacing = distance_matrix[left, right]
            route_length_m = float(spacing.sum())
            line_overlap_value_km = float(segment_overlap_km[left, right].sum())
        else:
            spacing = np.array([], dtype=float)
            route_length_m = 0.0
            line_overlap_value_km = 0.0
        coverage = coverage_matrix[order_array].max(axis=0)
        served_weight = float(np.sum(demand_weights * coverage))
        transfer_point_count = int(min(2, transfer_point_coverage[order_array].sum()))
        transfer_score = float(transfer_point_count * config.transfer_bonus_per_interchange)
        turn_metrics = _turn_metrics_from_xy(node_xy[order_array], config)
        shape_metrics = _corridor_shape_metrics_from_xy(node_xy[order_array], config)
        station_geology_penalty = float(node_geology_penalty[order_array].sum())
        existing_line_penalty = float(existing_line_node_penalty[order_array].sum())
        route_km = route_length_m / 1_000.0
        target_km = config.length_m / 1_000.0
        min_target_km = target_km * config.grid_min_length_ratio
        station_gap_excess_m = float(np.maximum(0.0, spacing - config.station_max_spacing_m).sum()) if len(spacing) else 0.0
        station_gap_penalty = station_gap_excess_m * config.grid_station_gap_penalty_per_m
        length_usage_bonus = min(route_km, target_km) * config.grid_length_usage_bonus_per_km
        length_underuse_penalty = max(0.0, min_target_km - route_km) * config.grid_length_underuse_penalty_per_km
        limit_penalties = spatial_limit_penalties(0.0, line_overlap_value_km, config)
        score = served_weight
        score -= station_geology_penalty
        score -= existing_line_penalty
        score -= line_overlap_value_km * config.line_overlap_penalty_per_km
        score -= limit_penalties["line_overlap_excess_penalty"]
        score -= turn_metrics["turn_penalty"]
        score -= shape_metrics["corridor_shape_penalty"]
        score -= station_gap_penalty
        score += transfer_score
        score += length_usage_bonus
        score -= length_underuse_penalty
        return {
            "score": score,
            "served_weight": served_weight,
            "served_share": served_weight / total_weight if total_weight else 0.0,
            "route_length_m": route_length_m,
            "forbidden_km": 0.0,
            "forbidden_excess_km": 0.0,
            "forbidden_excess_penalty": 0.0,
            "geology_factor": 1.0,
            "geology_excess_km": 0.0,
            "high_geology_km": 0.0,
            "station_geology_penalty": station_geology_penalty,
            "existing_line_node_penalty": existing_line_penalty,
            "water_crossing_km": 0.0,
            "transfer_score": transfer_score,
            "transfer_count": transfer_point_count,
            "line_overlap_km": line_overlap_value_km,
            "line_overlap_excess_km": limit_penalties["line_overlap_excess_km"],
            "line_overlap_excess_penalty": limit_penalties["line_overlap_excess_penalty"],
            "outside_city_km": 0.0,
            "outside_city_penalty": 0.0,
            "outside_city_excess_km": 0.0,
            "outside_city_excess_penalty": 0.0,
            "max_station_gap_m": float(spacing.max()) if len(spacing) else 0.0,
            "station_gap_excess_m": station_gap_excess_m,
            "station_gap_penalty": station_gap_penalty,
            "length_usage_bonus": length_usage_bonus,
            "length_underuse_penalty": length_underuse_penalty,
            **turn_metrics,
            **shape_metrics,
        }

    selected_order = [0]
    selected_nodes = {0}
    current_metrics = fast_grid_metrics(selected_order)
    log_rows = []

    while len(selected_order) < int(config.station_count):
        best_move = None
        for node_index in range(1, len(nodes)):
            if node_index in selected_nodes:
                continue
            for position in range(len(selected_order) + 1):
                proposal_order = selected_order[:position] + [node_index] + selected_order[position:]
                if not _grid_station_spacing_ok(proposal_order, distance_matrix, config):
                    continue
                metrics = fast_grid_metrics(proposal_order)
                if metrics["route_length_m"] > config.length_m:
                    continue
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                hard_limit_violation = (
                    metrics["forbidden_excess_km"] > 1e-6
                    or metrics["line_overlap_excess_km"] > 1e-6
                    or metrics["outside_city_excess_km"] > 1e-6
                )
                improvement = metrics["score"] - current_metrics["score"]
                added_length = max(1.0, metrics["route_length_m"] - current_metrics["route_length_m"])
                ranking_key = (
                    not hard_limit_violation,
                    improvement / added_length,
                    improvement,
                    metrics["served_weight"],
                    metrics["route_length_m"],
                )
                candidate_move = {
                    "node_index": node_index,
                    "position": position,
                    "proposal_order": proposal_order,
                    "metrics": metrics,
                    "improvement": improvement,
                    "added_length_m": added_length,
                    "ranking_key": ranking_key,
                }
                if best_move is None or ranking_key > best_move["ranking_key"]:
                    best_move = candidate_move

        if best_move is None:
            break
        if not force_station_count and best_move["improvement"] <= 0:
            break
        selected_order = best_move["proposal_order"]
        selected_nodes.add(best_move["node_index"])
        current_metrics = best_move["metrics"]
        log_rows.append(
            {
                "step": len(log_rows) + 1,
                "candidate_id": nodes[best_move["node_index"]]["candidate_id"],
                "insert_position": best_move["position"],
                "improvement": best_move["improvement"],
                "added_length_m": best_move["added_length_m"],
                **best_move["metrics"],
            }
        )

    improved = True
    iteration = 0
    while improved and iteration < 40 and len(selected_order) >= 4:
        improved = False
        iteration += 1
        best_metrics = fast_grid_metrics(selected_order)
        best_score = best_metrics["score"]
        for left in range(0, len(selected_order) - 2):
            for right in range(left + 2, len(selected_order)):
                proposal_order = (
                    selected_order[:left]
                    + list(reversed(selected_order[left : right + 1]))
                    + selected_order[right + 1 :]
                )
                if not _grid_station_spacing_ok(proposal_order, distance_matrix, config):
                    continue
                metrics = fast_grid_metrics(proposal_order)
                if (
                    metrics["route_length_m"] <= config.length_m
                    and metrics["score"] > best_score
                    and (
                        config.hard_max_turn_angle_deg <= 0.0
                        or metrics["max_turn_angle_deg"] <= config.hard_max_turn_angle_deg
                    )
                ):
                    selected_order = proposal_order
                    improved = True
                    break
            if improved:
                break

    selected_nodes_data = [nodes[index] for index in selected_order]
    control_points = [node["geometry"] for node in selected_nodes_data]
    city_boundary_geom = forbidden_union(city_boundary)
    if len(control_points) == 1:
        corridor_points = best_single_anchor_corridor_points(
            control_points[0],
            config,
            city_area=city_boundary_geom,
            existing_lines=transfer_lines,
        )
    else:
        corridor_points = extend_route_points_to_length(
            control_points,
            config.length_m,
            city_boundary_geom,
            flexible_terminal=True,
        )
    corridor = route_polyline(corridor_points)
    forbidden_geom = forbidden_union(forbidden)
    water_geom = forbidden_union(water_crossings)
    flood_buffer = (
        forbidden_geom.buffer(config.station_risk_buffer_m)
        if forbidden_geom is not None and not forbidden_geom.is_empty
        else None
    )
    water_buffer = (
        water_geom.buffer(config.station_water_buffer_m)
        if water_geom is not None and not water_geom.is_empty
        else None
    )

    if config.adaptive_station_placement and len(control_points) >= 2 and corridor.length > 0:
        stations = stations_for_corridor(
            corridor,
            demand,
            forbidden_geom,
            geology,
            config,
            line_id=line_id,
            weight_col=weight_col,
            water_geom=water_geom,
            route_anchor_points=control_points,
        )
        stations["candidate_id"] = stations["station_code"]
        stations["source"] = "adaptive_grid_station"
        stations["grid_selection_role"] = "adaptive_on_grid_corridor"
    else:
        station_points = control_points
        station_distances = [float(corridor.project(point)) for point in station_points]
        previous_spacing = np.concatenate(([np.nan], np.diff(station_distances)))
        next_spacing = np.concatenate((np.diff(station_distances), [np.nan]))
        stations = gpd.GeoDataFrame(
            {
                "line_id": line_id,
                "station_id": np.arange(1, len(station_points) + 1),
                "station_code": [f"L{line_id:02d}-S{idx:02d}" for idx in range(1, len(station_points) + 1)],
                "candidate_id": [node.get("candidate_id", "") for node in selected_nodes_data],
                "source": [node.get("source", "grid_station") for node in selected_nodes_data],
                "grid_selection_role": [
                    node.get("grid_selection_role", node.get("source", "grid_station"))
                    for node in selected_nodes_data
                ],
                "distance_m": station_distances,
                "spacing_from_previous_m": previous_spacing,
                "spacing_to_next_m": next_spacing,
                "station_placement": "grid_orienteering",
                "geology_factor": [geology_factor_at_point(point, geology) for point in station_points],
                "high_geology": [
                    geology_factor_at_point(point, geology) >= config.high_geology_factor_threshold
                    for point in station_points
                ],
                "in_station_flood_zone": [
                    bool(forbidden_geom.covers(point)) if forbidden_geom is not None and not forbidden_geom.is_empty else False
                    for point in station_points
                ],
                "in_station_water_zone": [
                    bool(water_geom.covers(point)) if water_geom is not None and not water_geom.is_empty else False
                    for point in station_points
                ],
                "in_station_flood_buffer": [
                    bool(flood_buffer.covers(point)) if flood_buffer is not None else False
                    for point in station_points
                ],
                "in_station_water_buffer": [
                    bool(water_buffer.covers(point)) if water_buffer is not None else False
                    for point in station_points
                ],
            },
            geometry=station_points,
            crs=demand.crs,
        )

    station_points = list(stations.geometry)
    station_metric_nodes = [{"geometry": point} for point in station_points]
    final_metrics = _grid_route_metrics(
        list(range(len(station_metric_nodes))),
        station_metric_nodes,
        demand,
        forbidden,
        geology,
        water_crossings,
        transfer_points,
        transfer_lines,
        city_boundary,
        config,
        weight_col,
    )
    estimated_cost, base_cost, flood_extra_cost = construction_cost_mln(
        corridor.length / 1_000.0,
        final_metrics["forbidden_km"],
        final_metrics["geology_factor"],
        config,
    )

    anchors = gpd.GeoDataFrame(
        {
            "line_id": line_id,
            "anchor_order": np.arange(1, len(control_points) + 1),
            "candidate_id": [node.get("candidate_id", "") for node in selected_nodes_data],
            "name": [node.get("name", "") for node in selected_nodes_data],
            "source": [node.get("source", "grid_station") for node in selected_nodes_data],
            "grid_selection_role": [
                node.get("grid_selection_role", node.get("source", "grid_station"))
                for node in selected_nodes_data
            ],
            "candidate_weight": [node.get("candidate_weight", np.nan) for node in selected_nodes_data],
            "direction_sector": [node.get("direction_sector", np.nan) for node in selected_nodes_data],
            "direction_octant": [node.get("direction_octant", np.nan) for node in selected_nodes_data],
            "distance_to_required_centre_m": [node.get("distance_to_required_centre_m", np.nan) for node in selected_nodes_data],
        },
        geometry=control_points,
        crs=demand.crs,
    )

    line_gdf = gpd.GeoDataFrame(
        {
            "line_id": [line_id],
            "algorithm": ["grid_station_orienteering"],
            "grid_cuda_requested": [bool(config.grid_use_cuda)],
            "grid_distance_backend": [
                "cuda" if "cuda" in {node_distance_backend, demand_distance_backend} else "numpy"
            ],
            "anchor_count": [len(control_points)],
            "served_weight": [final_metrics["served_weight"]],
            "served_share": [final_metrics["served_share"]],
            "forbidden_km": [final_metrics["forbidden_km"]],
            "forbidden_excess_km": [final_metrics["forbidden_excess_km"]],
            "forbidden_excess_penalty": [final_metrics["forbidden_excess_penalty"]],
            "geology_factor": [final_metrics["geology_factor"]],
            "geology_excess_km": [final_metrics["geology_excess_km"]],
            "high_geology_km": [final_metrics["high_geology_km"]],
            "station_geology_penalty": [final_metrics["station_geology_penalty"]],
            "anchor_geology_penalty": [0.0],
            "station_flood_zone_count": [int(stations["in_station_flood_zone"].sum())],
            "station_water_zone_count": [int(stations["in_station_water_zone"].sum())],
            "station_flood_buffer_count": [int(stations["in_station_flood_buffer"].sum())],
            "station_water_buffer_count": [int(stations["in_station_water_buffer"].sum())],
            "water_crossing_km": [final_metrics["water_crossing_km"]],
            "outside_city_km": [final_metrics["outside_city_km"]],
            "outside_city_penalty": [final_metrics["outside_city_penalty"]],
            "outside_city_excess_km": [final_metrics["outside_city_excess_km"]],
            "outside_city_excess_penalty": [final_metrics["outside_city_excess_penalty"]],
            "transfer_score": [final_metrics["transfer_score"]],
            "transfer_count": [final_metrics["transfer_count"]],
            "direction_priority_bonus": [0.0],
            "direction_priority_sector_count": [0],
            "line_overlap_km": [final_metrics["line_overlap_km"]],
            "line_overlap_excess_km": [final_metrics["line_overlap_excess_km"]],
            "line_overlap_excess_penalty": [final_metrics["line_overlap_excess_penalty"]],
            "arm_reuse_penalty": [0.0],
            "forced_direction_priority_anchor": [False],
            "forced_southern_anchor": [False],
            "turn_penalty": [final_metrics["turn_penalty"]],
            "max_turn_angle_deg": [final_metrics["max_turn_angle_deg"]],
            "mean_turn_angle_deg": [final_metrics["mean_turn_angle_deg"]],
            "sharp_turn_count": [final_metrics["sharp_turn_count"]],
            "curve_radius_violation_m": [final_metrics["curve_radius_violation_m"]],
            "max_station_gap_m": [final_metrics["max_station_gap_m"]],
            "station_gap_excess_m": [final_metrics["station_gap_excess_m"]],
            "station_gap_penalty": [final_metrics["station_gap_penalty"]],
            "corridor_detour_ratio": [final_metrics["corridor_detour_ratio"]],
            "corridor_backtrack_km": [final_metrics["corridor_backtrack_km"]],
            "corridor_shape_penalty": [final_metrics["corridor_shape_penalty"]],
            "length_usage_bonus": [final_metrics["length_usage_bonus"]],
            "length_underuse_penalty": [final_metrics["length_underuse_penalty"]],
            "estimated_cost_mln": [estimated_cost],
            "base_cost_mln": [base_cost],
            "flood_extra_cost_mln": [flood_extra_cost],
            "flood_cost_multiplier": [config.flood_cost_multiplier],
            "score": [final_metrics["score"]],
            "anchor_objective_score": [final_metrics["score"]],
            "anchor_served_weight": [final_metrics["served_weight"]],
        },
        geometry=[corridor],
        crs=demand.crs,
    )

    return {
        "line": line_gdf,
        "stations": stations,
        "anchors": anchors,
        "iteration_log": pd.DataFrame(log_rows),
        "best": {
            **final_metrics,
            "line": corridor,
            "stations": stations,
            "anchors": anchors,
        },
    }


def _two_opt_route(
    ordered: list[dict],
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    weight_col: str,
    max_iter: int = 60,
) -> list[dict]:
    """Open-route 2-opt improvement used after greedy insertion."""

    if len(ordered) < 4:
        return ordered

    best = list(ordered)
    best_score = route_score_from_points(
        [item["geometry"] for item in best],
        demand,
        forbidden,
        geology,
        config,
        weight_col=weight_col,
    )["score"]

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1 :]
                candidate_points = [item["geometry"] for item in candidate]
                if route_length_m(candidate_points) > config.length_m:
                    continue
                candidate_score = route_score_from_points(
                    candidate_points,
                    demand,
                    forbidden,
                    geology,
                    config,
                    weight_col=weight_col,
                )["score"]
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break
    return best


def solve_orienteering_route(
    demand: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    transfer_points: gpd.GeoDataFrame | None = None,
    transfer_lines: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
    force_station_count: bool = True,
    min_anchor_spacing_m: float = 550.0,
    line_id: int = 1,
    city_boundary: gpd.GeoDataFrame | None = None,
) -> dict:
    """Solve a prize-collecting TSP / orienteering approximation.

    The exact problem is NP-hard. This heuristic uses greedy best insertion under
    the length budget and then an open-route 2-opt local search.
    """

    demand = demand.copy()
    candidates = _align_crs(candidates, demand.crs).copy()
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    water_crossings = _align_crs(water_crossings, demand.crs)
    transfer_points = _align_crs(transfer_points, demand.crs)
    transfer_lines = _align_crs(transfer_lines, demand.crs)
    city_boundary = _align_crs(city_boundary, demand.crs)
    forbidden_geom = forbidden_union(forbidden)
    water_geom = forbidden_union(water_crossings)
    city_boundary_geom = forbidden_union(city_boundary)
    existing_line_geom = forbidden_union(transfer_lines)

    if "candidate_id" not in candidates.columns:
        candidates["candidate_id"] = [f"C{i:03d}" for i in range(1, len(candidates) + 1)]
    if "name" not in candidates.columns:
        candidates["name"] = candidates["candidate_id"]

    if "required" in candidates.columns:
        required_candidates = candidates[candidates["required"].astype(bool)]
    else:
        required_candidates = candidates.iloc[0:0]
    if not required_candidates.empty:
        centre_row = required_candidates.iloc[0]
        centre = centre_row.geometry
        centre_node = _node_from_candidate_row(centre_row, geometry=centre, candidate_id="FORCED-CENTRE")
        centre_node["source"] = "required"
        candidate_iter = candidates[~candidates.index.isin(required_candidates.index)]
    else:
        centre_node = {
            "candidate_id": "FORCED-CENTRE",
            "name": "forced centre",
            "source": "required",
            "geometry": centre,
        }
        candidate_iter = candidates

    nodes: list[dict] = [centre_node]
    for _, row in candidate_iter.iterrows():
        nodes.append(_node_from_candidate_row(row))

    matrices = _fast_orienteering_matrices(
        demand,
        nodes,
        forbidden_geom,
        geology,
        config,
        weight_col,
        water_geom=water_geom,
        transfer_points=transfer_points,
        transfer_lines=transfer_lines,
        existing_lines=transfer_lines,
        city_boundary_geom=city_boundary_geom,
    )
    distance_matrix = matrices["distance_matrix"]
    node_radial_overlap_km = np.zeros(len(nodes), dtype=float)
    if existing_line_geom is not None and not existing_line_geom.is_empty and not transfer_lines.empty:
        for node_index in range(1, len(nodes)):
            node_radial_overlap_km[node_index] = line_overlap_km(
                [centre_node["geometry"], nodes[node_index]["geometry"]],
                transfer_lines,
                config,
            )

    def fast_metrics(order: list[int]) -> dict:
        return _fast_order_metrics(order, matrices, config)

    target_anchor_count = max(1, min(int(config.route_anchor_count), len(nodes), config.station_count))

    selected_order = [0]
    selected_nodes = {0}
    log_rows = []
    current_metrics = fast_metrics(selected_order)
    forced_direction_priority_anchor = False
    forced_southern_anchor = False

    priority_sectors = {str(sector).lower() for sector in config.direction_priority_sectors}
    if (
        priority_sectors
        and config.direction_priority_seed_line_id is not None
        and line_id >= int(config.direction_priority_seed_line_id)
    ):
        seed_move = None
        for node_index in range(1, len(nodes)):
            node = nodes[node_index]
            octant = str(node.get("direction_octant", "")).lower()
            broad = str(node.get("direction_sector", "")).lower()
            if octant not in priority_sectors and broad not in priority_sectors:
                continue
            if (
                float(node.get("distance_to_required_centre_m", np.inf))
                < config.direction_priority_min_distance_m
            ):
                continue
            if distance_matrix[node_index, 0] < min_anchor_spacing_m:
                continue
            if (
                existing_line_geom is not None
                and not existing_line_geom.is_empty
                and config.candidate_existing_line_buffer_m > 0.0
                and node.get("source") != "required"
                and node["geometry"].distance(existing_line_geom) < config.candidate_existing_line_buffer_m
            ):
                continue
            if (
                config.candidate_radial_overlap_limit_km > 0.0
                and node.get("source") != "required"
                and node_radial_overlap_km[node_index] > config.candidate_radial_overlap_limit_km
            ):
                continue

            for proposal_order in ([0, node_index], [node_index, 0]):
                metrics = fast_metrics(proposal_order)
                if metrics["route_length_m"] > config.length_m:
                    continue
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                hard_limit_violation = (
                    metrics["forbidden_excess_km"] > 1e-6
                    or metrics["line_overlap_excess_km"] > 1e-6
                    or metrics["outside_city_excess_km"] > 1e-6
                )
                ranking_key = (
                    not hard_limit_violation,
                    metrics["score"],
                    metrics["served_weight"],
                    -metrics["outside_city_km"],
                    float(node.get("distance_to_required_centre_m", 0.0)),
                )
                candidate_move = {
                    "candidate_id": node["candidate_id"],
                    "node_index": node_index,
                    "proposal_order": list(proposal_order),
                    "metrics": metrics,
                    "ranking_key": ranking_key,
                }
                if seed_move is None or ranking_key > seed_move["ranking_key"]:
                    seed_move = candidate_move

        if seed_move is not None:
            previous_metrics = current_metrics
            selected_order = seed_move["proposal_order"]
            selected_nodes.add(seed_move["node_index"])
            current_metrics = seed_move["metrics"]
            forced_direction_priority_anchor = True
            log_rows.append(
                {
                    "step": 1,
                    "candidate_id": seed_move["candidate_id"],
                    "insert_position": selected_order.index(seed_move["node_index"]),
                    "improvement": current_metrics["score"] - previous_metrics["score"],
                    "added_length_m": current_metrics["route_length_m"] - previous_metrics["route_length_m"],
                    "forced_direction_priority_anchor": True,
                    "forced_southern_anchor": False,
                    **current_metrics,
                }
            )

    if (
        config.force_southern_anchor_line_id is not None
        and line_id >= int(config.force_southern_anchor_line_id)
        and len(selected_order) == 1
    ):
        seed_move = None
        for node_index in range(1, len(nodes)):
            node = nodes[node_index]
            if node.get("direction_sector") != "south":
                continue
            if (
                float(node.get("distance_to_required_centre_m", np.inf))
                < config.force_southern_anchor_min_distance_m
            ):
                continue
            if distance_matrix[node_index, 0] < min_anchor_spacing_m:
                continue

            for proposal_order in ([0, node_index], [node_index, 0]):
                metrics = fast_metrics(proposal_order)
                if metrics["route_length_m"] > config.length_m:
                    continue
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                hard_limit_violation = (
                    metrics["forbidden_excess_km"] > 1e-6
                    or metrics["line_overlap_excess_km"] > 1e-6
                    or metrics["outside_city_excess_km"] > 1e-6
                )
                ranking_key = (
                    not hard_limit_violation,
                    metrics["score"],
                    metrics["served_weight"],
                    -metrics["outside_city_km"],
                    -float(node.get("distance_to_required_centre_m", 0.0)),
                )
                candidate_move = {
                    "candidate_id": node["candidate_id"],
                    "node_index": node_index,
                    "proposal_order": list(proposal_order),
                    "metrics": metrics,
                    "ranking_key": ranking_key,
                }
                if seed_move is None or ranking_key > seed_move["ranking_key"]:
                    seed_move = candidate_move

        if seed_move is not None:
            previous_metrics = current_metrics
            selected_order = seed_move["proposal_order"]
            selected_nodes.add(seed_move["node_index"])
            current_metrics = seed_move["metrics"]
            forced_southern_anchor = True
            log_rows.append(
                {
                    "step": 1,
                    "candidate_id": seed_move["candidate_id"],
                    "insert_position": selected_order.index(seed_move["node_index"]),
                    "improvement": current_metrics["score"] - previous_metrics["score"],
                    "added_length_m": current_metrics["route_length_m"] - previous_metrics["route_length_m"],
                    "forced_direction_priority_anchor": False,
                    "forced_southern_anchor": True,
                    **current_metrics,
                }
            )

    while len(selected_order) < target_anchor_count:
        best_move = None
        fallback_move = None
        for node_index in range(1, len(nodes)):
            if node_index in selected_nodes:
                continue
            if any(distance_matrix[node_index, selected_index] < min_anchor_spacing_m for selected_index in selected_order):
                continue
            if (
                existing_line_geom is not None
                and not existing_line_geom.is_empty
                and config.candidate_existing_line_buffer_m > 0.0
                and nodes[node_index].get("source") != "required"
                and nodes[node_index]["geometry"].distance(existing_line_geom) < config.candidate_existing_line_buffer_m
            ):
                continue
            if (
                config.candidate_radial_overlap_limit_km > 0.0
                and nodes[node_index].get("source") != "required"
                and node_radial_overlap_km[node_index] > config.candidate_radial_overlap_limit_km
            ):
                continue

            for position in range(len(selected_order) + 1):
                proposal_order = selected_order[:position] + [node_index] + selected_order[position:]
                metrics = fast_metrics(proposal_order)
                proposal_length = metrics["route_length_m"]
                if proposal_length > config.length_m:
                    continue
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                improvement = metrics["score"] - current_metrics["score"]
                added_length = max(1.0, proposal_length - current_metrics["route_length_m"])
                ratio = improvement / added_length
                ranking_key = (ratio, improvement, metrics["served_weight"])
                fallback_key = (
                    -(
                        metrics["forbidden_excess_km"]
                        + metrics["line_overlap_excess_km"]
                        + metrics["outside_city_excess_km"]
                    ),
                    -metrics["forbidden_km"],
                    -metrics["line_overlap_km"],
                    -metrics["outside_city_km"],
                    ratio,
                    improvement,
                    metrics["served_weight"],
                )
                candidate_move = {
                    "candidate_id": nodes[node_index]["candidate_id"],
                    "node_index": node_index,
                    "position": position,
                    "proposal_order": proposal_order,
                    "metrics": metrics,
                    "improvement": improvement,
                    "added_length_m": added_length,
                    "ranking_key": ranking_key,
                    "fallback_key": fallback_key,
                }
                hard_limit_violation = (
                    metrics["forbidden_excess_km"] > 1e-6
                    or metrics["line_overlap_excess_km"] > 1e-6
                    or metrics["outside_city_excess_km"] > 1e-6
                )
                if hard_limit_violation:
                    if fallback_move is None or fallback_key > fallback_move["fallback_key"]:
                        fallback_move = candidate_move
                    continue
                if best_move is None or ranking_key > best_move["ranking_key"]:
                    best_move = candidate_move

        if best_move is None and fallback_move is not None:
            fallback_metrics = fallback_move["metrics"]
            worsens_spatial_limits = (
                fallback_metrics["forbidden_excess_km"] > current_metrics["forbidden_excess_km"] + 0.05
                or fallback_metrics["line_overlap_excess_km"] > current_metrics["line_overlap_excess_km"] + 0.05
                or fallback_metrics["outside_city_excess_km"] > current_metrics["outside_city_excess_km"] + 0.05
            )
            if len(selected_order) >= max(3, target_anchor_count - 2) and worsens_spatial_limits:
                break
            best_move = fallback_move
        if best_move is None:
            break
        if not force_station_count and best_move["improvement"] <= 0:
            break

        selected_order = best_move["proposal_order"]
        selected_nodes.add(best_move["node_index"])
        current_metrics = best_move["metrics"]
        log_rows.append(
            {
                "step": len(log_rows) + 1,
                "candidate_id": best_move["candidate_id"],
                "insert_position": best_move["position"],
                "improvement": best_move["improvement"],
                "added_length_m": best_move["added_length_m"],
                "forced_direction_priority_anchor": False,
                "forced_southern_anchor": False,
                **best_move["metrics"],
            }
        )

    improved = True
    iteration = 0
    while improved and iteration < 60 and len(selected_order) >= 4:
        improved = False
        iteration += 1
        best_metrics = fast_metrics(selected_order)
        best_score = best_metrics["score"]
        for left in range(0, len(selected_order) - 2):
            for right in range(left + 2, len(selected_order)):
                proposal_order = (
                    selected_order[:left]
                    + list(reversed(selected_order[left:right + 1]))
                    + selected_order[right + 1 :]
                )
                metrics = fast_metrics(proposal_order)
                if (
                    metrics["route_length_m"] <= config.length_m
                    and metrics["score"] > best_score
                    and (
                        config.hard_max_turn_angle_deg <= 0.0
                        or metrics["max_turn_angle_deg"] <= config.hard_max_turn_angle_deg
                    )
                    and metrics["forbidden_excess_km"] <= best_metrics["forbidden_excess_km"] + 1e-6
                    and metrics["line_overlap_excess_km"] <= best_metrics["line_overlap_excess_km"] + 1e-6
                    and metrics["outside_city_excess_km"] <= best_metrics["outside_city_excess_km"] + 1e-6
                ):
                    selected_order = proposal_order
                    improved = True
                    break
            if improved:
                break

    anchor_metrics = fast_metrics(selected_order)
    selected = [nodes[index] for index in selected_order]

    anchor_points = [item["geometry"] for item in selected]
    if len(anchor_points) == 1:
        corridor_points = best_single_anchor_corridor_points(
            anchor_points[0],
            config,
            city_area=city_boundary_geom,
            existing_lines=transfer_lines,
        )
    else:
        corridor_points = extend_route_points_to_length(
            anchor_points,
            config.length_m,
            city_boundary_geom,
            flexible_terminal=True,
        )
    corridor = route_polyline(corridor_points)
    corridor_turn_metrics = route_turn_metrics(corridor_points, config)
    corridor_shape = corridor_shape_metrics(corridor_points, config)
    outside_city_km = line_outside_city_km(corridor, city_boundary)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km
    corridor_overlap_km = line_overlap_km(corridor_points, transfer_lines, config)
    corridor_forbidden_km = route_forbidden_km(corridor_points, forbidden_geom)

    stations = stations_for_corridor(
        corridor,
        demand,
        forbidden_geom,
        geology,
        config,
        line_id,
        weight_col=weight_col,
        water_geom=water_geom,
        route_anchor_points=anchor_points,
    )

    final_points = list(stations.geometry)
    final_metrics = route_score_from_points(
        final_points,
        demand,
        forbidden_geom,
        geology,
        config,
        weight_col=weight_col,
        water_crossings=water_crossings,
        transfer_points=transfer_points,
        transfer_lines=transfer_lines,
        existing_lines=transfer_lines,
    )
    previous_overlap_km = final_metrics["line_overlap_km"]
    previous_overlap_penalties = spatial_limit_penalties(0.0, previous_overlap_km, config)
    previous_forbidden_km = final_metrics["forbidden_km"]
    previous_forbidden_penalties = forbidden_limit_penalties(previous_forbidden_km, config)
    corridor_forbidden_penalties = forbidden_limit_penalties(corridor_forbidden_km, config)
    corridor_limit_penalties = spatial_limit_penalties(outside_city_km, corridor_overlap_km, config)
    final_metrics["score"] += previous_forbidden_km * config.forbidden_penalty_per_km
    final_metrics["score"] += previous_forbidden_penalties["forbidden_excess_penalty"]
    final_metrics["score"] -= corridor_forbidden_km * config.forbidden_penalty_per_km
    final_metrics["score"] -= corridor_forbidden_penalties["forbidden_excess_penalty"]
    final_metrics["score"] += previous_overlap_km * config.line_overlap_penalty_per_km
    final_metrics["score"] += previous_overlap_penalties["line_overlap_excess_penalty"]
    final_metrics["score"] -= outside_city_penalty
    final_metrics["score"] -= corridor_overlap_km * config.line_overlap_penalty_per_km
    final_metrics["score"] -= corridor_limit_penalties["outside_city_excess_penalty"]
    final_metrics["score"] -= corridor_limit_penalties["line_overlap_excess_penalty"]
    final_metrics["outside_city_km"] = outside_city_km
    final_metrics["outside_city_penalty"] = outside_city_penalty
    final_metrics["outside_city_excess_km"] = corridor_limit_penalties["outside_city_excess_km"]
    final_metrics["outside_city_excess_penalty"] = corridor_limit_penalties["outside_city_excess_penalty"]
    final_metrics["forbidden_km"] = corridor_forbidden_km
    final_metrics["forbidden_excess_km"] = corridor_forbidden_penalties["forbidden_excess_km"]
    final_metrics["forbidden_excess_penalty"] = corridor_forbidden_penalties["forbidden_excess_penalty"]
    final_metrics["line_overlap_km"] = corridor_overlap_km
    final_metrics["line_overlap_excess_km"] = corridor_limit_penalties["line_overlap_excess_km"]
    final_metrics["line_overlap_excess_penalty"] = corridor_limit_penalties["line_overlap_excess_penalty"]
    final_metrics["direction_priority_bonus"] = anchor_metrics.get("direction_priority_bonus", 0.0)
    final_metrics["direction_priority_sector_count"] = anchor_metrics.get("direction_priority_sector_count", 0)
    final_metrics["score"] += final_metrics["direction_priority_bonus"]
    estimated_cost, base_cost, flood_extra_cost = construction_cost_mln(
        corridor.length / 1_000.0,
        final_metrics["forbidden_km"],
        final_metrics["geology_factor"],
        config,
    )

    anchor_data = {
        "line_id": line_id,
        "anchor_order": np.arange(1, len(selected) + 1),
        "candidate_id": [item["candidate_id"] for item in selected],
        "name": [item["name"] for item in selected],
        "source": [item["source"] for item in selected],
    }
    optional_anchor_fields = [
        "candidate_weight",
        "base_candidate_weight",
        "catchment_weight",
        "risk_safety_factor",
        "anchor_relocated_m",
        "in_flood_zone",
        "in_flood_buffer",
        "distance_to_flood_m",
        "in_water_zone",
        "in_water_buffer",
        "distance_to_water_m",
        "geology_factor",
        "high_geology",
        "geology_excess_factor",
        "geology_weight_factor",
        "anchor_geology_penalty",
        "near_required_centre",
        "distance_to_required_centre_m",
        "direction_sector",
        "direction_octant",
    ]
    for field in optional_anchor_fields:
        anchor_data[field] = [item.get(field, np.nan) for item in selected]

    anchors = gpd.GeoDataFrame(
        anchor_data,
        geometry=[item["geometry"] for item in selected],
        crs=demand.crs,
    )

    line_gdf = gpd.GeoDataFrame(
        {
            "line_id": [line_id],
            "algorithm": ["orienteering_greedy_insertion_2opt"],
            "anchor_count": [len(selected)],
            "served_weight": [final_metrics["served_weight"]],
            "served_share": [final_metrics["served_share"]],
            "forbidden_km": [final_metrics["forbidden_km"]],
            "forbidden_excess_km": [final_metrics["forbidden_excess_km"]],
            "forbidden_excess_penalty": [final_metrics["forbidden_excess_penalty"]],
            "geology_factor": [final_metrics["geology_factor"]],
            "geology_excess_km": [final_metrics["geology_excess_km"]],
            "high_geology_km": [final_metrics["high_geology_km"]],
            "station_geology_penalty": [final_metrics["station_geology_penalty"]],
            "anchor_geology_penalty": [anchor_metrics["anchor_geology_penalty"]],
            "station_flood_zone_count": [int(stations["in_station_flood_zone"].sum())],
            "station_water_zone_count": [int(stations["in_station_water_zone"].sum())],
            "station_flood_buffer_count": [int(stations["in_station_flood_buffer"].sum())],
            "station_water_buffer_count": [int(stations["in_station_water_buffer"].sum())],
            "water_crossing_km": [final_metrics["water_crossing_km"]],
            "outside_city_km": [outside_city_km],
            "outside_city_penalty": [outside_city_penalty],
            "outside_city_excess_km": [final_metrics["outside_city_excess_km"]],
            "outside_city_excess_penalty": [final_metrics["outside_city_excess_penalty"]],
            "transfer_score": [final_metrics["transfer_score"]],
            "transfer_count": [final_metrics["transfer_count"]],
            "direction_priority_bonus": [final_metrics["direction_priority_bonus"]],
            "direction_priority_sector_count": [final_metrics["direction_priority_sector_count"]],
            "line_overlap_km": [final_metrics["line_overlap_km"]],
            "line_overlap_excess_km": [final_metrics["line_overlap_excess_km"]],
            "line_overlap_excess_penalty": [final_metrics["line_overlap_excess_penalty"]],
            "arm_reuse_penalty": [anchor_metrics.get("arm_reuse_penalty", 0.0)],
            "forced_direction_priority_anchor": [forced_direction_priority_anchor],
            "forced_southern_anchor": [forced_southern_anchor],
            "turn_penalty": [corridor_turn_metrics["turn_penalty"]],
            "max_turn_angle_deg": [corridor_turn_metrics["max_turn_angle_deg"]],
            "mean_turn_angle_deg": [corridor_turn_metrics["mean_turn_angle_deg"]],
            "sharp_turn_count": [corridor_turn_metrics["sharp_turn_count"]],
            "curve_radius_violation_m": [corridor_turn_metrics["curve_radius_violation_m"]],
            "corridor_detour_ratio": [corridor_shape["corridor_detour_ratio"]],
            "corridor_backtrack_km": [corridor_shape["corridor_backtrack_km"]],
            "corridor_shape_penalty": [corridor_shape["corridor_shape_penalty"]],
            "estimated_cost_mln": [estimated_cost],
            "base_cost_mln": [base_cost],
            "flood_extra_cost_mln": [flood_extra_cost],
            "flood_cost_multiplier": [config.flood_cost_multiplier],
            "score": [final_metrics["score"]],
            "anchor_objective_score": [anchor_metrics["score"]],
            "anchor_served_weight": [anchor_metrics["served_weight"]],
        },
        geometry=[corridor],
        crs=demand.crs,
    )

    return {
        "line": line_gdf,
        "stations": stations,
        "anchors": anchors,
        "iteration_log": pd.DataFrame(log_rows),
        "best": {
            **final_metrics,
            "anchor_objective_score": anchor_metrics["score"],
            "anchor_served_weight": anchor_metrics["served_weight"],
            "line": corridor,
            "stations": stations,
            "anchors": anchors,
        },
    }


def _node_from_candidate_row(
    row: pd.Series,
    geometry: Point | None = None,
    candidate_id: str | None = None,
) -> dict:
    node = {
        "candidate_id": candidate_id or str(row.get("candidate_id", "")),
        "name": row.get("name", str(row.get("candidate_id", "candidate"))),
        "source": row.get("source", "candidate"),
        "geometry": geometry or row.geometry,
    }
    for field in [
        "candidate_weight",
        "base_candidate_weight",
        "catchment_weight",
        "risk_safety_factor",
        "anchor_relocated_m",
        "in_flood_zone",
        "in_flood_buffer",
        "distance_to_flood_m",
        "in_water_zone",
        "in_water_buffer",
        "distance_to_water_m",
        "geology_factor",
        "high_geology",
        "geology_excess_factor",
        "geology_weight_factor",
        "anchor_geology_penalty",
        "near_required_centre",
        "distance_to_required_centre_m",
        "direction_sector",
        "direction_octant",
        "grid_selection_role",
    ]:
        if field in row:
            node[field] = row.get(field)
    return node


def _nodes_from_candidates(
    candidates: gpd.GeoDataFrame,
    centre: Point,
) -> list[dict]:
    if "required" in candidates.columns:
        required_candidates = candidates[candidates["required"].astype(bool)]
    else:
        required_candidates = candidates.iloc[0:0]

    if not required_candidates.empty:
        centre_row = required_candidates.iloc[0]
        centre_node = _node_from_candidate_row(centre_row, geometry=centre_row.geometry, candidate_id="FORCED-CENTRE")
        centre_node["source"] = "required"
        candidate_iter = candidates[~candidates.index.isin(required_candidates.index)]
    else:
        centre_node = {
            "candidate_id": "FORCED-CENTRE",
            "name": "forced centre",
            "source": "required",
            "geometry": centre,
        }
        candidate_iter = candidates

    nodes: list[dict] = [centre_node]
    for _, row in candidate_iter.iterrows():
        nodes.append(_node_from_candidate_row(row))
    return nodes


def _fast_orienteering_matrices(
    demand: gpd.GeoDataFrame,
    nodes: list[dict],
    forbidden_geom,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    weight_col: str,
    water_geom=None,
    transfer_points: gpd.GeoDataFrame | None = None,
    transfer_lines: gpd.GeoDataFrame | None = None,
    existing_lines: gpd.GeoDataFrame | None = None,
    city_boundary_geom=None,
) -> dict:
    node_xy = np.array([(item["geometry"].x, item["geometry"].y) for item in nodes], dtype=float)
    deltas = node_xy[:, None, :] - node_xy[None, :, :]
    distance_matrix = np.sqrt((deltas**2).sum(axis=2))

    forbidden_matrix = np.zeros_like(distance_matrix)
    if forbidden_geom is not None and not forbidden_geom.is_empty:
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                value = float(segment.intersection(forbidden_geom).length / 1_000.0)
                forbidden_matrix[left, right] = value
                forbidden_matrix[right, left] = value

    geology_excess_matrix = np.zeros_like(distance_matrix)
    high_geology_matrix = np.zeros_like(distance_matrix)
    node_geology_factor = np.ones(len(nodes), dtype=float)
    node_geology_penalty = np.zeros(len(nodes), dtype=float)
    if geology is not None and not geology.empty:
        for index, node in enumerate(nodes):
            cached_factor = node.get("geology_factor", np.nan)
            factor = (
                geology_factor_at_point(node["geometry"], geology)
                if pd.isna(cached_factor)
                else float(cached_factor)
            )
            node_geology_factor[index] = factor
            if node.get("source") != "required":
                cached_penalty = node.get("anchor_geology_penalty", np.nan)
                if pd.isna(cached_penalty):
                    cached_penalty = (
                        max(0.0, factor - 1.0) * config.anchor_geology_penalty_per_excess
                        + (
                            config.anchor_high_geology_penalty
                            if factor >= config.high_geology_factor_threshold
                            else 0.0
                        )
                    )
                node_geology_penalty[index] = float(cached_penalty)
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                value = geology_excess_km_for_line(segment, geology, sample_count=24)
                high_value = geology_high_factor_km_for_line(
                    segment,
                    geology,
                    threshold=config.high_geology_factor_threshold,
                    sample_count=24,
                )
                geology_excess_matrix[left, right] = value
                geology_excess_matrix[right, left] = value
                high_geology_matrix[left, right] = high_value
                high_geology_matrix[right, left] = high_value

    water_matrix = np.zeros_like(distance_matrix)
    if water_geom is not None and not water_geom.is_empty:
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                value = float(segment.intersection(water_geom).length / 1_000.0)
                water_matrix[left, right] = value
                water_matrix[right, left] = value

    outside_city_matrix = np.zeros_like(distance_matrix)
    if city_boundary_geom is not None and not city_boundary_geom.is_empty:
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                value = float(segment.difference(city_boundary_geom).length / 1_000.0)
                outside_city_matrix[left, right] = value
                outside_city_matrix[right, left] = value

    transfer_segment_matrix = np.zeros_like(distance_matrix)
    if transfer_lines is not None and not transfer_lines.empty:
        transfer_geoms = list(transfer_lines.geometry)
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                count = sum(1 for geom in transfer_geoms if segment.crosses(geom) or segment.intersects(geom))
                transfer_segment_matrix[left, right] = count
                transfer_segment_matrix[right, left] = count

    overlap_matrix = np.zeros_like(distance_matrix)
    overlap_zone_geom = None
    overlap_segments: list[tuple[LineString, float]] = []
    existing_arm_angles: list[float] = []
    overlap_lines = existing_lines if existing_lines is not None and not existing_lines.empty else transfer_lines
    if overlap_lines is not None and not overlap_lines.empty:
        overlap_geoms = [geom for geom in overlap_lines.geometry if geom is not None and not geom.is_empty]
        overlap_union = unary_union(overlap_geoms)
        if not overlap_union.is_empty:
            overlap_zone = overlap_union.buffer(config.parallel_line_buffer_m)
            overlap_zone_geom = overlap_zone
            overlap_segments = _line_segments_with_angles(overlap_geoms)
            existing_arm_angles = _existing_line_arm_angles(overlap_geoms, node_xy[0])
            for left in range(len(nodes)):
                for right in range(left + 1, len(nodes)):
                    segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                    value = float(
                        _parallel_overlap_length_m(
                            segment,
                            overlap_zone,
                            overlap_segments,
                            config,
                        )
                        / 1_000.0
                    )
                    overlap_matrix[left, right] = value
                    overlap_matrix[right, left] = value

    transfer_point_coverage = np.zeros(len(nodes), dtype=float)
    if transfer_points is not None and not transfer_points.empty:
        transfer_xy = np.array([(point.x, point.y) for point in transfer_points.geometry], dtype=float)
        if len(transfer_xy):
            distances = np.sqrt(((node_xy[:, None, :] - transfer_xy[None, :, :]) ** 2).sum(axis=2))
            transfer_point_coverage = (distances.min(axis=1) <= config.interchange_radius_m).astype(float)

    priority_sectors = {str(sector).lower() for sector in config.direction_priority_sectors}
    priority_sector_names = sorted(priority_sectors)
    priority_sector_lookup = {sector: idx for idx, sector in enumerate(priority_sector_names)}
    direction_priority_sector_index = np.full(len(nodes), -1, dtype=int)
    if priority_sectors and config.direction_priority_bonus > 0.0:
        for index, node in enumerate(nodes):
            if index == 0 or node.get("source") == "required":
                continue
            distance_to_centre = float(node.get("distance_to_required_centre_m", 0.0) or 0.0)
            if distance_to_centre < config.direction_priority_min_distance_m:
                continue
            octant = str(node.get("direction_octant", "")).lower()
            broad = str(node.get("direction_sector", "")).lower()
            matched_sector = None
            if octant in priority_sectors:
                matched_sector = octant
            elif broad in priority_sectors:
                matched_sector = broad
            if matched_sector is not None:
                direction_priority_sector_index[index] = priority_sector_lookup[matched_sector]

    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    demand_weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    node_to_demand = np.sqrt(((node_xy[:, None, :] - demand_xy[None, :, :]) ** 2).sum(axis=2))
    coverage_matrix = np.clip(1.0 - node_to_demand / config.walk_radius_m, 0.0, 1.0)

    return {
        "node_xy": node_xy,
        "forbidden_geom": forbidden_geom,
        "city_boundary_geom": city_boundary_geom,
        "distance_matrix": distance_matrix,
        "forbidden_matrix": forbidden_matrix,
        "geology_excess_matrix": geology_excess_matrix,
        "high_geology_matrix": high_geology_matrix,
        "node_geology_factor": node_geology_factor,
        "node_geology_penalty": node_geology_penalty,
        "water_matrix": water_matrix,
        "outside_city_matrix": outside_city_matrix,
        "transfer_segment_matrix": transfer_segment_matrix,
        "transfer_point_coverage": transfer_point_coverage,
        "direction_priority_sector_index": direction_priority_sector_index,
        "direction_priority_sector_names": priority_sector_names,
        "overlap_matrix": overlap_matrix,
        "overlap_zone_geom": overlap_zone_geom,
        "overlap_segments": overlap_segments,
        "existing_arm_angles": existing_arm_angles,
        "coverage_matrix": coverage_matrix,
        "demand_weights": demand_weights,
        "total_weight": float(demand_weights.sum()),
    }


def _fast_order_metrics(
    order: list[int],
    matrices: Mapping,
    config: MetroConfig,
    extend_to_length: bool = True,
) -> dict:
    if not order:
        return {
            "score": 0.0,
            "served_weight": 0.0,
            "served_share": 0.0,
            "route_length_m": 0.0,
            "forbidden_km": 0.0,
            "forbidden_excess_km": 0.0,
            "forbidden_excess_penalty": 0.0,
            "geology_factor": 1.0,
            "geology_excess_km": 0.0,
            "high_geology_km": 0.0,
            "anchor_geology_penalty": 0.0,
            "line_overlap_km": 0.0,
            "line_overlap_excess_km": 0.0,
            "line_overlap_excess_penalty": 0.0,
            "outside_city_km": 0.0,
            "outside_city_penalty": 0.0,
            "outside_city_excess_km": 0.0,
            "outside_city_excess_penalty": 0.0,
            **_empty_turn_metrics(),
            **_empty_corridor_shape_metrics(),
        }

    node_xy = matrices["node_xy"]
    forbidden_geom = matrices.get("forbidden_geom")
    city_boundary_geom = matrices.get("city_boundary_geom")
    distance_matrix = matrices["distance_matrix"]
    forbidden_matrix = matrices["forbidden_matrix"]
    geology_excess_matrix = matrices.get("geology_excess_matrix", np.zeros_like(distance_matrix))
    high_geology_matrix = matrices.get("high_geology_matrix", np.zeros_like(distance_matrix))
    node_geology_penalty = matrices.get("node_geology_penalty", np.zeros(len(distance_matrix)))
    water_matrix = matrices.get("water_matrix", np.zeros_like(distance_matrix))
    outside_city_matrix = matrices.get("outside_city_matrix", np.zeros_like(distance_matrix))
    transfer_segment_matrix = matrices.get("transfer_segment_matrix", np.zeros_like(distance_matrix))
    transfer_point_coverage = matrices.get("transfer_point_coverage", np.zeros(len(distance_matrix)))
    direction_priority_sector_index = matrices.get(
        "direction_priority_sector_index",
        np.full(len(distance_matrix), -1, dtype=int),
    )
    overlap_matrix = matrices.get("overlap_matrix", np.zeros_like(distance_matrix))
    overlap_zone_geom = matrices.get("overlap_zone_geom")
    overlap_segments = matrices.get("overlap_segments", [])
    existing_arm_angles = matrices.get("existing_arm_angles", [])
    coverage_matrix = matrices["coverage_matrix"]
    demand_weights = matrices["demand_weights"]
    total_weight = matrices["total_weight"]

    if len(order) >= 2:
        left = np.array(order[:-1], dtype=int)
        right = np.array(order[1:], dtype=int)
        length_m = float(distance_matrix[left, right].sum())
        forbidden_km = float(forbidden_matrix[left, right].sum())
        geology_excess_km = float(geology_excess_matrix[left, right].sum())
        high_geology_km = float(high_geology_matrix[left, right].sum())
        water_km = float(water_matrix[left, right].sum())
        outside_city_km = float(outside_city_matrix[left, right].sum())
        transfer_segment_count = int(min(2, transfer_segment_matrix[left, right].sum()))
        overlap_km = float(overlap_matrix[left, right].sum())
    else:
        length_m = 0.0
        forbidden_km = 0.0
        geology_excess_km = 0.0
        high_geology_km = 0.0
        water_km = 0.0
        outside_city_km = 0.0
        transfer_segment_count = 0
        overlap_km = 0.0

    order_array = np.array(order, dtype=int)
    has_city_boundary = city_boundary_geom is not None and not city_boundary_geom.is_empty
    has_forbidden_geom = forbidden_geom is not None and not forbidden_geom.is_empty
    has_overlap_zone = overlap_zone_geom is not None and not overlap_zone_geom.is_empty
    if extend_to_length and len(order) >= 2 and (has_forbidden_geom or has_city_boundary or has_overlap_zone):
        full_corridor_points = [Point(float(x), float(y)) for x, y in node_xy[order_array]]
        full_corridor_points = extend_route_points_to_length(
            full_corridor_points,
            config.length_m,
            city_boundary_geom,
        )
        full_corridor = route_polyline(full_corridor_points)
        if has_forbidden_geom:
            forbidden_km = float(full_corridor.intersection(forbidden_geom).length / 1_000.0)
        if has_city_boundary:
            outside_city_km = line_outside_city_km(full_corridor, city_boundary_geom)
        if has_overlap_zone:
            overlap_km = float(
                _parallel_overlap_length_m(
                    full_corridor,
                    overlap_zone_geom,
                    overlap_segments,
                    config,
                )
                / 1_000.0
            )

    coverage = coverage_matrix[order_array].max(axis=0)
    served_weight = float(np.sum(demand_weights * coverage))
    anchor_geology_penalty = float(node_geology_penalty[order_array].sum())
    transfer_point_count = int(min(2, transfer_point_coverage[order_array].sum()))
    transfer_count = int(min(3, transfer_point_count + transfer_segment_count))
    transfer_score = float(transfer_count * config.transfer_bonus_per_interchange)
    selected_priority_sectors = {
        int(sector_index)
        for sector_index in direction_priority_sector_index[order_array]
        if int(sector_index) >= 0
    }
    direction_priority_sector_count = min(
        len(selected_priority_sectors),
        max(0, int(config.direction_priority_max_per_route)),
    )
    direction_priority_bonus = float(direction_priority_sector_count * config.direction_priority_bonus)
    route_km = length_m / 1_000.0
    geology_factor = 1.0 + geology_excess_km / route_km if route_km else 1.0
    forbidden_limit = forbidden_limit_penalties(forbidden_km, config)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km
    limit_penalties = spatial_limit_penalties(outside_city_km, overlap_km, config)
    arm_reuse_penalty = _arm_reuse_penalty(order, node_xy, existing_arm_angles, config)
    turn_metrics = _turn_metrics_from_xy(node_xy[order_array], config)
    shape_metrics = _corridor_shape_metrics_from_xy(node_xy[order_array], config)
    score = served_weight - forbidden_km * config.forbidden_penalty_per_km
    score -= forbidden_limit["forbidden_excess_penalty"]
    score -= geology_excess_km * config.geology_penalty_per_km
    score -= high_geology_km * config.high_geology_penalty_per_km
    score -= anchor_geology_penalty
    score -= outside_city_penalty
    score -= overlap_km * config.line_overlap_penalty_per_km
    score -= limit_penalties["outside_city_excess_penalty"]
    score -= limit_penalties["line_overlap_excess_penalty"]
    score -= arm_reuse_penalty
    score -= turn_metrics["turn_penalty"]
    score -= shape_metrics["corridor_shape_penalty"]
    score += water_km * config.river_crossing_bonus_per_km
    score += transfer_score
    score += direction_priority_bonus
    return {
        "score": score,
        "served_weight": served_weight,
        "served_share": served_weight / total_weight if total_weight else 0.0,
        "route_length_m": length_m,
        "forbidden_km": forbidden_km,
        "forbidden_excess_km": forbidden_limit["forbidden_excess_km"],
        "forbidden_excess_penalty": forbidden_limit["forbidden_excess_penalty"],
        "geology_factor": geology_factor,
        "geology_excess_km": geology_excess_km,
        "high_geology_km": high_geology_km,
        "anchor_geology_penalty": anchor_geology_penalty,
        "water_crossing_km": water_km,
        "transfer_score": transfer_score,
        "transfer_count": transfer_count,
        "direction_priority_bonus": direction_priority_bonus,
        "direction_priority_sector_count": direction_priority_sector_count,
        "line_overlap_km": overlap_km,
        "line_overlap_excess_km": limit_penalties["line_overlap_excess_km"],
        "line_overlap_excess_penalty": limit_penalties["line_overlap_excess_penalty"],
        "arm_reuse_penalty": arm_reuse_penalty,
        "outside_city_km": outside_city_km,
        "outside_city_penalty": outside_city_penalty,
        "outside_city_excess_km": limit_penalties["outside_city_excess_km"],
        "outside_city_excess_penalty": limit_penalties["outside_city_excess_penalty"],
        **turn_metrics,
        **shape_metrics,
    }


def exhaustive_route_count(candidate_count: int, max_selected: int | None = None) -> int:
    """Count open routes through a required centre for a brute-force search."""

    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    max_selected = candidate_count if max_selected is None else min(max_selected, candidate_count)
    total = 0
    arrangements = 1
    for selected_count in range(0, max_selected + 1):
        if selected_count == 0:
            arrangements = 1
        elif selected_count == 1:
            arrangements = candidate_count
        else:
            arrangements *= candidate_count - selected_count + 1
        total += arrangements * (selected_count + 1)
    return int(total)


def complexity_growth_table(candidate_counts: Iterable[int], max_selected: int | None = None) -> pd.DataFrame:
    """Small table showing why exact TSP/orienteering search explodes."""

    rows = []
    for candidate_count in candidate_counts:
        exact_routes = exhaustive_route_count(candidate_count, max_selected=max_selected)
        rows.append(
            {
                "candidate_count": candidate_count,
                "max_selected": candidate_count if max_selected is None else min(max_selected, candidate_count),
                "open_routes_through_centre": exact_routes,
            }
        )
    return pd.DataFrame(rows)


def solve_exact_orienteering_bruteforce(
    demand: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
    max_optional_candidates: int = 8,
    max_selected_candidates: int | None = None,
    min_anchor_spacing_m: float = 550.0,
    line_id: int = 1,
    city_boundary: gpd.GeoDataFrame | None = None,
) -> dict:
    """Brute-force exact search for a deliberately small candidate subset.

    This is not intended for the full city-scale problem. It is included for
    teaching and validation: exact enumeration becomes infeasible quickly, which
    motivates the heuristic used by ``solve_orienteering_route``.
    """

    if max_optional_candidates > 9:
        raise ValueError("Brute-force mode is intentionally capped at 9 optional candidates.")

    demand = demand.copy()
    candidates = _align_crs(candidates, demand.crs).copy()
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    city_boundary = _align_crs(city_boundary, demand.crs)
    forbidden_geom = forbidden_union(forbidden)
    city_boundary_geom = forbidden_union(city_boundary)

    if "candidate_id" not in candidates.columns:
        candidates["candidate_id"] = [f"C{i:03d}" for i in range(1, len(candidates) + 1)]
    if "candidate_weight" not in candidates.columns:
        candidates["candidate_weight"] = 0.0

    candidates = (
        candidates.sort_values(["required", "candidate_weight"], ascending=[False, False])
        .head(max_optional_candidates)
        .reset_index(drop=True)
    )

    nodes = _nodes_from_candidates(candidates, centre)
    matrices = _fast_orienteering_matrices(
        demand,
        nodes,
        forbidden_geom,
        geology,
        config,
        weight_col,
        city_boundary_geom=city_boundary_geom,
    )
    distance_matrix = matrices["distance_matrix"]
    optional_ids = list(range(1, len(nodes)))
    if max_selected_candidates is None:
        max_selected = min(len(optional_ids), max(0, int(config.route_anchor_count) - 1))
    else:
        max_selected = min(max_selected_candidates, len(optional_ids), max(0, int(config.route_anchor_count) - 1))

    best_order = [0]
    best_metrics = _fast_order_metrics(best_order, matrices, config)
    evaluated_routes = 1

    for selected_count in range(1, max_selected + 1):
        for route_without_centre in permutations(optional_ids, selected_count):
            for centre_position in range(selected_count + 1):
                order = list(route_without_centre[:centre_position]) + [0] + list(route_without_centre[centre_position:])
                evaluated_routes += 1

                too_close = False
                for left_index, left_node in enumerate(order):
                    for right_node in order[left_index + 1 :]:
                        if left_node != right_node and distance_matrix[left_node, right_node] < min_anchor_spacing_m:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    continue

                metrics = _fast_order_metrics(order, matrices, config)
                if metrics["route_length_m"] > config.length_m:
                    continue
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                if metrics["score"] > best_metrics["score"]:
                    best_order = order
                    best_metrics = metrics

    selected = [nodes[index] for index in best_order]
    anchor_points = [item["geometry"] for item in selected]
    corridor_points = extend_route_points_to_length(
        anchor_points,
        config.length_m,
        city_boundary_geom,
        flexible_terminal=True,
    )
    corridor = route_polyline(corridor_points)

    corridor_turn_metrics = route_turn_metrics(corridor_points, config)
    corridor_shape = corridor_shape_metrics(corridor_points, config)
    outside_city_km = line_outside_city_km(corridor, city_boundary)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km
    corridor_limit_penalties = spatial_limit_penalties(outside_city_km, 0.0, config)

    stations = stations_for_corridor(
        corridor,
        demand,
        forbidden_geom,
        geology,
        config,
        line_id,
        weight_col=weight_col,
        route_anchor_points=anchor_points,
    )

    final_metrics = route_score_from_points(
        list(stations.geometry),
        demand,
        forbidden_geom,
        geology,
        config,
        weight_col=weight_col,
    )
    final_metrics["score"] -= outside_city_penalty
    final_metrics["score"] -= corridor_limit_penalties["outside_city_excess_penalty"]
    final_metrics["outside_city_km"] = outside_city_km
    final_metrics["outside_city_penalty"] = outside_city_penalty
    final_metrics["outside_city_excess_km"] = corridor_limit_penalties["outside_city_excess_km"]
    final_metrics["outside_city_excess_penalty"] = corridor_limit_penalties["outside_city_excess_penalty"]
    estimated_cost, base_cost, flood_extra_cost = construction_cost_mln(
        corridor.length / 1_000.0,
        final_metrics["forbidden_km"],
        final_metrics["geology_factor"],
        config,
    )

    anchors = gpd.GeoDataFrame(
        {
            "line_id": line_id,
            "anchor_order": np.arange(1, len(selected) + 1),
            "candidate_id": [item["candidate_id"] for item in selected],
            "name": [item["name"] for item in selected],
            "source": [item["source"] for item in selected],
        },
        geometry=[item["geometry"] for item in selected],
        crs=demand.crs,
    )

    line_gdf = gpd.GeoDataFrame(
        {
            "line_id": [line_id],
            "algorithm": ["exact_bruteforce_small_orienteering"],
            "anchor_count": [len(selected)],
            "served_weight": [final_metrics["served_weight"]],
            "served_share": [final_metrics["served_share"]],
            "forbidden_km": [final_metrics["forbidden_km"]],
            "geology_factor": [final_metrics["geology_factor"]],
            "geology_excess_km": [final_metrics["geology_excess_km"]],
            "high_geology_km": [final_metrics["high_geology_km"]],
            "station_geology_penalty": [final_metrics["station_geology_penalty"]],
            "anchor_geology_penalty": [best_metrics["anchor_geology_penalty"]],
            "station_flood_zone_count": [int(stations["in_station_flood_zone"].sum())],
            "station_water_zone_count": [int(stations["in_station_water_zone"].sum())],
            "station_flood_buffer_count": [int(stations["in_station_flood_buffer"].sum())],
            "station_water_buffer_count": [int(stations["in_station_water_buffer"].sum())],
            "water_crossing_km": [final_metrics["water_crossing_km"]],
            "outside_city_km": [outside_city_km],
            "outside_city_penalty": [outside_city_penalty],
            "outside_city_excess_km": [final_metrics["outside_city_excess_km"]],
            "outside_city_excess_penalty": [final_metrics["outside_city_excess_penalty"]],
            "transfer_score": [final_metrics["transfer_score"]],
            "transfer_count": [final_metrics["transfer_count"]],
            "line_overlap_km": [final_metrics["line_overlap_km"]],
            "line_overlap_excess_km": [final_metrics["line_overlap_excess_km"]],
            "line_overlap_excess_penalty": [final_metrics["line_overlap_excess_penalty"]],
            "turn_penalty": [corridor_turn_metrics["turn_penalty"]],
            "max_turn_angle_deg": [corridor_turn_metrics["max_turn_angle_deg"]],
            "mean_turn_angle_deg": [corridor_turn_metrics["mean_turn_angle_deg"]],
            "sharp_turn_count": [corridor_turn_metrics["sharp_turn_count"]],
            "curve_radius_violation_m": [corridor_turn_metrics["curve_radius_violation_m"]],
            "corridor_detour_ratio": [corridor_shape["corridor_detour_ratio"]],
            "corridor_backtrack_km": [corridor_shape["corridor_backtrack_km"]],
            "corridor_shape_penalty": [corridor_shape["corridor_shape_penalty"]],
            "estimated_cost_mln": [estimated_cost],
            "base_cost_mln": [base_cost],
            "flood_extra_cost_mln": [flood_extra_cost],
            "flood_cost_multiplier": [config.flood_cost_multiplier],
            "score": [final_metrics["score"]],
            "anchor_objective_score": [best_metrics["score"]],
            "anchor_served_weight": [best_metrics["served_weight"]],
            "evaluated_routes": [evaluated_routes],
        },
        geometry=[corridor],
        crs=demand.crs,
    )

    return {
        "line": line_gdf,
        "stations": stations,
        "anchors": anchors,
        "iteration_log": pd.DataFrame(
            [
                {
                    "algorithm": "exact_bruteforce_small_orienteering",
                    "candidate_count": len(optional_ids),
                    "max_selected": max_selected,
                    "evaluated_routes": evaluated_routes,
                    **best_metrics,
                }
            ]
        ),
        "best": {
            **final_metrics,
            "anchor_objective_score": best_metrics["score"],
            "evaluated_routes": evaluated_routes,
            "line": corridor,
            "stations": stations,
            "anchors": anchors,
        },
    }


def compare_exact_and_heuristic(
    demand: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
    max_optional_candidates: int = 8,
    max_selected_candidates: int | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compare exact enumeration and heuristic on the same small candidate set."""

    subset = (
        candidates.sort_values(["required", "candidate_weight"], ascending=[False, False])
        .head(max_optional_candidates)
        .reset_index(drop=True)
    )
    exact = solve_exact_orienteering_bruteforce(
        demand,
        subset,
        centre,
        forbidden=forbidden,
        geology=geology,
        config=config,
        weight_col=weight_col,
        max_optional_candidates=max_optional_candidates,
        max_selected_candidates=max_selected_candidates,
        city_boundary=city_boundary,
    )
    heuristic = solve_orienteering_route(
        demand,
        subset,
        centre,
        forbidden=forbidden,
        geology=geology,
        config=config,
        weight_col=weight_col,
        force_station_count=False,
        city_boundary=city_boundary,
    )
    rows = []
    for label, result in [("exact", exact), ("heuristic", heuristic)]:
        line = result["line"].iloc[0]
        rows.append(
            {
                "algorithm": label,
                "anchor_count": int(line["anchor_count"]),
                "anchor_objective_score": float(line["anchor_objective_score"]),
                "anchor_served_weight": float(line["anchor_served_weight"]),
                "served_weight": float(line["served_weight"]),
                "served_share": float(line["served_share"]),
                "final_score": float(line["score"]),
                "forbidden_km": float(line["forbidden_km"]),
                "outside_city_km": float(line.get("outside_city_km", 0.0)),
                "outside_city_excess_km": float(line.get("outside_city_excess_km", 0.0)),
                "geology_factor": float(line["geology_factor"]),
                "geology_excess_km": float(line.get("geology_excess_km", 0.0)),
                "high_geology_km": float(line.get("high_geology_km", 0.0)),
                "anchor_geology_penalty": float(line.get("anchor_geology_penalty", 0.0)),
                "station_geology_penalty": float(line.get("station_geology_penalty", 0.0)),
                "station_flood_zone_count": int(line.get("station_flood_zone_count", 0)),
                "station_water_zone_count": int(line.get("station_water_zone_count", 0)),
                "station_flood_buffer_count": int(line.get("station_flood_buffer_count", 0)),
                "station_water_buffer_count": int(line.get("station_water_buffer_count", 0)),
                "max_turn_angle_deg": float(line.get("max_turn_angle_deg", 0.0)),
                "sharp_turn_count": int(line.get("sharp_turn_count", 0)),
                "corridor_detour_ratio": float(line.get("corridor_detour_ratio", 1.0)),
                "corridor_backtrack_km": float(line.get("corridor_backtrack_km", 0.0)),
                "line_overlap_km": float(line.get("line_overlap_km", 0.0)),
                "line_overlap_excess_km": float(line.get("line_overlap_excess_km", 0.0)),
                "evaluated_routes": int(line.get("evaluated_routes", len(result["iteration_log"]))),
            }
        )
    table = pd.DataFrame(rows)
    best_exact = table.loc[table["algorithm"].eq("exact"), "anchor_objective_score"].iloc[0]
    table["anchor_score_vs_exact"] = table["anchor_objective_score"] / best_exact if best_exact else np.nan
    return table, {"exact": exact, "heuristic": heuristic}


def line_through_anchor(anchor: Point, angle_deg: float, length_m: float) -> LineString:
    angle = radians(angle_deg)
    half = length_m / 2.0
    dx = cos(angle) * half
    dy = sin(angle) * half
    return LineString([(anchor.x - dx, anchor.y - dy), (anchor.x + dx, anchor.y + dy)])


def stations_along_line(line: LineString, station_count: int, crs) -> gpd.GeoDataFrame:
    distances = np.linspace(0.0, line.length, station_count)
    stations = [line.interpolate(distance) for distance in distances]
    return gpd.GeoDataFrame(
        {
            "station_id": np.arange(1, station_count + 1),
            "distance_m": distances,
        },
        geometry=stations,
        crs=crs,
    )


def accessibility_to_stations(
    demand: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    weight_col: str = "population",
    walk_radius_m: float = 800.0,
) -> gpd.GeoDataFrame:
    """Compute a simple linear-decay station catchment score for each point."""

    demand = demand.copy()
    stations = _align_crs(stations, demand.crs)
    station_geoms = list(stations.geometry)
    if not station_geoms:
        demand["nearest_station_m"] = np.inf
        demand["coverage_score"] = 0.0
        demand["served_weight"] = 0.0
        return demand

    nearest = [min(point.distance(station) for station in station_geoms) for point in demand.geometry]
    demand["nearest_station_m"] = nearest
    demand["coverage_score"] = np.clip(1.0 - demand["nearest_station_m"] / walk_radius_m, 0.0, 1.0)
    demand["served_weight"] = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0) * demand["coverage_score"]
    return demand


def accessibility_to_corridors(
    demand: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    weight_col: str = "population",
    corridor_radius_m: float = 2_200.0,
) -> gpd.GeoDataFrame:
    """Compute linear-decay catchment along already planned line corridors."""

    demand = demand.copy()
    lines = _align_crs(lines, demand.crs)
    line_geoms = [geom for geom in lines.geometry if geom is not None and not geom.is_empty]
    if not line_geoms or corridor_radius_m <= 0.0:
        demand["nearest_corridor_m"] = np.inf
        demand["corridor_coverage_score"] = 0.0
        demand["corridor_served_weight"] = 0.0
        return demand

    nearest = [min(point.distance(line) for line in line_geoms) for point in demand.geometry]
    demand["nearest_corridor_m"] = nearest
    demand["corridor_coverage_score"] = np.clip(
        1.0 - demand["nearest_corridor_m"] / corridor_radius_m,
        0.0,
        1.0,
    )
    demand["corridor_served_weight"] = (
        pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0) * demand["corridor_coverage_score"]
    )
    return demand


def forbidden_length_m(line: LineString, forbidden: gpd.GeoDataFrame | None) -> float:
    union = forbidden_union(forbidden)
    if union is None or union.is_empty:
        return 0.0
    return float(line.intersection(union).length)


def geology_multiplier_for_line(
    line: LineString,
    geology: gpd.GeoDataFrame | None,
    cost_col: str = "cost_factor",
    sample_count: int = 80,
) -> float:
    if line.length == 0:
        return 1.0
    excess_km = geology_excess_km_for_line(line, geology, cost_col=cost_col, sample_count=sample_count)
    return 1.0 + excess_km / (line.length / 1_000.0)


def geology_excess_km_for_line(
    line: LineString,
    geology: gpd.GeoDataFrame | None,
    cost_col: str = "cost_factor",
    sample_count: int = 80,
) -> float:
    """Length-weighted excess construction multiplier from geology/cost zones.

    A cost factor of 1.0 is neutral. A 1 km segment through a 1.35 zone adds
    0.35 geology-excess km.
    """

    if geology is None or geology.empty or cost_col not in geology.columns:
        return 0.0
    if line.length == 0:
        return 0.0

    sample_count = max(2, sample_count)
    distances = (np.arange(sample_count, dtype=float) + 0.5) * line.length / sample_count
    excess_factors: list[float] = []
    for distance in distances:
        point = line.interpolate(distance)
        hits = geology[geology.geometry.covers(point)]
        if hits.empty:
            excess_factors.append(0.0)
        else:
            cost_factor = float(pd.to_numeric(hits[cost_col], errors="coerce").fillna(1.0).max())
            excess_factors.append(max(0.0, cost_factor - 1.0))
    return float(np.mean(excess_factors) * line.length / 1_000.0)


def geology_high_factor_km_for_line(
    line: LineString,
    geology: gpd.GeoDataFrame | None,
    cost_col: str = "cost_factor",
    threshold: float = 1.50,
    sample_count: int = 80,
) -> float:
    """Approximate line length crossing very difficult geology/cost zones."""

    if geology is None or geology.empty or cost_col not in geology.columns:
        return 0.0
    if line.length == 0:
        return 0.0

    sample_count = max(2, sample_count)
    distances = (np.arange(sample_count, dtype=float) + 0.5) * line.length / sample_count
    high_samples = 0
    for distance in distances:
        point = line.interpolate(distance)
        if geology_factor_at_point(point, geology, cost_col=cost_col) >= threshold:
            high_samples += 1
    return float((high_samples / sample_count) * line.length / 1_000.0)


def geology_factor_at_point(
    point: Point,
    geology: gpd.GeoDataFrame | None,
    cost_col: str = "cost_factor",
) -> float:
    if geology is None or geology.empty or cost_col not in geology.columns:
        return 1.0
    hits = geology[geology.geometry.covers(point)]
    if hits.empty:
        return 1.0
    return float(pd.to_numeric(hits[cost_col], errors="coerce").fillna(1.0).max())


def geology_point_penalty(
    points: Iterable[Point],
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    penalty_per_excess: float,
    high_penalty: float,
    cost_col: str = "cost_factor",
) -> float:
    if geology is None or geology.empty or cost_col not in geology.columns:
        return 0.0
    penalty = 0.0
    for point in points:
        factor = geology_factor_at_point(point, geology, cost_col=cost_col)
        penalty += max(0.0, factor - 1.0) * penalty_per_excess
        if factor >= config.high_geology_factor_threshold:
            penalty += high_penalty
    return float(penalty)


def score_line(
    line: LineString,
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    config: MetroConfig,
    weight_col: str = "population",
) -> dict:
    stations = stations_along_line(line, config.station_count, demand.crs)
    served = accessibility_to_stations(demand, stations, weight_col=weight_col, walk_radius_m=config.walk_radius_m)
    forbidden_km = forbidden_length_m(line, forbidden) / 1_000.0
    geology_excess_km = geology_excess_km_for_line(line, geology)
    geology_factor = 1.0 + geology_excess_km / (line.length / 1_000.0) if line.length else 1.0
    high_geology_km = geology_high_factor_km_for_line(
        line,
        geology,
        threshold=config.high_geology_factor_threshold,
    )
    station_geology_penalty = geology_point_penalty(
        list(stations.geometry),
        geology,
        config,
        penalty_per_excess=config.station_geology_penalty_per_excess,
        high_penalty=config.station_high_geology_penalty,
    )
    raw_served = float(served["served_weight"].sum())
    score = raw_served - forbidden_km * config.forbidden_penalty_per_km
    score -= geology_excess_km * config.geology_penalty_per_km
    score -= high_geology_km * config.high_geology_penalty_per_km
    score -= station_geology_penalty
    return {
        "score": score,
        "served_weight": raw_served,
        "forbidden_km": forbidden_km,
        "geology_factor": geology_factor,
        "geology_excess_km": geology_excess_km,
        "high_geology_km": high_geology_km,
        "station_geology_penalty": station_geology_penalty,
        "stations": stations,
    }


def optimise_radial_line(
    demand: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    weight_col: str = "population",
) -> tuple[dict, pd.DataFrame]:
    """Search line angles and return the best radial line through the centre."""

    demand = demand.copy()
    forbidden = _align_crs(forbidden, demand.crs)
    geology = _align_crs(geology, demand.crs)
    rows = []
    best: dict | None = None
    for angle in np.arange(0.0, 180.0, config.angle_step_deg):
        line = line_through_anchor(centre, float(angle), config.length_m)
        result = score_line(line, demand, forbidden, geology, config, weight_col=weight_col)
        result["angle_deg"] = float(angle)
        result["line"] = line
        rows.append({k: v for k, v in result.items() if k not in {"stations", "line"}})
        if best is None or result["score"] > best["score"]:
            best = result

    if best is None:
        raise RuntimeError("No line candidates were generated.")

    best["line_gdf"] = gpd.GeoDataFrame(
        {
            "angle_deg": [best["angle_deg"]],
            "served_weight": [best["served_weight"]],
            "forbidden_km": [best["forbidden_km"]],
            "geology_factor": [best["geology_factor"]],
            "geology_excess_km": [best["geology_excess_km"]],
            "high_geology_km": [best["high_geology_km"]],
            "station_geology_penalty": [best["station_geology_penalty"]],
            "score": [best["score"]],
        },
        geometry=[best["line"]],
        crs=demand.crs,
    )
    return best, pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def plan_network(
    demand: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    n_lines: int = 1,
    weight_col: str = "population",
    name: str = "scenario",
) -> dict:
    """Plan one or more radial lines. Later lines optimise residual demand."""

    residual = demand.copy()
    residual["_planning_weight"] = pd.to_numeric(residual[weight_col], errors="coerce").fillna(0.0)
    lines = []
    candidate_tables = []
    for line_id in range(1, n_lines + 1):
        best, candidates = optimise_radial_line(
            residual,
            centre=centre,
            forbidden=forbidden,
            geology=geology,
            config=config,
            weight_col="_planning_weight",
        )
        line_gdf = best["line_gdf"].copy()
        line_gdf["line_id"] = line_id
        stations = best["stations"].copy()
        stations["line_id"] = line_id
        stations["station_code"] = [f"L{line_id:02d}-S{sid:02d}" for sid in stations["station_id"]]
        lines.append({"line": line_gdf, "stations": stations, "best": best})
        candidates["line_id"] = line_id
        candidate_tables.append(candidates)

        coverage = accessibility_to_stations(
            residual,
            stations,
            weight_col="_planning_weight",
            walk_radius_m=config.walk_radius_m,
        )
        used_coverage = np.clip(coverage["coverage_score"] * config.residual_coverage_multiplier, 0.0, 1.0)
        residual["_planning_weight"] = residual["_planning_weight"] * (1.0 - used_coverage)

    return {
        "name": name,
        "config": config,
        "lines": lines,
        "candidates": pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame(),
    }


def plan_orienteering_network(
    demand: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    n_lines: int = 1,
    weight_col: str = "population",
    name: str = "orienteering scenario",
) -> dict:
    """Plan a network with an NP-hard orienteering formulation and a heuristic."""

    residual = demand.copy()
    residual["_planning_weight"] = pd.to_numeric(residual[weight_col], errors="coerce").fillna(0.0)
    lines = []
    logs = []
    previous_station_frames: list[gpd.GeoDataFrame] = []
    previous_line_frames: list[gpd.GeoDataFrame] = []

    for line_id in range(1, n_lines + 1):
        transfer_points = (
            gpd.GeoDataFrame(pd.concat(previous_station_frames, ignore_index=True), crs=demand.crs)
            if previous_station_frames
            else None
        )
        transfer_lines = (
            gpd.GeoDataFrame(pd.concat(previous_line_frames, ignore_index=True), crs=demand.crs)
            if previous_line_frames
            else None
        )
        result = solve_orienteering_route(
            demand=residual,
            candidates=candidates,
            centre=centre,
            forbidden=forbidden,
            geology=geology,
            water_crossings=water_crossings,
            transfer_points=transfer_points,
            transfer_lines=transfer_lines,
            city_boundary=city_boundary,
            config=config,
            weight_col="_planning_weight",
            force_station_count=True,
            line_id=line_id,
        )
        lines.append(result)
        previous_station_frames.append(result["stations"])
        previous_line_frames.append(result["line"])
        line_log = result["iteration_log"].copy()
        line_log["line_id"] = line_id
        logs.append(line_log)

        coverage = accessibility_to_stations(
            residual,
            result["stations"],
            weight_col="_planning_weight",
            walk_radius_m=config.walk_radius_m,
        )
        used_coverage = np.clip(coverage["coverage_score"] * config.residual_coverage_multiplier, 0.0, 1.0)
        residual["_planning_weight"] = residual["_planning_weight"] * (1.0 - used_coverage)

    return {
        "name": name,
        "config": config,
        "algorithm": "orienteering_greedy_insertion_2opt",
        "lines": lines,
        "candidates": pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(),
    }


def plan_grid_station_network(
    demand: gpd.GeoDataFrame,
    station_candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    config: MetroConfig = MetroConfig(),
    n_lines: int = 1,
    weight_col: str = "population",
    name: str = "grid station scenario",
) -> dict:
    """Plan a network directly from grid station candidates, without route anchors."""

    residual = demand.copy()
    residual["_planning_weight"] = pd.to_numeric(residual[weight_col], errors="coerce").fillna(0.0)
    lines = []
    logs = []
    previous_station_frames: list[gpd.GeoDataFrame] = []
    previous_line_frames: list[gpd.GeoDataFrame] = []

    for line_id in range(1, n_lines + 1):
        transfer_points = (
            gpd.GeoDataFrame(pd.concat(previous_station_frames, ignore_index=True), crs=demand.crs)
            if previous_station_frames
            else None
        )
        transfer_lines = (
            gpd.GeoDataFrame(pd.concat(previous_line_frames, ignore_index=True), crs=demand.crs)
            if previous_line_frames
            else None
        )
        result = solve_grid_station_route(
            demand=residual,
            station_candidates=station_candidates,
            centre=centre,
            forbidden=forbidden,
            geology=geology,
            water_crossings=water_crossings,
            transfer_points=transfer_points,
            transfer_lines=transfer_lines,
            city_boundary=city_boundary,
            config=config,
            weight_col="_planning_weight",
            line_id=line_id,
            force_station_count=True,
        )
        lines.append(result)
        previous_station_frames.append(result["stations"])
        previous_line_frames.append(result["line"])
        line_log = result["iteration_log"].copy()
        line_log["line_id"] = line_id
        logs.append(line_log)

        coverage = accessibility_to_stations(
            residual,
            result["stations"],
            weight_col="_planning_weight",
            walk_radius_m=config.walk_radius_m,
        )
        corridor_coverage = accessibility_to_corridors(
            residual,
            result["line"],
            weight_col="_planning_weight",
            corridor_radius_m=config.residual_corridor_radius_m,
        )
        station_used_coverage = coverage["coverage_score"] * config.residual_coverage_multiplier
        corridor_used_coverage = (
            corridor_coverage["corridor_coverage_score"] * config.residual_corridor_coverage_multiplier
        )
        used_coverage = np.clip(np.maximum(station_used_coverage, corridor_used_coverage), 0.0, 1.0)
        residual["_planning_weight"] = residual["_planning_weight"] * (1.0 - used_coverage)

    return {
        "name": name,
        "config": config,
        "algorithm": "grid_station_orienteering",
        "lines": lines,
        "candidates": pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(),
        "station_candidates": station_candidates,
    }


def remove_stations_in_forbidden_zones(
    scenario: dict,
    forbidden: gpd.GeoDataFrame | None,
) -> dict:
    """Return a scenario copy with stations/entrances inside forbidden polygons removed."""

    out = dict(scenario)
    out_lines = []
    for item in scenario["lines"]:
        new_item = dict(item)
        stations = item["stations"].copy()
        forbidden_aligned = _align_crs(forbidden, stations.crs)
        union = forbidden_union(forbidden_aligned)
        if union is not None and not union.is_empty and not stations.empty:
            safe_mask = ~stations.geometry.apply(lambda point: bool(union.covers(point)))
            stations = stations.loc[safe_mask].copy()

        stations = stations.reset_index(drop=True)
        if not stations.empty:
            line_id = int(stations["line_id"].iloc[0]) if "line_id" in stations else int(item["line"]["line_id"].iloc[0])
            stations["station_id"] = np.arange(1, len(stations) + 1)
            stations["station_code"] = [f"L{line_id:02d}-S{sid:02d}" for sid in stations["station_id"]]

        line = item["line"].copy()
        removed_count = len(item["stations"]) - len(stations)
        line["removed_flood_station_count"] = removed_count
        line["station_flood_zone_count"] = 0
        if union is not None and not union.is_empty and not stations.empty:
            buffer_geom = union.buffer(scenario["config"].station_risk_buffer_m)
            line["station_flood_buffer_count"] = int(stations.geometry.apply(lambda point: bool(buffer_geom.covers(point))).sum())
        else:
            line["station_flood_buffer_count"] = 0

        new_item["stations"] = stations
        new_item["line"] = line
        if "best" in new_item:
            best = dict(new_item["best"])
            best["stations"] = stations
            new_item["best"] = best
        out_lines.append(new_item)

    out["lines"] = out_lines
    return out


def trim_scenario_lines_to_station_extent(
    scenario: dict,
    city_boundary: gpd.GeoDataFrame | None = None,
    forbidden: gpd.GeoDataFrame | None = None,
    require_stations_in_city: bool = False,
) -> dict:
    """Trim each line to its first/last remaining station.

    When ``require_stations_in_city`` is true, stations outside the city boundary
    are removed first, so the corridor ends at the first and last station still
    inside Wroclaw.
    """

    out = dict(scenario)
    config: MetroConfig = scenario["config"]
    out_lines = []
    trimmed_previous_line_frames: list[gpd.GeoDataFrame] = []

    for item in scenario["lines"]:
        new_item = dict(item)
        line = item["line"].copy()
        stations = item["stations"].copy()
        line_geom = line.geometry.iloc[0]
        original_length_km = float(line_geom.length / 1_000.0) if line_geom is not None else 0.0

        city_area = None
        if city_boundary is not None:
            city_area = forbidden_union(_align_crs(city_boundary, line.crs))

        removed_outside_city = 0
        if require_stations_in_city and city_area is not None and not city_area.is_empty and not stations.empty:
            city_area_for_stations = forbidden_union(_align_crs(city_boundary, stations.crs))
            inside_mask = stations.geometry.apply(lambda point: bool(city_area_for_stations.covers(point)))
            removed_outside_city = int((~inside_mask).sum())
            stations = stations.loc[inside_mask].copy()

        if line_geom is not None and not line_geom.is_empty and line_geom.geom_type == "LineString" and len(stations) >= 2:
            projected = stations.geometry.apply(lambda point: float(line_geom.project(point))).to_numpy()
            start_m = max(0.0, float(np.nanmin(projected)))
            end_m = min(float(line_geom.length), float(np.nanmax(projected)))
            if end_m > start_m:
                trimmed_geom = substring(line_geom, start_m, end_m)
                if trimmed_geom.geom_type == "LineString" and not trimmed_geom.is_empty:
                    line = line.set_geometry(gpd.GeoSeries([trimmed_geom], index=line.index, crs=line.crs))
                    stations = stations.copy()
                    stations["distance_m"] = stations.geometry.apply(lambda point: float(trimmed_geom.project(point)))
                    stations = stations.sort_values("distance_m").reset_index(drop=True)
                    station_count = len(stations)
                    if station_count:
                        line_id = int(line["line_id"].iloc[0]) if "line_id" in line else int(stations["line_id"].iloc[0])
                        stations["station_id"] = np.arange(1, station_count + 1)
                        stations["station_code"] = [f"L{line_id:02d}-S{sid:02d}" for sid in stations["station_id"]]
                        distances = pd.to_numeric(stations["distance_m"], errors="coerce").to_numpy()
                        stations["spacing_from_previous_m"] = np.concatenate(([np.nan], np.diff(distances)))
                        stations["spacing_to_next_m"] = np.concatenate((np.diff(distances), [np.nan]))

        new_geom = line.geometry.iloc[0]
        new_length_km = float(new_geom.length / 1_000.0) if new_geom is not None else 0.0
        line["trimmed_line_tail_km"] = max(0.0, original_length_km - new_length_km)
        line["removed_outside_city_station_count"] = removed_outside_city
        if city_area is not None and not city_area.is_empty:
            outside_city_km = line_outside_city_km(new_geom, _align_crs(city_boundary, line.crs))
            outside_limit = spatial_limit_penalties(outside_city_km, 0.0, config)
            line["outside_city_km"] = outside_city_km
            line["outside_city_penalty"] = outside_city_km * config.outside_city_penalty_per_km
            line["outside_city_excess_km"] = outside_limit["outside_city_excess_km"]
            line["outside_city_excess_penalty"] = outside_limit["outside_city_excess_penalty"]

        if "line_overlap_km" in line:
            if trimmed_previous_line_frames and new_geom is not None and not new_geom.is_empty:
                previous_lines = gpd.GeoDataFrame(
                    pd.concat(trimmed_previous_line_frames, ignore_index=True),
                    crs=line.crs,
                )
                overlap_points = [Point(x, y) for x, y in new_geom.coords]
                overlap_km = line_overlap_km(overlap_points, previous_lines, config)
                overlap_limit = spatial_limit_penalties(0.0, overlap_km, config)
                line["line_overlap_km"] = overlap_km
                line["line_overlap_excess_km"] = overlap_limit["line_overlap_excess_km"]
                line["line_overlap_excess_penalty"] = overlap_limit["line_overlap_excess_penalty"]
            else:
                line["line_overlap_km"] = 0.0
                line["line_overlap_excess_km"] = 0.0
                line["line_overlap_excess_penalty"] = 0.0

        forbidden_area = forbidden_union(_align_crs(forbidden, line.crs)) if forbidden is not None else None
        if forbidden_area is not None and not forbidden_area.is_empty:
            forbidden_km = float(new_geom.intersection(forbidden_area).length / 1_000.0)
            forbidden_limit = forbidden_limit_penalties(forbidden_km, config)
            line["forbidden_km"] = forbidden_km
            line["forbidden_excess_km"] = forbidden_limit["forbidden_excess_km"]
            line["forbidden_excess_penalty"] = forbidden_limit["forbidden_excess_penalty"]

        if "geology_factor" in line:
            _, base_cost, flood_extra_cost = construction_cost_mln(
                new_length_km,
                float(line["forbidden_km"].iloc[0]) if "forbidden_km" in line else 0.0,
                float(line["geology_factor"].iloc[0]),
                config,
            )
            line["base_cost_mln"] = base_cost
            line["flood_extra_cost_mln"] = flood_extra_cost
            line["estimated_cost_mln"] = base_cost + flood_extra_cost

        new_item["line"] = line
        new_item["stations"] = stations
        if "best" in new_item:
            best = dict(new_item["best"])
            best["line"] = new_geom
            best["stations"] = stations
            new_item["best"] = best
        out_lines.append(new_item)
        trimmed_previous_line_frames.append(line)

    out["lines"] = out_lines
    return out


def scenario_geodataframes(scenario: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    line_frames = [item["line"] for item in scenario["lines"]]
    station_frames = [item["stations"] for item in scenario["lines"]]
    lines = gpd.GeoDataFrame(pd.concat(line_frames, ignore_index=True), crs=line_frames[0].crs)
    stations = gpd.GeoDataFrame(pd.concat(station_frames, ignore_index=True), crs=station_frames[0].crs)
    return lines, stations


def scenario_metrics(
    scenario: dict,
    demand: gpd.GeoDataFrame,
    weight_col: str = "population",
) -> dict:
    lines, stations = scenario_geodataframes(scenario)
    config: MetroConfig = scenario["config"]
    coverage = accessibility_to_stations(demand, stations, weight_col=weight_col, walk_radius_m=config.walk_radius_m)
    length_km = float(lines.geometry.length.sum() / 1_000.0)
    station_count = int(len(stations))
    weighted_served = float(coverage["served_weight"].sum())
    total_weight = float(pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).sum())
    served_share = weighted_served / total_weight if total_weight else 0.0
    average_factor = float(lines["geology_factor"].mean()) if "geology_factor" in lines else 1.0
    geology_excess = float(lines["geology_excess_km"].sum()) if "geology_excess_km" in lines else 0.0
    high_geology = float(lines["high_geology_km"].sum()) if "high_geology_km" in lines else 0.0
    anchor_geology_penalty = (
        float(lines["anchor_geology_penalty"].sum()) if "anchor_geology_penalty" in lines else 0.0
    )
    station_geology_penalty = (
        float(lines["station_geology_penalty"].sum()) if "station_geology_penalty" in lines else 0.0
    )
    forbidden_km = float(lines["forbidden_km"].sum()) if "forbidden_km" in lines else 0.0
    forbidden_excess = float(lines["forbidden_excess_km"].sum()) if "forbidden_excess_km" in lines else 0.0
    forbidden_excess_penalty = (
        float(lines["forbidden_excess_penalty"].sum()) if "forbidden_excess_penalty" in lines else 0.0
    )
    if "estimated_cost_mln" in lines:
        cost_mln = float(lines["estimated_cost_mln"].sum())
        base_cost_mln = float(lines["base_cost_mln"].sum()) if "base_cost_mln" in lines else length_km * config.cost_per_km_mln * average_factor
        flood_extra_cost_mln = float(lines["flood_extra_cost_mln"].sum()) if "flood_extra_cost_mln" in lines else 0.0
    else:
        cost_mln, base_cost_mln, flood_extra_cost_mln = construction_cost_mln(
            length_km,
            forbidden_km,
            average_factor,
            config,
        )
    water_crossing = float(lines["water_crossing_km"].sum()) if "water_crossing_km" in lines else 0.0
    station_flood_zone_count = (
        int(lines["station_flood_zone_count"].sum()) if "station_flood_zone_count" in lines else 0
    )
    station_water_zone_count = (
        int(lines["station_water_zone_count"].sum()) if "station_water_zone_count" in lines else 0
    )
    station_flood_buffer_count = (
        int(lines["station_flood_buffer_count"].sum()) if "station_flood_buffer_count" in lines else 0
    )
    station_water_buffer_count = (
        int(lines["station_water_buffer_count"].sum()) if "station_water_buffer_count" in lines else 0
    )
    removed_flood_station_count = (
        int(lines["removed_flood_station_count"].sum()) if "removed_flood_station_count" in lines else 0
    )
    removed_outside_city_station_count = (
        int(lines["removed_outside_city_station_count"].sum()) if "removed_outside_city_station_count" in lines else 0
    )
    trimmed_line_tail_km = float(lines["trimmed_line_tail_km"].sum()) if "trimmed_line_tail_km" in lines else 0.0
    outside_city = float(lines["outside_city_km"].sum()) if "outside_city_km" in lines else 0.0
    outside_city_penalty = float(lines["outside_city_penalty"].sum()) if "outside_city_penalty" in lines else 0.0
    outside_city_excess = float(lines["outside_city_excess_km"].sum()) if "outside_city_excess_km" in lines else 0.0
    outside_city_excess_penalty = (
        float(lines["outside_city_excess_penalty"].sum()) if "outside_city_excess_penalty" in lines else 0.0
    )
    transfer_score = float(lines["transfer_score"].sum()) if "transfer_score" in lines else 0.0
    transfer_count = int(lines["transfer_count"].sum()) if "transfer_count" in lines else 0
    direction_priority_bonus = (
        float(lines["direction_priority_bonus"].sum()) if "direction_priority_bonus" in lines else 0.0
    )
    direction_priority_sector_count = (
        int(lines["direction_priority_sector_count"].sum()) if "direction_priority_sector_count" in lines else 0
    )
    line_overlap = float(lines["line_overlap_km"].sum()) if "line_overlap_km" in lines else 0.0
    line_overlap_excess = float(lines["line_overlap_excess_km"].sum()) if "line_overlap_excess_km" in lines else 0.0
    line_overlap_excess_penalty = (
        float(lines["line_overlap_excess_penalty"].sum()) if "line_overlap_excess_penalty" in lines else 0.0
    )
    arm_reuse_penalty = float(lines["arm_reuse_penalty"].sum()) if "arm_reuse_penalty" in lines else 0.0
    forced_direction_priority_line_count = (
        int(lines["forced_direction_priority_anchor"].astype(bool).sum())
        if "forced_direction_priority_anchor" in lines
        else 0
    )
    forced_southern_line_count = (
        int(lines["forced_southern_anchor"].astype(bool).sum())
        if "forced_southern_anchor" in lines
        else 0
    )
    turn_penalty = float(lines["turn_penalty"].sum()) if "turn_penalty" in lines else 0.0
    max_turn_angle = float(lines["max_turn_angle_deg"].max()) if "max_turn_angle_deg" in lines else 0.0
    mean_turn_angle = float(lines["mean_turn_angle_deg"].mean()) if "mean_turn_angle_deg" in lines else 0.0
    sharp_turn_count = int(lines["sharp_turn_count"].sum()) if "sharp_turn_count" in lines else 0
    curve_radius_violation = (
        float(lines["curve_radius_violation_m"].sum()) if "curve_radius_violation_m" in lines else 0.0
    )
    corridor_shape_penalty = (
        float(lines["corridor_shape_penalty"].sum()) if "corridor_shape_penalty" in lines else 0.0
    )
    corridor_detour = float(lines["corridor_detour_ratio"].mean()) if "corridor_detour_ratio" in lines else 1.0
    corridor_backtrack = (
        float(lines["corridor_backtrack_km"].sum()) if "corridor_backtrack_km" in lines else 0.0
    )
    if "spacing_from_previous_m" in stations.columns:
        spacing = pd.to_numeric(stations["spacing_from_previous_m"], errors="coerce").dropna()
        avg_spacing = float(spacing.mean()) if not spacing.empty else config.station_spacing_m
        min_spacing = float(spacing.min()) if not spacing.empty else config.station_spacing_m
        max_spacing = float(spacing.max()) if not spacing.empty else config.station_spacing_m
    else:
        avg_spacing = config.station_spacing_m
        min_spacing = config.station_spacing_m
        max_spacing = config.station_spacing_m
    return {
        "scenario": scenario["name"],
        "lines": len(scenario["lines"]),
        "stations": station_count,
        "length_km": length_km,
        "avg_station_spacing_m": avg_spacing,
        "min_station_spacing_m": min_spacing,
        "max_station_spacing_m": max_spacing,
        "served_weight": weighted_served,
        "served_share": served_share,
        "avg_geology_factor": average_factor,
        "geology_excess_km": geology_excess,
        "high_geology_km": high_geology,
        "anchor_geology_penalty": anchor_geology_penalty,
        "station_geology_penalty": station_geology_penalty,
        "forbidden_km": forbidden_km,
        "forbidden_excess_km": forbidden_excess,
        "forbidden_excess_penalty": forbidden_excess_penalty,
        "station_flood_zone_count": station_flood_zone_count,
        "removed_flood_station_count": removed_flood_station_count,
        "removed_outside_city_station_count": removed_outside_city_station_count,
        "trimmed_line_tail_km": trimmed_line_tail_km,
        "station_water_zone_count": station_water_zone_count,
        "station_flood_buffer_count": station_flood_buffer_count,
        "station_water_buffer_count": station_water_buffer_count,
        "water_crossing_km": water_crossing,
        "outside_city_km": outside_city,
        "outside_city_penalty": outside_city_penalty,
        "outside_city_excess_km": outside_city_excess,
        "outside_city_excess_penalty": outside_city_excess_penalty,
        "transfer_count": transfer_count,
        "transfer_score": transfer_score,
        "direction_priority_bonus": direction_priority_bonus,
        "direction_priority_sector_count": direction_priority_sector_count,
        "line_overlap_km": line_overlap,
        "line_overlap_excess_km": line_overlap_excess,
        "line_overlap_excess_penalty": line_overlap_excess_penalty,
        "arm_reuse_penalty": arm_reuse_penalty,
        "forced_direction_priority_line_count": forced_direction_priority_line_count,
        "forced_southern_line_count": forced_southern_line_count,
        "turn_penalty": turn_penalty,
        "max_turn_angle_deg": max_turn_angle,
        "mean_turn_angle_deg": mean_turn_angle,
        "sharp_turn_count": sharp_turn_count,
        "curve_radius_violation_m": curve_radius_violation,
        "corridor_detour_ratio": corridor_detour,
        "corridor_backtrack_km": corridor_backtrack,
        "corridor_shape_penalty": corridor_shape_penalty,
        "base_cost_mln": base_cost_mln,
        "flood_extra_cost_mln": flood_extra_cost_mln,
        "estimated_cost_mln": cost_mln,
    }


def scenario_summary_table(scenarios: Mapping[str, dict], demand: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = [scenario_metrics(scenario, demand) for scenario in scenarios.values()]
    table = pd.DataFrame(rows)
    for col in [
        "length_km",
        "avg_station_spacing_m",
        "min_station_spacing_m",
        "max_station_spacing_m",
        "served_share",
        "avg_geology_factor",
        "geology_excess_km",
        "high_geology_km",
        "anchor_geology_penalty",
        "station_geology_penalty",
        "forbidden_km",
        "forbidden_excess_km",
        "forbidden_excess_penalty",
        "trimmed_line_tail_km",
        "water_crossing_km",
        "outside_city_km",
        "outside_city_penalty",
        "outside_city_excess_km",
        "outside_city_excess_penalty",
        "transfer_score",
        "direction_priority_bonus",
        "line_overlap_km",
        "line_overlap_excess_km",
        "line_overlap_excess_penalty",
        "arm_reuse_penalty",
        "turn_penalty",
        "max_turn_angle_deg",
        "mean_turn_angle_deg",
        "curve_radius_violation_m",
        "corridor_detour_ratio",
        "corridor_backtrack_km",
        "corridor_shape_penalty",
        "base_cost_mln",
        "flood_extra_cost_mln",
        "estimated_cost_mln",
    ]:
        table[col] = table[col].astype(float)
    return table


def build_scenarios(
    demand: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    base_config: MetroConfig = MetroConfig(),
) -> dict[str, dict]:
    return {
        "1": plan_network(demand, centre, forbidden, geology, config=base_config, n_lines=1, name="1 linia"),
        "2": plan_network(demand, centre, forbidden, geology, config=base_config, n_lines=2, name="2 linie"),
        "3": plan_network(demand, centre, forbidden, geology, config=base_config, n_lines=3, name="3 linie"),
    }


def build_orienteering_scenarios(
    demand: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    base_config: MetroConfig = MetroConfig(),
) -> dict[str, dict]:
    full_network = plan_orienteering_network(
        demand,
        candidates,
        centre,
        forbidden,
        geology,
        water_crossings=water_crossings,
        city_boundary=city_boundary,
        config=base_config,
        n_lines=3,
        name="3 linie - orienteering",
    )

    def scenario_prefix(line_count: int, name: str) -> dict:
        scenario = dict(full_network)
        scenario["name"] = name
        scenario["lines"] = full_network["lines"][:line_count]
        logs = full_network.get("candidates", pd.DataFrame())
        if isinstance(logs, pd.DataFrame) and "line_id" in logs.columns:
            scenario["candidates"] = logs[logs["line_id"] <= line_count].copy()
        else:
            scenario["candidates"] = logs.copy() if isinstance(logs, pd.DataFrame) else logs
        return scenario

    return {
        "1": scenario_prefix(1, "1 linia - orienteering"),
        "2": scenario_prefix(2, "2 linie - orienteering"),
        "3": scenario_prefix(3, "3 linie - orienteering"),
    }


def build_grid_station_scenarios(
    demand: gpd.GeoDataFrame,
    station_candidates: gpd.GeoDataFrame,
    centre: Point,
    forbidden: gpd.GeoDataFrame | None,
    geology: gpd.GeoDataFrame | None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    base_config: MetroConfig = MetroConfig(),
) -> dict[str, dict]:
    full_network = plan_grid_station_network(
        demand,
        station_candidates,
        centre,
        forbidden,
        geology,
        water_crossings=water_crossings,
        city_boundary=city_boundary,
        config=base_config,
        n_lines=3,
        name="3 linie - grid station orienteering",
    )

    def scenario_prefix(line_count: int, name: str) -> dict:
        scenario = dict(full_network)
        scenario["name"] = name
        scenario["lines"] = full_network["lines"][:line_count]
        logs = full_network.get("candidates", pd.DataFrame())
        if isinstance(logs, pd.DataFrame) and "line_id" in logs.columns:
            scenario["candidates"] = logs[logs["line_id"] <= line_count].copy()
        else:
            scenario["candidates"] = logs.copy() if isinstance(logs, pd.DataFrame) else logs
        return scenario

    return {
        "1": scenario_prefix(1, "1 linia - grid station"),
        "2": scenario_prefix(2, "2 linie - grid station"),
        "3": scenario_prefix(3, "3 linie - grid station"),
    }


def add_satellite_basemap(ax, zoom: int | str = 12) -> bool:
    """Add an Esri satellite basemap if contextily and network tiles are available."""

    try:
        import contextily as cx

        cx.add_basemap(
            ax,
            source=cx.providers.Esri.WorldImagery,
            zoom=zoom,
            attribution_size=6,
        )
        return True
    except Exception as exc:
        ax.text(
            0.01,
            0.01,
            f"Satellite basemap unavailable: {type(exc).__name__}",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 3},
        )
        return False


def set_padded_extent(ax, gdf: gpd.GeoDataFrame, padding_ratio: float = 0.04) -> None:
    if gdf is None or gdf.empty:
        return
    minx, miny, maxx, maxy = gdf.total_bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    ax.set_xlim(minx - width * padding_ratio, maxx + width * padding_ratio)
    ax.set_ylim(miny - height * padding_ratio, maxy + height * padding_ratio)


def plot_data_layer(
    layer: gpd.GeoDataFrame | None,
    title: str,
    column: str | None = None,
    extent_layer: gpd.GeoDataFrame | None = None,
    satellite: bool = True,
    figsize: tuple[int, int] = (9, 8),
    cmap: str = "Greys",
    alpha: float = 0.55,
    color: str = "#d62828",
    edgecolor: str | None = "#ffffff",
    linewidth: float = 0.5,
    markersize: float = 26.0,
    legend: bool = True,
):
    """Plot one input layer for presentation/debugging."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    target_crs = WEB_MERCATOR if satellite else (extent_layer.crs if extent_layer is not None and extent_layer.crs else LOCAL_CRS)

    extent = _align_crs(extent_layer, target_crs) if extent_layer is not None and not extent_layer.empty else None
    if extent is not None and not extent.empty:
        set_padded_extent(ax, extent)

    if satellite:
        add_satellite_basemap(ax)

    if layer is None or layer.empty:
        ax.text(0.5, 0.5, "Brak warstwy", ha="center", va="center", transform=ax.transAxes)
    else:
        layer_map = _align_crs(layer, target_crs)
        has_area = layer_map.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any()
        has_line = layer_map.geometry.geom_type.isin(["LineString", "MultiLineString"]).any()
        if column and column in layer_map.columns:
            layer_map.plot(
                ax=ax,
                column=column,
                cmap=cmap,
                alpha=alpha,
                edgecolor=edgecolor if has_area else color,
                linewidth=linewidth,
                markersize=markersize,
                legend=legend,
                zorder=3,
            )
        elif has_line:
            layer_map.plot(ax=ax, color=color, alpha=alpha, linewidth=linewidth, zorder=3)
        else:
            layer_map.plot(
                ax=ax,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                markersize=markersize,
                zorder=3,
            )

    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def plot_demand_relocation_map(
    demand_raw: gpd.GeoDataFrame,
    demand: gpd.GeoDataFrame,
    demand_areas: gpd.GeoDataFrame | None = None,
    forbidden: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    centre_area: gpd.GeoDataFrame | None = None,
    relocation_areas: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    centres: gpd.GeoDataFrame | None = None,
    weight_col: str = "population",
    satellite: bool = True,
    figsize: tuple[int, int] = (10, 10),
):
    """Diagnostic map for demand polygons, flood zones, river, and relocation vectors."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    target_crs = WEB_MERCATOR if satellite else demand.crs
    fig, ax = plt.subplots(figsize=figsize)

    extent_source = None
    for layer in (demand_areas, city_boundary, demand_raw, demand):
        if isinstance(layer, gpd.GeoDataFrame) and not layer.empty:
            extent_source = layer
            break
    if extent_source is not None:
        set_padded_extent(ax, _align_crs(extent_source, target_crs), padding_ratio=0.035)

    if satellite:
        add_satellite_basemap(ax)

    if demand_areas is not None and not demand_areas.empty:
        areas = _align_crs(demand_areas, target_crs)
        column = "population_density" if "population_density" in areas.columns else weight_col
        areas.plot(
            ax=ax,
            column=column,
            cmap="YlGnBu",
            alpha=0.50,
            edgecolor="none",
            legend=False,
            zorder=2,
        )

    if relocation_areas is not None and not relocation_areas.empty:
        areas = _align_crs(relocation_areas, target_crs)
        areas.plot(
            ax=ax,
            facecolor="#00c2a8",
            edgecolor="#006d77",
            alpha=0.13,
            linewidth=1.0,
            zorder=4,
        )

    if water_crossings is not None and not water_crossings.empty:
        _align_crs(water_crossings, target_crs).plot(
            ax=ax,
            color="#58c4ff",
            edgecolor="#156b8a",
            alpha=0.46,
            linewidth=0.45,
            zorder=5,
        )

    if forbidden is not None and not forbidden.empty:
        _align_crs(forbidden, target_crs).plot(
            ax=ax,
            facecolor="#ff3b30",
            edgecolor="#7a1111",
            alpha=0.28,
            linewidth=0.65,
            zorder=6,
        )

    if city_boundary is not None and not city_boundary.empty:
        _align_crs(city_boundary, target_crs).boundary.plot(
            ax=ax,
            color="#ffffff" if satellite else "#111111",
            alpha=0.95,
            linewidth=1.4,
            linestyle="--",
            zorder=7,
        )

    if centre_area is not None and not centre_area.empty:
        centre_map = _align_crs(centre_area, target_crs)
        centre_map.plot(
            ax=ax,
            facecolor="#ffd60a",
            edgecolor="#111111",
            alpha=0.20,
            linewidth=4.0,
            zorder=8,
        )
        centre_map.boundary.plot(
            ax=ax,
            color="#ffd60a",
            linewidth=2.0,
            zorder=9,
        )

    vectors = demand_relocation_vectors(demand)
    if not vectors.empty:
        vectors_map = _align_crs(vectors, target_crs)
        for _, row in vectors_map.iterrows():
            line_width = float(np.clip(row.get("relocated_m", 0.0) / 220.0, 0.7, 3.2))
            gpd.GeoSeries([row.geometry], crs=target_crs).plot(
                ax=ax,
                color="#ff5400",
                linewidth=line_width,
                alpha=0.78,
                zorder=10,
            )

    if demand_raw is not None and not demand_raw.empty:
        raw_map = _align_crs(demand_raw, target_crs)
        raw_map.plot(
            ax=ax,
            color="#3b3b3b",
            markersize=9,
            alpha=0.28,
            zorder=11,
        )

    demand_map = _align_crs(demand, target_crs)
    relocated_mask = (
        demand_map["was_relocated"].astype(bool)
        if "was_relocated" in demand_map.columns
        else pd.Series(False, index=demand_map.index)
    )
    unchanged = demand_map[~relocated_mask]
    relocated = demand_map[relocated_mask]

    if not unchanged.empty:
        unchanged.plot(
            ax=ax,
            color="#101010",
            markersize=7,
            alpha=0.25,
            zorder=12,
        )
    if not relocated.empty:
        sizes = np.sqrt(pd.to_numeric(relocated[weight_col], errors="coerce").fillna(0.0)).clip(8, 55)
        relocated.plot(
            ax=ax,
            column="relocated_m",
            cmap="magma",
            markersize=sizes,
            edgecolor="#ffffff",
            linewidth=0.55,
            alpha=0.92,
            legend=False,
            zorder=13,
        )

    if centres is not None and not centres.empty:
        centres_map = _align_crs(centres, target_crs)
        centres_map.plot(
            ax=ax,
            marker="*",
            color="#111111",
            edgecolor="#111111",
            markersize=185,
            linewidth=1.0,
            zorder=14,
        )
        centres_map.plot(
            ax=ax,
            marker="*",
            color="#ffd60a",
            edgecolor="#111111",
            markersize=120,
            linewidth=0.6,
            zorder=15,
        )

    offset_values = pd.to_numeric(demand.get("relocated_m", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    max_offset = float(offset_values.max())
    relocated_offsets = offset_values[offset_values > 0]
    median_offset = float(relocated_offsets.median()) if not relocated_offsets.empty else 0.0
    relocated_count = int(relocated_mask.sum())
    ax.set_title("Relokacja popytu: MZP, rzeka, Stare Miasto i obszary docelowe", fontsize=14, pad=10)
    ax.text(
        0.01,
        0.99,
        f"Przesunięte punkty: {relocated_count} | mediana: {median_offset:.0f} m | max: {max_offset:.0f} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
        zorder=30,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

    handles = [
        Patch(facecolor="#225ea8", edgecolor="none", alpha=0.45, label="gęstość popytu (ciemniej = więcej)"),
        Patch(facecolor="#00c2a8", edgecolor="#006d77", alpha=0.22, label="większe obszary relokacji"),
        Patch(facecolor="#ff3b30", edgecolor="#7a1111", alpha=0.30, label="MZP / obszary zalewowe"),
        Patch(facecolor="#ffd60a", edgecolor="#111111", alpha=0.25, label="Stare Miasto"),
        Line2D([0], [0], color="#58c4ff", linewidth=2.0, label="rzeka / wody powierzchniowe"),
        Line2D([0], [0], color="#ff5400", linewidth=2.0, label="wektor przesunięcia wagi"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3b3b3b", markersize=5, label="punkt popytu przed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d53e4f", markeredgecolor="#ffffff", markersize=7, label="punkt popytu po relokacji"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.86)
    return fig, ax


def plot_scenario_satellite(
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    scenario: dict,
    centres: gpd.GeoDataFrame | None = None,
    regional_centres: gpd.GeoDataFrame | None = None,
    regional_clusters: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    demand_areas: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    centre_area: gpd.GeoDataFrame | None = None,
    weight_col: str = "population",
    figsize: tuple[int, int] = (10, 10),
):
    """Presentation map: satellite basemap, translucent density, opaque metro lines."""

    import matplotlib.pyplot as plt

    target_crs = WEB_MERCATOR
    demand_map = _align_crs(demand, target_crs)
    lines, stations = scenario_geodataframes(scenario)
    lines = _align_crs(lines, target_crs)
    stations = _align_crs(stations, target_crs)
    fig, ax = plt.subplots(figsize=figsize)

    extent = _align_crs(demand_areas, target_crs) if demand_areas is not None and not demand_areas.empty else demand_map
    set_padded_extent(ax, extent)
    add_satellite_basemap(ax)

    if city_boundary is not None and not city_boundary.empty:
        _align_crs(city_boundary, target_crs).boundary.plot(
            ax=ax,
            color="#ffffff",
            alpha=0.95,
            linewidth=2.0,
            zorder=7,
        )
        _align_crs(city_boundary, target_crs).boundary.plot(
            ax=ax,
            color="#111111",
            alpha=0.95,
            linewidth=0.9,
            zorder=8,
        )

    if demand_areas is not None and not demand_areas.empty:
        areas = _align_crs(demand_areas, target_crs)
        column = "population_density" if "population_density" in areas.columns else weight_col
        areas.plot(
            ax=ax,
            column=column,
            cmap="YlGn",
            alpha=0.46,
            edgecolor="none",
            legend=True,
            zorder=2,
        )

    if water_crossings is not None and not water_crossings.empty:
        _align_crs(water_crossings, target_crs).plot(
            ax=ax,
            color="#6ec6ff",
            alpha=0.36,
            edgecolor="#1f78a8",
            linewidth=0.35,
            zorder=3,
        )

    if regional_clusters is not None and not regional_clusters.empty:
        clusters = _align_crs(regional_clusters, target_crs)
        clusters.boundary.plot(ax=ax, color="#f72585", alpha=0.85, linewidth=1.5, zorder=6)
        clusters.plot(
            ax=ax,
            facecolor="#f72585",
            alpha=0.12,
            edgecolor="none",
            zorder=5,
        )

    if forbidden is not None and not forbidden.empty:
        _align_crs(forbidden, target_crs).plot(
            ax=ax,
            facecolor="#ff3b30",
            edgecolor="#7a1111",
            alpha=0.28,
            linewidth=0.7,
            zorder=4,
        )

    if geology is not None and not geology.empty:
        _align_crs(geology, target_crs).boundary.plot(
            ax=ax,
            color="#f5c542",
            alpha=0.45,
            linewidth=0.8,
            zorder=5,
        )

    if centre_area is not None and not centre_area.empty:
        _align_crs(centre_area, target_crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="#111111",
            linewidth=4.4,
            linestyle="-",
            zorder=8,
        )
        _align_crs(centre_area, target_crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="#ffd60a",
            linewidth=2.4,
            linestyle="-",
            zorder=9,
        )

    if regional_centres is not None and not regional_centres.empty:
        _align_crs(regional_centres, target_crs).plot(
            ax=ax,
            marker="x",
            color="#ffffff",
            markersize=110,
            linewidth=3.0,
            zorder=10,
        )
        _align_crs(regional_centres, target_crs).plot(
            ax=ax,
            marker="x",
            color="#7a1f5c",
            markersize=78,
            linewidth=2.0,
            zorder=11,
        )

    colours = ["#ff2d2d", "#00c2a8", "#4d7cff", "#ffb000"]
    for idx, line_id in enumerate(sorted(lines["line_id"].unique())):
        line_part = lines[lines["line_id"] == line_id]
        station_part = stations[stations["line_id"] == line_id]
        colour = colours[idx % len(colours)]
        line_part.plot(ax=ax, color="#111111", linewidth=7.0, alpha=1.0, zorder=20)
        line_part.plot(ax=ax, color=colour, linewidth=4.2, alpha=1.0, zorder=21)
        station_part.plot(ax=ax, color="#ffffff", edgecolor="#111111", markersize=52, linewidth=1.4, zorder=22)
        station_part.plot(ax=ax, color=colour, edgecolor="#ffffff", markersize=24, linewidth=0.7, zorder=23)

    if centres is not None and not centres.empty:
        centres_map = _align_crs(centres, target_crs)
        centres_map.plot(ax=ax, markersize=220, marker="*", color="#111111", edgecolor="#111111", linewidth=1.2, zorder=24)
        centres_map.plot(ax=ax, markersize=145, marker="*", color="#ffd60a", edgecolor="#111111", linewidth=0.8, zorder=25)

    ax.set_title(f"{scenario['name']} - mapa satelitarna")
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def plot_scenario(
    demand: gpd.GeoDataFrame,
    forbidden: gpd.GeoDataFrame | None,
    scenario: dict,
    centres: gpd.GeoDataFrame | None = None,
    regional_centres: gpd.GeoDataFrame | None = None,
    regional_clusters: gpd.GeoDataFrame | None = None,
    geology: gpd.GeoDataFrame | None = None,
    demand_areas: gpd.GeoDataFrame | None = None,
    water_crossings: gpd.GeoDataFrame | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
    centre_area: gpd.GeoDataFrame | None = None,
    weight_col: str = "population",
    figsize: tuple[int, int] = (10, 10),
):
    """Matplotlib map for notebook iteration."""

    import matplotlib.pyplot as plt

    demand = demand.copy()
    lines, stations = scenario_geodataframes(scenario)
    fig, ax = plt.subplots(figsize=figsize)

    if demand_areas is not None and not demand_areas.empty:
        areas = _align_crs(demand_areas, demand.crs)
        column = "population_density" if "population_density" in areas.columns else weight_col
        areas.plot(
            ax=ax,
            column=column,
            cmap="YlGn",
            alpha=0.54,
            edgecolor="#ffffff",
            linewidth=0.12,
            legend=True,
        )

    if geology is not None and not geology.empty:
        _align_crs(geology, demand.crs).plot(
            ax=ax,
            column="cost_factor" if "cost_factor" in geology.columns else None,
            cmap="YlOrBr",
            alpha=0.18,
            edgecolor="none",
            legend=True,
        )

    if water_crossings is not None and not water_crossings.empty:
        _align_crs(water_crossings, demand.crs).plot(
            ax=ax,
            color="#74b6d7",
            alpha=0.42,
            edgecolor="#2f6f95",
            linewidth=0.45,
        )

    if city_boundary is not None and not city_boundary.empty:
        _align_crs(city_boundary, demand.crs).boundary.plot(
            ax=ax,
            color="#111111",
            alpha=0.88,
            linewidth=1.5,
            linestyle="--",
        )

    if forbidden is not None and not forbidden.empty:
        _align_crs(forbidden, demand.crs).plot(ax=ax, color="#ff3b30", alpha=0.32, edgecolor="#7a1111", linewidth=0.8)

    if regional_clusters is not None and not regional_clusters.empty:
        clusters = _align_crs(regional_clusters, demand.crs)
        clusters.boundary.plot(ax=ax, color="#7a1f5c", alpha=0.85, linewidth=1.2)

    if centre_area is not None and not centre_area.empty:
        _align_crs(centre_area, demand.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="#111111",
            linewidth=3.8,
            linestyle="-",
        )
        _align_crs(centre_area, demand.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor="#ffd60a",
            linewidth=2.0,
            linestyle="-",
        )

    sizes = np.sqrt(pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0)) * 1.2
    if demand_areas is None or demand_areas.empty:
        demand.plot(ax=ax, markersize=sizes, color="#363636", alpha=0.55)
    else:
        demand.plot(ax=ax, markersize=np.minimum(sizes, 16), color="#1f1f1f", alpha=0.20)

    if regional_centres is not None and not regional_centres.empty:
        _align_crs(regional_centres, demand.crs).plot(
            ax=ax,
            markersize=90,
            marker="x",
            color="#7a1f5c",
            linewidth=2.0,
        )

    colours = ["#d62828", "#2a9d8f", "#4361ee", "#f77f00"]
    for idx, line_id in enumerate(sorted(lines["line_id"].unique())):
        line_part = lines[lines["line_id"] == line_id]
        station_part = stations[stations["line_id"] == line_id]
        colour = colours[idx % len(colours)]
        line_part.plot(ax=ax, color=colour, linewidth=3.0)
        station_part.plot(ax=ax, color="white", edgecolor=colour, markersize=44, linewidth=1.4)

    if centres is not None and not centres.empty:
        centres_map = _align_crs(centres, demand.crs)
        centres_map.plot(ax=ax, markersize=160, marker="*", color="#111111", edgecolor="#111111", linewidth=1.0)
        centres_map.plot(ax=ax, markersize=105, marker="*", color="#ffd60a", edgecolor="#111111", linewidth=0.6)

    ax.set_title(scenario["name"])
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax

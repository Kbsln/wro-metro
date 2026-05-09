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
from math import cos, pi, radians, sin
from pathlib import Path
from typing import Iterable, Mapping
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union

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
    outside_city_penalty_per_km: float = 120_000.0
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
    line_overlap_penalty_per_km: float = 220_000.0
    parallel_line_buffer_m: float = 900.0
    residual_coverage_multiplier: float = 1.35
    relocation_search_radius_m: float = 3_000.0
    relocation_step_m: float = 100.0
    station_risk_buffer_m: float = 260.0
    candidate_catchment_radius_m: float = 1_600.0
    central_anchor_radius_m: float = 3_000.0
    min_central_anchor_candidates: int = 8
    max_central_anchor_share: float = 0.32
    min_regional_anchor_candidates: int = 8
    min_directional_anchor_candidates_per_sector: int = 5
    regional_anchor_weight_floor_fraction: float = 0.08
    route_anchor_count: int = 8
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


def candidate_catchment_weights(
    candidate_points: list[Point],
    demand: gpd.GeoDataFrame,
    radius_m: float,
    weight_col: str = "population",
) -> np.ndarray:
    """Estimate station-anchor value from nearby demand, not only its own polygon."""

    if not candidate_points or demand.empty or radius_m <= 0:
        return np.zeros(len(candidate_points), dtype=float)

    candidate_xy = np.array([(point.x, point.y) for point in candidate_points], dtype=float)
    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    demand_weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    distances = np.sqrt(((candidate_xy[:, None, :] - demand_xy[None, :, :]) ** 2).sum(axis=2))
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
    Flood-risk areas are used to move anchor candidates away from risky polygons
    and a small buffer around them, while preserving their demand weight. Geology
    reduces the score of anchors placed on difficult or high-risk ground.
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
        if forbidden_geom is not None and not forbidden_geom.is_empty:
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
            new_point, offset = nearest_free_point_from_boundary(
                point,
                avoidance_geom,
                step_m=config.relocation_step_m,
                max_extra_m=config.relocation_search_radius_m,
            )
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
        candidates["direction_sector"] = np.where(
            np.abs(dy) >= np.abs(dx),
            np.where(dy >= 0.0, "north", "south"),
            np.where(dx >= 0.0, "east", "west"),
        )
    else:
        candidates["distance_to_required_centre_m"] = np.nan
        candidates["near_required_centre"] = False
        candidates["direction_sector"] = "unknown"

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


def extend_route_points_to_length(points: list[Point], target_length_m: float) -> list[Point]:
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
        return [Point(centre.x - half, centre.y), Point(centre.x + half, centre.y)]

    extended = list(points)
    first, second = extended[0], extended[1]
    last, before_last = extended[-1], extended[-2]

    def extend_endpoint(endpoint: Point, neighbour: Point, extra_m: float) -> Point:
        vector = np.array([endpoint.x - neighbour.x, endpoint.y - neighbour.y], dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return endpoint
        unit = vector / norm
        return Point(endpoint.x + unit[0] * extra_m, endpoint.y + unit[1] * extra_m)

    extended[0] = extend_endpoint(first, second, deficit / 2.0)
    extended[-1] = extend_endpoint(last, before_last, deficit / 2.0)
    return extended


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


def line_overlap_km(
    points: list[Point],
    existing_lines: gpd.GeoDataFrame | None,
    config: MetroConfig,
) -> float:
    """Length of a route that runs inside an existing metro corridor buffer."""

    if existing_lines is None or existing_lines.empty or len(points) < 2:
        return 0.0
    corridor = route_polyline(points)
    existing_union = unary_union([geom for geom in existing_lines.geometry if geom is not None and not geom.is_empty])
    if existing_union.is_empty:
        return 0.0
    overlap_zone = existing_union.buffer(config.parallel_line_buffer_m)
    return float(corridor.intersection(overlap_zone).length / 1_000.0)


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
            "geology_factor": 1.0,
            "geology_excess_km": 0.0,
            "high_geology_km": 0.0,
            "station_geology_penalty": 0.0,
            "water_crossing_km": 0.0,
            "transfer_score": 0.0,
            "transfer_count": 0,
            "line_overlap_km": 0.0,
            "outside_city_km": 0.0,
            "outside_city_penalty": 0.0,
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
    turn_metrics = route_turn_metrics(points, config)
    shape_metrics = corridor_shape_metrics(points, config)
    score = served_weight
    score -= forbidden_km * config.forbidden_penalty_per_km
    score -= outside_city_penalty
    score -= geology_excess_km * config.geology_penalty_per_km
    score -= high_geology_km * config.high_geology_penalty_per_km
    score -= station_geology_penalty
    score -= overlap_km * config.line_overlap_penalty_per_km
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
        "geology_factor": geology_factor,
        "geology_excess_km": geology_excess_km,
        "high_geology_km": high_geology_km,
        "station_geology_penalty": station_geology_penalty,
        "water_crossing_km": water_km,
        "transfer_score": transfer_score,
        "transfer_count": transfer_count,
        "line_overlap_km": overlap_km,
        "outside_city_km": outside_city_km,
        "outside_city_penalty": outside_city_penalty,
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

    if forbidden_geom is not None and not forbidden_geom.is_empty:
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
    repaired_distances, repaired = repair_station_distances_along_line(
        line,
        chosen_distances,
        distances,
        forbidden_geom,
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

    def fast_metrics(order: list[int]) -> dict:
        return _fast_order_metrics(order, matrices, config)

    target_anchor_count = max(1, min(int(config.route_anchor_count), len(nodes), config.station_count))

    selected_order = [0]
    selected_nodes = {0}
    log_rows = []
    current_metrics = fast_metrics(selected_order)

    while len(selected_order) < target_anchor_count:
        best_move = None
        for node_index in range(1, len(nodes)):
            if node_index in selected_nodes:
                continue
            if any(distance_matrix[node_index, selected_index] < min_anchor_spacing_m for selected_index in selected_order):
                continue

            for position in range(len(selected_order) + 1):
                proposal_order = selected_order[:position] + [node_index] + selected_order[position:]
                proposal_length = fast_metrics(proposal_order)["route_length_m"]
                if proposal_length > config.length_m:
                    continue
                metrics = fast_metrics(proposal_order)
                if (
                    config.hard_max_turn_angle_deg > 0.0
                    and metrics["max_turn_angle_deg"] > config.hard_max_turn_angle_deg
                ):
                    continue
                improvement = metrics["score"] - current_metrics["score"]
                added_length = max(1.0, proposal_length - current_metrics["route_length_m"])
                ratio = improvement / added_length
                ranking_key = (ratio, improvement, metrics["served_weight"])
                if best_move is None or ranking_key > best_move["ranking_key"]:
                    best_move = {
                        "candidate_id": nodes[node_index]["candidate_id"],
                        "node_index": node_index,
                        "position": position,
                        "proposal_order": proposal_order,
                        "metrics": metrics,
                        "improvement": improvement,
                        "added_length_m": added_length,
                        "ranking_key": ranking_key,
                    }

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
                **best_move["metrics"],
            }
        )

    improved = True
    iteration = 0
    while improved and iteration < 60 and len(selected_order) >= 4:
        improved = False
        iteration += 1
        best_score = fast_metrics(selected_order)["score"]
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
                ):
                    selected_order = proposal_order
                    improved = True
                    break
            if improved:
                break

    anchor_metrics = fast_metrics(selected_order)
    selected = [nodes[index] for index in selected_order]

    anchor_points = [item["geometry"] for item in selected]
    corridor_points = extend_route_points_to_length(anchor_points, config.length_m)
    corridor = route_polyline(corridor_points)
    corridor_turn_metrics = route_turn_metrics(corridor_points, config)
    corridor_shape = corridor_shape_metrics(corridor_points, config)
    outside_city_km = line_outside_city_km(corridor, city_boundary)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km

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
    final_metrics["score"] -= outside_city_penalty
    final_metrics["outside_city_km"] = outside_city_km
    final_metrics["outside_city_penalty"] = outside_city_penalty
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
            "transfer_score": [final_metrics["transfer_score"]],
            "transfer_count": [final_metrics["transfer_count"]],
            "line_overlap_km": [final_metrics["line_overlap_km"]],
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
    overlap_lines = existing_lines if existing_lines is not None and not existing_lines.empty else transfer_lines
    if overlap_lines is not None and not overlap_lines.empty:
        overlap_union = unary_union([geom for geom in overlap_lines.geometry if geom is not None and not geom.is_empty])
        if not overlap_union.is_empty:
            overlap_zone = overlap_union.buffer(config.parallel_line_buffer_m)
            for left in range(len(nodes)):
                for right in range(left + 1, len(nodes)):
                    segment = LineString([tuple(node_xy[left]), tuple(node_xy[right])])
                    value = float(segment.intersection(overlap_zone).length / 1_000.0)
                    overlap_matrix[left, right] = value
                    overlap_matrix[right, left] = value

    transfer_point_coverage = np.zeros(len(nodes), dtype=float)
    if transfer_points is not None and not transfer_points.empty:
        transfer_xy = np.array([(point.x, point.y) for point in transfer_points.geometry], dtype=float)
        if len(transfer_xy):
            distances = np.sqrt(((node_xy[:, None, :] - transfer_xy[None, :, :]) ** 2).sum(axis=2))
            transfer_point_coverage = (distances.min(axis=1) <= config.interchange_radius_m).astype(float)

    demand_xy = np.column_stack([demand.geometry.x.to_numpy(), demand.geometry.y.to_numpy()])
    demand_weights = pd.to_numeric(demand[weight_col], errors="coerce").fillna(0.0).to_numpy()
    node_to_demand = np.sqrt(((node_xy[:, None, :] - demand_xy[None, :, :]) ** 2).sum(axis=2))
    coverage_matrix = np.clip(1.0 - node_to_demand / config.walk_radius_m, 0.0, 1.0)

    return {
        "node_xy": node_xy,
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
        "overlap_matrix": overlap_matrix,
        "coverage_matrix": coverage_matrix,
        "demand_weights": demand_weights,
        "total_weight": float(demand_weights.sum()),
    }


def _fast_order_metrics(order: list[int], matrices: Mapping, config: MetroConfig) -> dict:
    if not order:
        return {
            "score": 0.0,
            "served_weight": 0.0,
            "served_share": 0.0,
            "route_length_m": 0.0,
            "forbidden_km": 0.0,
            "geology_factor": 1.0,
            "geology_excess_km": 0.0,
            "high_geology_km": 0.0,
            "anchor_geology_penalty": 0.0,
            "outside_city_km": 0.0,
            "outside_city_penalty": 0.0,
            **_empty_turn_metrics(),
            **_empty_corridor_shape_metrics(),
        }

    node_xy = matrices["node_xy"]
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
    overlap_matrix = matrices.get("overlap_matrix", np.zeros_like(distance_matrix))
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
    if city_boundary_geom is not None and not city_boundary_geom.is_empty:
        full_corridor_points = [Point(float(x), float(y)) for x, y in node_xy[order_array]]
        full_corridor_points = extend_route_points_to_length(full_corridor_points, config.length_m)
        outside_city_km = line_outside_city_km(route_polyline(full_corridor_points), city_boundary_geom)

    coverage = coverage_matrix[order_array].max(axis=0)
    served_weight = float(np.sum(demand_weights * coverage))
    anchor_geology_penalty = float(node_geology_penalty[order_array].sum())
    transfer_point_count = int(min(2, transfer_point_coverage[order_array].sum()))
    transfer_count = int(min(3, transfer_point_count + transfer_segment_count))
    transfer_score = float(transfer_count * config.transfer_bonus_per_interchange)
    route_km = length_m / 1_000.0
    geology_factor = 1.0 + geology_excess_km / route_km if route_km else 1.0
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km
    turn_metrics = _turn_metrics_from_xy(node_xy[order_array], config)
    shape_metrics = _corridor_shape_metrics_from_xy(node_xy[order_array], config)
    score = served_weight - forbidden_km * config.forbidden_penalty_per_km
    score -= geology_excess_km * config.geology_penalty_per_km
    score -= high_geology_km * config.high_geology_penalty_per_km
    score -= anchor_geology_penalty
    score -= outside_city_penalty
    score -= overlap_km * config.line_overlap_penalty_per_km
    score -= turn_metrics["turn_penalty"]
    score -= shape_metrics["corridor_shape_penalty"]
    score += water_km * config.river_crossing_bonus_per_km
    score += transfer_score
    return {
        "score": score,
        "served_weight": served_weight,
        "served_share": served_weight / total_weight if total_weight else 0.0,
        "route_length_m": length_m,
        "forbidden_km": forbidden_km,
        "geology_factor": geology_factor,
        "geology_excess_km": geology_excess_km,
        "high_geology_km": high_geology_km,
        "anchor_geology_penalty": anchor_geology_penalty,
        "water_crossing_km": water_km,
        "transfer_score": transfer_score,
        "transfer_count": transfer_count,
        "line_overlap_km": overlap_km,
        "outside_city_km": outside_city_km,
        "outside_city_penalty": outside_city_penalty,
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
    corridor_points = extend_route_points_to_length(anchor_points, config.length_m)
    corridor = route_polyline(corridor_points)

    corridor_turn_metrics = route_turn_metrics(corridor_points, config)
    corridor_shape = corridor_shape_metrics(corridor_points, config)
    outside_city_km = line_outside_city_km(corridor, city_boundary)
    outside_city_penalty = outside_city_km * config.outside_city_penalty_per_km

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
    final_metrics["outside_city_km"] = outside_city_km
    final_metrics["outside_city_penalty"] = outside_city_penalty
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
            "transfer_score": [final_metrics["transfer_score"]],
            "transfer_count": [final_metrics["transfer_count"]],
            "line_overlap_km": [final_metrics["line_overlap_km"]],
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
    outside_city = float(lines["outside_city_km"].sum()) if "outside_city_km" in lines else 0.0
    outside_city_penalty = float(lines["outside_city_penalty"].sum()) if "outside_city_penalty" in lines else 0.0
    transfer_score = float(lines["transfer_score"].sum()) if "transfer_score" in lines else 0.0
    transfer_count = int(lines["transfer_count"].sum()) if "transfer_count" in lines else 0
    line_overlap = float(lines["line_overlap_km"].sum()) if "line_overlap_km" in lines else 0.0
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
        "station_flood_zone_count": station_flood_zone_count,
        "station_water_zone_count": station_water_zone_count,
        "station_flood_buffer_count": station_flood_buffer_count,
        "station_water_buffer_count": station_water_buffer_count,
        "water_crossing_km": water_crossing,
        "outside_city_km": outside_city,
        "outside_city_penalty": outside_city_penalty,
        "transfer_count": transfer_count,
        "transfer_score": transfer_score,
        "line_overlap_km": line_overlap,
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
        "water_crossing_km",
        "outside_city_km",
        "outside_city_penalty",
        "transfer_score",
        "line_overlap_km",
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
    return {
        "1": plan_orienteering_network(
            demand,
            candidates,
            centre,
            forbidden,
            geology,
            water_crossings=water_crossings,
            city_boundary=city_boundary,
            config=base_config,
            n_lines=1,
            name="1 linia - orienteering",
        ),
        "2": plan_orienteering_network(
            demand,
            candidates,
            centre,
            forbidden,
            geology,
            water_crossings=water_crossings,
            city_boundary=city_boundary,
            config=base_config,
            n_lines=2,
            name="2 linie - orienteering",
        ),
        "3": plan_orienteering_network(
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
        ),
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

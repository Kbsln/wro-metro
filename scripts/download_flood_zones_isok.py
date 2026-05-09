"""Download Wroclaw flood hazard polygons from the official ISOK/Wody Polskie WFS.

The WMS layers shown by the QGIS plugin are useful for previewing the map, but
they are rasters/tiles. This script uses the WFS download service, clips the
returned hazard polygons to Wroclaw, and writes a local GeoJSON file consumed by
the notebook.
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wro_metro.planning import load_wroclaw_osiedla  # noqa: E402

WFS_URL = "https://wody.isok.gov.pl/wss/INSPIRE/INSPIRE_NZ_HY_MZPMRP_WFS"
TYPE_NAME = "nz-core:HazardArea"


def wfs_bbox_for_epsg_4326(bounds: tuple[float, float, float, float], padding_deg: float) -> str:
    """Return WFS 1.1 bbox string for EPSG:4326.

    WFS 1.1 follows the official EPSG:4326 axis order, so the bbox is sent as
    min_lat,min_lon,max_lat,max_lon. Most desktop GIS UIs display the same CRS
    as lon/lat, which is why this looks reversed.
    """

    min_lon, min_lat, max_lon, max_lat = bounds
    return (
        f"{min_lat - padding_deg},{min_lon - padding_deg},"
        f"{max_lat + padding_deg},{max_lon + padding_deg},urn:ogc:def:crs:EPSG::4326"
    )


def wfs_page_url(bbox: str, start_index: int, page_size: int) -> str:
    params = {
        "SERVICE": "WFS",
        "VERSION": "1.1.0",
        "REQUEST": "GetFeature",
        "TYPENAME": TYPE_NAME,
        "SRSNAME": "EPSG:4326",
        "BBOX": bbox,
        "MAXFEATURES": str(page_size),
        "STARTINDEX": str(start_index),
    }
    return f"{WFS_URL}?{urlencode(params)}"


def read_wfs_page(url: str) -> gpd.GeoDataFrame:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    text_start = response.text[:300]
    if "ExceptionReport" in text_start or "ServiceExceptionReport" in text_start:
        raise RuntimeError(f"WFS returned an exception:\n{text_start}")

    frame = gpd.read_file(BytesIO(response.content))
    if frame.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    if "geometry" not in frame.columns:
        raise RuntimeError("WFS response has no geometry column.")

    geometry = frame["geometry"].map(wkt.loads)
    return gpd.GeoDataFrame(frame.drop(columns=["geometry"]), geometry=geometry, crs=4326)


def normalize_scenario_columns(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    zones = zones.copy()
    zones["source_layer"] = "ISOK_MZP_MRP_WFS:nz-core:HazardArea"
    zones["risk"] = "flood_zone"
    if "qualitativeLikelihood" in zones.columns:
        zones["scenario"] = zones["qualitativeLikelihood"].astype(str)
    if "probabilityOfOccurrence" in zones.columns:
        zones["probability"] = pd.to_numeric(zones["probabilityOfOccurrence"], errors="coerce")
    if "returnPeriod" in zones.columns:
        zones["return_period_years"] = pd.to_numeric(zones["returnPeriod"], errors="coerce")
    return zones


def download_flood_zones(
    data_dir: Path,
    output: Path,
    page_size: int,
    padding_deg: float,
    max_pages: int,
) -> gpd.GeoDataFrame:
    boundary = load_wroclaw_osiedla(data_dir, target_crs=4326).dissolve()
    bbox = wfs_bbox_for_epsg_4326(tuple(boundary.total_bounds), padding_deg=padding_deg)

    pages: list[gpd.GeoDataFrame] = []
    for page in range(max_pages):
        start_index = page * page_size
        print(f"Downloading WFS page startIndex={start_index} maxFeatures={page_size}...")
        gdf = read_wfs_page(wfs_page_url(bbox, start_index, page_size))
        if gdf.empty:
            break
        pages.append(gdf)
        print(f"  received {len(gdf)} features")
        if len(gdf) < page_size:
            break

    if not pages:
        raise RuntimeError("No flood hazard features returned by the WFS service.")

    raw = pd.concat(pages, ignore_index=True)
    zones = gpd.GeoDataFrame(raw, geometry="geometry", crs=4326)
    zones = zones.drop_duplicates(subset=["gml_id"]).reset_index(drop=True)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    zones = zones[zones.intersects(boundary.geometry.iloc[0])].copy()
    zones = gpd.clip(zones, boundary)
    zones = normalize_scenario_columns(zones)

    if zones.empty:
        raise RuntimeError("WFS returned features, but none intersect Wroclaw after clipping.")

    output.parent.mkdir(parents=True, exist_ok=True)
    zones.to_file(output, driver="GeoJSON")
    return zones


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "flood_zones.geojson")
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--padding-deg", type=float, default=0.02)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()

    zones = download_flood_zones(
        data_dir=args.data_dir,
        output=args.output,
        page_size=args.page_size,
        padding_deg=args.padding_deg,
        max_pages=args.max_pages,
    )
    print(f"Saved {len(zones)} clipped flood hazard polygons to {args.output}")
    if "scenario" in zones.columns:
        print(zones["scenario"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

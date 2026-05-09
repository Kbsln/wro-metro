"""Download and prepare a Wroclaw geology cost layer from PIG-PIB CBDG."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
from shapely.geometry import box

MLP50K_LAYER_URL = "https://cbdgmapa.pgi.gov.pl/arcgis/rest/services/kartografia/mlp50k/MapServer/6"
WROCLAW_BBOX_WGS84 = (16.80, 51.00, 17.22, 51.22)
SIMPLIFY_TOLERANCE_DEG = 0.00008


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _lithology_cost_factor(label: str) -> float:
    """Heuristic construction difficulty multiplier from MLP lithology labels."""

    lower = label.lower()
    if "wody powierzchniowe" in lower:
        return 1.60
    if "torf" in lower:
        return 1.75
    if "iły" in lower or "ily" in lower:
        return 1.50
    if "pyły ilaste" in lower:
        return 1.45
    if "pyły" in lower or "pyl" in lower:
        return 1.30
    if "gliny pyłowate" in lower or "gliny pylowate" in lower:
        return 1.35
    if "gliny" in lower:
        return 1.25
    if "piaski gliniaste" in lower:
        return 1.25
    if "piaski pyłowate" in lower or "piaski pylowate" in lower:
        return 1.20
    if "tło pod znakiem form antropogenicznych" in lower:
        return 1.30
    if "kreda" in lower:
        return 1.30
    if "piaski żwirowate" in lower or "piaski zwirowate" in lower:
        return 1.10
    if "piaski" in lower:
        return 1.08
    if "żwiry" in lower or "zwiry" in lower:
        return 1.06
    return 1.15


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata = _fetch_json(f"{MLP50K_LAYER_URL}?f=pjson")
    renderer = metadata["drawingInfo"]["renderer"]
    labels = {
        str(info["value"]): info.get("label", str(info["value"]))
        for info in renderer.get("uniqueValueInfos", [])
    }

    bbox = ",".join(str(value) for value in WROCLAW_BBOX_WGS84)
    params = urlencode(
        {
            "f": "geojson",
            "where": "1=1",
            "outFields": "*",
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "true",
        }
    )
    geology = _fetch_json(f"{MLP50K_LAYER_URL}/query?{params}")

    for feature in geology.get("features", []):
        properties = feature.setdefault("properties", {})
        code = str(properties.get("KOD", ""))
        lithology = labels.get(code, code)
        properties["source_code"] = code
        properties["lithology"] = lithology
        properties["cost_factor"] = _lithology_cost_factor(lithology)

    geology_gdf = gpd.GeoDataFrame.from_features(geology["features"], crs="EPSG:4326")
    geology_gdf = gpd.clip(geology_gdf, box(*WROCLAW_BBOX_WGS84))
    keep_cols = [
        "source_code",
        "lithology",
        "cost_factor",
        "geometry",
    ]
    geology_gdf = geology_gdf[keep_cols]
    geology_gdf = geology_gdf.dissolve(
        by=[
            "source_code",
            "lithology",
            "cost_factor",
        ],
        as_index=False,
    )
    geology_gdf["geometry"] = geology_gdf.geometry.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    geology_gdf = geology_gdf.explode(index_parts=False).reset_index(drop=True)
    geology_gdf = geology_gdf[~geology_gdf.geometry.is_empty & geology_gdf.geometry.notna()]

    target = raw_dir / "geology.geojson"
    geology_gdf.to_file(target, driver="GeoJSON", COORDINATE_PRECISION="6")
    print(f"saved {target.relative_to(root)} ({len(geology_gdf)} features)")


if __name__ == "__main__":
    main()

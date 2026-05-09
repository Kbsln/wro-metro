## Wroclaw Metro Planner — quick guide for AI coding agents

This repository is an exploratory, notebook-driven Python project for designing radial metro lines for Wrocław. The single package is `src/wro_metro/planning.py` and the interactive entrypoint is the notebook `notebooks/01_wroclaw_metro_planner.ipynb`.

Key facts & entrypoints
- Main library: `src/wro_metro/planning.py` (all planning helpers, heuristics, plotting).
- Notebook: `notebooks/01_wroclaw_metro_planner.ipynb` — the user-facing workflow and examples.
- Data folder: `data/raw/` — loaders expect specific filenames (see below) or optional demo functions in the module.

Important files/names
- `scripts/download_wroclaw_data.py` — convenience script to fetch SIP data.
- `requirements.txt` — Python deps (GeoPandas, Shapely, contextily optional for basemaps).
- Data filenames the code looks for in `data/raw/`: `dem-rejurb-rejstat-shp.zip`, `granice-osiedli.zip`, `wody-powierzchniowe.zip`, `flood_zones.*` (geojson/gpkg/shp/zip), optional `geology.*`.

Data shapes & CRS
- Most functions accept and return GeoPandas GeoDataFrame objects.
- Spatial operations use LOCAL_CRS = 2177 (defined in the module). Loaders and helpers call `_align_crs()` to convert inputs — prefer feeding WGS84 or let loaders convert.
- Demand must contain a numeric population weight column (the code will infer a suitable column via `guess_weight_column`). Many helpers assume a `population` or equivalent numeric column.

Core concepts & APIs to reuse
- MetroConfig (dataclass) holds all knobs: length, station_count, walk_radius_m, penalties, buffers. Use it to change experiments.
- Candidate pipeline: `to_demand_points`, `regional_centres_from_demand`, `candidate_station_sites` -> `solve_orienteering_route` (heuristic) or `solve_exact_orienteering_bruteforce` (small brute-force teaching mode).
- Scenario builders: `build_scenarios` (radial search) and `build_orienteering_scenarios` (heuristic NP-hard formulation).
- Scenario utilities: `scenario_geodataframes`, `scenario_metrics`, `scenario_summary_table` for metric extraction and plotting helpers `plot_scenario` / `plot_scenario_satellite`.

Conventions & patterns
- Candidate IDs: `C###` (e.g. `C001`); station codes: `L{line:02d}-S{sid:02d}` (e.g. `L01-S01`). Anchor node with forced centre uses `FORCED-CENTRE`.
- The orienteering heuristic enforces minimum anchor spacing and a total line length budget — brute force mode is intentionally capped (<=9 optional candidates).
- Flood/forbidden handling: prefer adding `data/raw/flood_zones.geojson` — otherwise demo proxies exist (`demo_flood_zones`). Use `relocate_demand_from_forbidden` to move demand away from forbidden areas.

Developer workflows and commands
- Create environment and install deps: python3 -m pip install -r requirements.txt
- Run interactive exploration: python3 -m jupyter lab (or open the notebook in Jupyter/VS Code)
- Download official SIP data (macOS / Linux): python3 scripts/download_wroclaw_data.py
- Avoid leaking secrets: `geocode_google_addresses` requires an API key parameter; prefer passing it via an env var in the notebook, not checked-in files.

Quick examples (minimal, runnable inside the repo or notebook)
```py
from wro_metro import planning as pl
dem = pl.demo_demand()
centre = pl.demo_centres().geometry.iloc[0]
forbidden = pl.demo_flood_zones()
scenarios = pl.build_scenarios(dem, centre, forbidden, None)
metrics = pl.scenario_metrics(scenarios['1'], dem)
print(metrics)
``` 

Notes for agents
- Read `planning.py` to understand assumptions: many helper functions coerce/guess column names and CRS — mimic those patterns when adding features.
- Keep changes small and test in the notebook; there are no unit tests in the repo. Prefer adding small example notebooks or functions rather than broad refactors.
- Performance: heavy use of GeoPandas/Shapely geometric operations (unary_union, intersections). For large data, operations may be slow — consider sampling or vectorized matrix helpers already present (`_fast_orienteering_matrices`).

If anything is unclear or you need specific examples (e.g., adding a new loader, changing MetroConfig defaults, or adding tests), tell me which area to expand.

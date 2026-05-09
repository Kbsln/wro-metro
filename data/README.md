# Dane projektu

Umieszczaj surowe pliki w `data/raw`, a przetworzone eksporty w `data/processed`.

Proponowane pliki wejściowe:

- `data/raw/dem-rejurb-rejstat-shp.zip` albo rozpakowany shapefile z demografia SIP Wroclawia.
- `data/raw/dem-rejurb-rejstat-xls.zip` jako wariant tabelaryczny demografii.
- `data/raw/granice-osiedli.zip` z granicami osiedli.
- `data/raw/wody-powierzchniowe.zip` z wodami powierzchniowymi SIP; to warstwa rzek/wody do oceny przeciec komunikacyjnych, nie mapa zalewowa i nie zakaz dla metra.
- `data/raw/flood_zones.geojson`, `data/raw/flood_zones.gpkg`, `data/raw/flood_zones.zip` albo shapefile obszarow zagrozenia powodziowego z Hydroportalu ISOK.
- `data/raw/polling_places.csv` z kolumnami `address`, `votes`, opcjonalnie `lat`, `lon`.
- `data/raw/geology.geojson` albo warstwa odwiertow/geologii z kolumna `cost_factor`, jesli chcesz modelowac koszt geologiczny.

Notebook ma dane demo, wiec uruchamia sie bez tych plikow. Po pobraniu danych realnych podmieniasz tylko sekcje ladowania danych.

Instrukcja eksportu obszarow zalewowych MZP z QGIS jest w:

- `docs/qgis_flood_zones_export.md`

Pobranie oficjalnych paczek SIP:

```powershell
py -3.11 scripts\download_wroclaw_data.py
```

"""Download official Wroclaw GIS datasets used by the notebook."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


DATASETS = {
    "dem-rejurb-rejstat-shp.zip": "https://geoportal.wroclaw.pl/www/pliki/dem-rejurb-rejstat-shp.zip",
    "dem-rejurb-rejstat-xls.zip": "https://geoportal.wroclaw.pl/www/pliki/dem-rejurb-rejstat-xls.zip",
    "granice-osiedli.zip": "https://geoportal.wroclaw.pl/www/pliki/osiedla/granice-osiedli.zip",
    "wody-powierzchniowe.zip": "https://geoportal.wroclaw.pl/www/pliki/wody-powierzchniowe.zip",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in DATASETS.items():
        target = raw_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists  {target.relative_to(root)}")
            continue
        print(f"download {url}")
        urlretrieve(url, target)
        print(f"saved   {target.relative_to(root)}")


if __name__ == "__main__":
    main()

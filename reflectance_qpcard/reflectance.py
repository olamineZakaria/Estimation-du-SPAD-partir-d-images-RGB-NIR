#!/usr/bin/env python3
"""Calibration de réflectance QPCard RGB+NIR sans Airflow.

Sans argument, le script découvre récursivement tous les jeux de données sous
data/ à partir de leur dossier images_rgb_rect/.

Exemples :
    py -3 reflectance_qpcard/reflectance.py
    py -3 reflectance_qpcard/reflectance.py data/toulouse
    py -3 reflectance_qpcard/reflectance.py --no-plots

Exécuter d'abord :
    py -3 processing/decalage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFLECTANCE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

# Les modules historiques de reflectance_qpcard utilisent des imports locaux
# (config, pipeline, fileio, viz).
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REFLECTANCE_ROOT))

def discover_datasets(data_root: Path = DATA_ROOT) -> list[Path]:
    """Découvre les dossiers contenant images_rgb_rect/."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dossier data introuvable : {data_root}")
    return sorted(
        directory.parent
        for directory in data_root.rglob("images_rgb_rect")
        if directory.is_dir()
    )


def process_site(site_root: Path, save_plots: bool) -> bool:
    """Calibre un site et sauvegarde ses cartes, métadonnées et métriques."""
    from fileio.loader import save_metrics_csv
    from pipeline.processor import process_site_dataset

    shift_path = site_root / "masks" / "qpcard_nir_shift.json"
    if not shift_path.is_file():
        print(f"[ERROR] Décalage absent : {shift_path}")
        print("        Exécuter d'abord : py -3 processing/decalage.py")
        return False

    results = process_site_dataset(
        data_root=site_root,
        save_dir=site_root,
        metadata_dir=site_root / "metadata",
        max_images=None,
        save_plots=save_plots,
    )
    if not results:
        print(f"[WARN] Aucun résultat produit pour {site_root}")
        return False

    save_metrics_csv(results, site_root / "reflectance")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibration de réflectance QPCard RGB+NIR sans Airflow."
    )
    parser.add_argument(
        "sites",
        nargs="*",
        type=Path,
        help="Dossier(s) à traiter. Sans argument, découvre tout sous data/.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactive les graphiques de diagnostic pour accélérer le traitement.",
    )
    args = parser.parse_args()

    try:
        sites = [path.resolve() for path in args.sites] or discover_datasets()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not sites:
        print(f"[ERROR] Aucun jeu de données trouvé sous {DATA_ROOT}")
        return 1

    print(f"{len(sites)} jeu(x) de données trouvé(s).")
    failures = 0

    for site in sites:
        print(f"\n=== {site} ===")
        if not site.is_dir():
            print(f"[ERROR] Dossier introuvable : {site}")
            failures += 1
            continue

        try:
            if not process_site(site, not args.no_plots):
                failures += 1
        except Exception as exc:
            print(f"[ERROR] Échec de la calibration de {site} : {exc}")
            failures += 1

    if failures:
        print(f"\nTerminé avec {failures} jeu(x) de données en erreur.")
        return 1

    print("\nToutes les calibrations sont terminées.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

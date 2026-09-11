#!/usr/bin/env python3
"""
Génération du masque combiné final pour le scénario :
Vegetation-no-specular-no-shadow-no-epi-no-tige-no-Senescent_veg-no-height20-120
Sauvegarde dans data/<site>/masks/combined/<stem>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from processing.spad_pixel_stats import (
    create_vegetation_mask,
    detect_shadow_mask,
    detect_specular_mask,
    detect_epi_mask,
    detect_tige_mask,
    detect_senescent_mask,
    detect_height_mask
)

# Seuil de hauteur (cm) a utiliser par site. Les sites organises par date
# (ex: zurich) sont identifies par "<site>/<date>", cle qui correspond au
# "site" passe a process_site_combined (meme convention que web_app/app.py::list_sites).
SITE_HEIGHT_TAG = {
    "clermont-ferrand": "45-100",
    "mauguio": "20-100",
    "toulouse": "20-100",
    "zurich/2026-03-23": "8-100",
    "zurich/2026-04-07": "10-100",
    "zurich/2026-04-21": "20-88",
    "zurich/2026-05-05": "44-100",
    "zurich/2026-06-01": "40-100",
}
DEFAULT_HEIGHT_TAG = "20-100"

def process_one_image_combined(site_root: Path, site: str, image_name: str) -> None:
    stem = Path(image_name).stem

    # Répertoires de sortie
    out_dir = site_root / "masks" / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.png"

    # Charger les masques binaires de base
    try:
        veg_mask = create_vegetation_mask(site_root, stem)
    except Exception as e:
        print(f"      [SKIP] Masque végétation absent pour {stem} ({e})")
        return

    # Déterminer le scénario pour charger le bon masque de hauteur
    height_tag = SITE_HEIGHT_TAG.get(site, DEFAULT_HEIGHT_TAG)
    scenario_name = f"Vegetation-no-specular-no-shadow-no-epi-no-tige-no-Senescent_veg-no-height{height_tag}"

    shape = veg_mask.shape
    shadow = detect_shadow_mask(site_root, stem, shape)
    specular = detect_specular_mask(site_root, stem, shape)
    epi = detect_epi_mask(site_root, stem, shape)
    tige = detect_tige_mask(site_root, stem, shape)
    senescent = detect_senescent_mask(site_root, stem, shape)
    height = detect_height_mask(site_root, stem, shape, scenario_name=scenario_name)

    # Combinaison logique des masques
    # Végétation ET non(ombres) ET non(reflets) ET non(épis) ET non(tiges) ET non(sénescence) ET hauteur
    combined = veg_mask & ~shadow & ~specular & ~epi & ~tige & ~senescent & height

    # Convertir en binaire 0/255 (L) et sauvegarder
    out_img = (combined.astype(np.uint8)) * 255
    cv2.imwrite(str(out_path), out_img)

def process_site_combined(site_root: Path, site: str, max_workers: int | None = None) -> None:
    print(f"--- Début génération masques combinés pour : {site} ---")
    
    # On se base sur les images existantes du dossier images_rgb_rect
    rgb_dir = site_root / "images_rgb_rect"
    if not rgb_dir.is_dir():
        print(f"[SKIP] Aucun dossier images_rgb_rect pour {site}.")
        return

    images = sorted([f.name for f in rgb_dir.glob("*.png")])
    if not images:
        print(f"[SKIP] Aucune image trouvée pour {site}.")
        return

    print(f"Images à traiter : {len(images)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_image_combined, site_root, site, img): img
            for img in images
        }
        for future in as_completed(futures):
            img = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"   [ERROR] Erreur sur {img} : {e}")

    print(f"--- Fin génération masques combinés pour {site} ---")


def discover_datasets(data_root: Path = DATA_ROOT) -> list[Path]:
    """Découvre tous les sites/dates contenant images_rgb_rect/."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dossier data introuvable : {data_root}")
    return sorted(
        rgb_dir.parent
        for rgb_dir in data_root.rglob("images_rgb_rect")
        if rgb_dir.is_dir()
    )


def site_identifier(site_root: Path) -> str:
    """Retourne l'identifiant relatif utilisé pour les seuils de hauteur."""
    try:
        return site_root.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return site_root.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Génère les masques combinés sans Airflow."
    )
    parser.add_argument(
        "sites",
        nargs="*",
        type=Path,
        help="Dossier(s) à traiter. Sans argument, découvre tout sous data/.",
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

    for site_root in sites:
        print(f"\n=== {site_root} ===")
        if not site_root.is_dir():
            print(f"[ERROR] Dossier introuvable : {site_root}")
            failures += 1
            continue

        try:
            process_site_combined(
                site_root=site_root,
                site=site_identifier(site_root),
                max_workers=None,
            )
        except Exception as exc:
            print(f"[ERROR] Échec de {site_root} : {exc}")
            failures += 1

    if failures:
        print(f"\nTerminé avec {failures} jeu(x) de données en erreur.")
        return 1

    print("\nTous les masques combinés sont terminés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

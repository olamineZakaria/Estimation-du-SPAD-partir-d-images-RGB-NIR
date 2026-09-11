#!/usr/bin/env python3
"""Calcule le décalage QPCard RGB/NIR sans Airflow.

Exemples :
    py -3 processing/decalage.py
    py -3 processing/decalage.py data/toulouse
    py -3 processing/decalage.py data/toulouse data/mauguio

Sans argument, tous les dossiers situés sous data/ et contenant un dossier
helper/ sont traités automatiquement, y compris les sites imbriqués.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from mask_shift import MaskShift


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def discover_sites(data_root: Path = DATA_ROOT) -> list[Path]:
    """Trouve récursivement tous les jeux de données possédant helper/."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dossier data introuvable : {data_root}")
    return sorted(path.parent for path in data_root.rglob("helper") if path.is_dir())


def find_helper_mask(helper_dir: Path, camera_tag: str) -> Path:
    matches = sorted(
        path
        for path in helper_dir.iterdir()
        if path.is_file()
        and path.name.startswith("mask_")
        and camera_tag in path.name
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not matches:
        raise FileNotFoundError(
            f"Aucun masque helper 'mask_*{camera_tag}*' dans {helper_dir}"
        )
    return matches[0]


def calculate_shift(site: Path, rgb_camera_tag: str, nir_camera_tag: str) -> Path:
    helper_dir = site / "helper"
    if not helper_dir.is_dir():
        raise FileNotFoundError(f"Dossier helper introuvable : {helper_dir}")

    rgb_path = find_helper_mask(helper_dir, rgb_camera_tag)
    nir_path = find_helper_mask(helper_dir, nir_camera_tag)
    rgb_mask = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    nir_mask = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)

    if rgb_mask is None:
        raise FileNotFoundError(f"Impossible de lire {rgb_path}")
    if nir_mask is None:
        raise FileNotFoundError(f"Impossible de lire {nir_path}")

    dx, dy = MaskShift(rgb_mask, nir_mask).get_shift()
    output_path = site / "masks" / "qpcard_nir_shift.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dx": dx,
                "dy": dy,
                "rgb_helper": rgb_path.name,
                "nir_helper": nir_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] dx={dx}px, dy={dy}px -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcule qpcard_nir_shift.json à partir des masques helper RGB et NIR."
    )
    parser.add_argument(
        "sites",
        nargs="*",
        type=Path,
        help="Dossier(s) de site. Sans argument, découvre tous les sites sous data/.",
    )
    parser.add_argument("--rgb-camera-tag", default="Camera1")
    parser.add_argument("--nir-camera-tag", default="Camera3")
    args = parser.parse_args()

    try:
        sites = args.sites or discover_sites()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not sites:
        print(f"[ERROR] Aucun dossier helper trouvé sous {DATA_ROOT}")
        return 1

    print(f"{len(sites)} jeu(x) de données trouvé(s).")
    failures = 0

    for site in sites:
        site = site.resolve()
        print(f"\n=== {site} ===")

        if not site.is_dir():
            print(f"[ERROR] Dossier de site introuvable : {site}")
            failures += 1
            continue

        try:
            calculate_shift(site, args.rgb_camera_tag, args.nir_camera_tag)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            failures += 1

    if failures:
        print(f"\nTerminé avec {failures} erreur(s).")
        return 1

    print("\nTous les décalages ont été calculés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

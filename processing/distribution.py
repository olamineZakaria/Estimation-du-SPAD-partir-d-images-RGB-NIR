"""
Distribution des bandes de réflectance avec le masque combiné
==================================================================
Script autonome sans Airflow. Pour chaque image,
trace la distribution des valeurs de pixel des 4 bandes (B, G, R, NIR)
restreinte aux pixels retenus par le masque combine final (genere par
processing/combine_masks.py) :

    data/<site>/masks/combined/<stem>.png

Sauvegarde dans data/<site>/distribution/combined/<stem>.png (meme nom
que l'image source).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

SCENARIO_NAME = "combined"


def process_one_image(site_root: Path, stem: str, save_dir: Path) -> list[str]:
    """Retourne la liste des scenarios ignores (masque vide, rien a tracer)."""
    from processing.reflectance_distribution import plot_band_distributions
    from processing.spad_pixel_stats import apply_mask, load_reflectance_bands

    bands = load_reflectance_bands(site_root, stem)

    mask_path = site_root / "masks" / "combined" / f"{stem}.png"
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Masque combine absent : {mask_path}")
    mask = mask_img > 0

    masked_bands = apply_mask(bands, mask)
    ok = plot_band_distributions(
        masked_bands,
        f"{stem} — {SCENARIO_NAME}",
        save_dir / SCENARIO_NAME / f"{stem}.png",
    )
    return [] if ok else [SCENARIO_NAME]


def _process_stem(site_root: Path, stem: str, save_dir: Path) -> tuple[str, list[str] | str]:
    """Traite une image dans un thread worker."""
    try:
        return "ok", process_one_image(site_root, stem, save_dir)
    except FileNotFoundError:
        return "skip", f"Cartes de réflectance absentes (image probablement exclue lors de l'étalonnage)."


def process_site(site_root: Path, max_images: int | None = None, max_workers: int | None = None) -> None:
    site_root = Path(site_root)
    stems = sorted(p.stem for p in (site_root / "reflectance" / "B").glob("*.tif"))
    if max_images is not None:
        stems = stems[:max_images]

    save_dir = site_root / "distribution"
    total = len(stems)
    if total == 0:
        print(f"[DONE] 0/0 images traitees, 0 ignorees — {site_root.name}")
        return

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, total)
    max_workers = max(1, max_workers)

    skipped = 0
    done = 0

    # Chaque image est indépendante (I/O + graphiques matplotlib) : traitement
    # parallèle avec des threads, sans dupliquer les grands tableaux en mémoire.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_stem, site_root, stem, save_dir): stem
            for stem in stems
        }
        for future in as_completed(futures):
            stem = futures[future]
            done += 1
            status, payload = future.result()
            if status == "ok":
                note = f" (vide: {', '.join(payload)})" if payload else ""
                print(f"[OK]   ({done}/{total}) {site_root.name} — {stem}{note}")
            else:
                skipped += 1
                print(f"[SKIP] ({done}/{total}) {site_root.name} — {payload}")

    print(f"[DONE] {total - skipped}/{total} images traitees, {skipped} ignorees — {site_root.name}")


def discover_datasets(data_root: Path = DATA_ROOT) -> list[Path]:
    """Découvre tous les sites/dates contenant reflectance/B/."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dossier data introuvable : {data_root}")
    return sorted(
        reflectance_dir.parent
        for reflectance_dir in data_root.rglob("reflectance")
        if reflectance_dir.is_dir() and (reflectance_dir / "B").is_dir()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distribution des bandes B, G, R et NIR sous masque combiné."
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
        print(f"[ERROR] Aucun jeu de données avec reflectance/B trouvé sous {DATA_ROOT}")
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
            process_site(site, max_images=None, max_workers=None)
        except Exception as exc:
            print(f"[ERROR] Échec de {site} : {exc}")
            failures += 1

    if failures:
        print(f"\nTerminé avec {failures} jeu(x) de données en erreur.")
        return 1

    print("\nToutes les distributions sont terminées.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

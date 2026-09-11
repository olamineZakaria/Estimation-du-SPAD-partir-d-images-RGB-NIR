"""
Diagnostic complet des donnees pour_spad_* dans data/.

Verifie pour chaque dossier :
  - presence des sous-dossiers requis
  - nombre de fichiers par sous-dossier
  - stems complets (RGB+NIR) et RGB seul
  - stems manquants par dossier
  - taille des images (premier fichier trouve)

Usage :
    python diagnostic.py
    python diagnostic.py --data-dir ../data
"""
import argparse
import sys
from pathlib import Path

import cv2


REQUIRED_RGB     = ["rgb", "xml_rgb", "masks/qpcard"]
REQUIRED_NIR     = ["nir", "xml_nir"]
OPTIONAL         = ["masks/vegetation"]


def check_folder(folder: Path) -> dict:
    info = {
        "name":           folder.name,
        "path":           str(folder),
        "dirs":           {},
        "stems_rgb":      set(),
        "stems_nir":      set(),
        "stems_xml_rgb":  set(),
        "stems_xml_nir":  set(),
        "stems_qpcard":   set(),
        "stems_veg":      set(),
        "complete_rgb":   set(),
        "complete_full":  set(),
        "missing_dirs":   [],
        "image_size_rgb": None,
        "image_size_nir": None,
    }

    # --- Inventaire des sous-dossiers ---
    for sub in ["rgb", "nir", "xml_rgb", "xml_nir",
                "masks/qpcard", "masks/vegetation"]:
        d = folder / sub.replace("/", "\\")
        if d.exists():
            ext = "*.xml" if "xml" in sub else "*.png"
            files = list(d.glob(ext))
            info["dirs"][sub] = len(files)
        else:
            info["dirs"][sub] = None

    # --- Stems par dossier ---
    def stems(sub, ext):
        d = folder / sub.replace("/", "\\")
        if not d.exists():
            return set()
        return {f.stem for f in d.glob(f"*.{ext}")}

    info["stems_rgb"]     = stems("rgb",              "png")
    info["stems_nir"]     = stems("nir",              "png")
    info["stems_xml_rgb"] = stems("xml_rgb",          "xml")
    info["stems_xml_nir"] = stems("xml_nir",          "xml")
    info["stems_qpcard"]  = stems("masks/qpcard",     "png")
    info["stems_veg"]     = stems("masks/vegetation", "png")

    # --- Completude ---
    info["complete_rgb"] = (
        info["stems_rgb"] & info["stems_xml_rgb"] & info["stems_qpcard"]
    )
    info["complete_full"] = (
        info["complete_rgb"] & info["stems_nir"] & info["stems_xml_nir"]
    )

    # --- Dossiers manquants ---
    for req in REQUIRED_RGB + REQUIRED_NIR:
        d = folder / req.replace("/", "\\")
        if not d.exists():
            info["missing_dirs"].append(req)

    # --- Taille image RGB ---
    rgb_dir = folder / "rgb"
    if rgb_dir.exists():
        sample = next(rgb_dir.glob("*.png"), None)
        if sample:
            img = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
            if img is not None:
                info["image_size_rgb"] = (img.shape[1], img.shape[0])

    # --- Taille image NIR ---
    nir_dir = folder / "nir"
    if nir_dir.exists():
        sample = next(nir_dir.glob("*.png"), None)
        if sample:
            img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                info["image_size_nir"] = (img.shape[1], img.shape[0])

    return info


def print_report(infos: list, log_path: Path):
    lines = []

    def w(msg=""):
        lines.append(msg)
        print(msg)

    w("=" * 70)
    w("  DIAGNOSTIC COMPLET DES DONNEES")
    w("=" * 70)

    total_complete_rgb  = 0
    total_complete_full = 0

    for info in infos:
        w()
        w(f"  Dossier : {info['name']}")
        w("-" * 70)

        # Sous-dossiers
        w("  Sous-dossiers :")
        for sub, count in info["dirs"].items():
            if count is None:
                status = "ABSENT"
            else:
                status = f"{count} fichier(s)"
            w(f"    {'[OK]' if count is not None else '[!!]'}  {sub:<25} {status}")

        # Taille images
        if info["image_size_rgb"]:
            w(f"  Taille RGB : {info['image_size_rgb'][0]} x {info['image_size_rgb'][1]} px")
        if info["image_size_nir"]:
            w(f"  Taille NIR : {info['image_size_nir'][0]} x {info['image_size_nir'][1]} px")

        # Completude
        n_rgb  = len(info["complete_rgb"])
        n_full = len(info["complete_full"])
        total_complete_rgb  += n_rgb
        total_complete_full += n_full

        w()
        w(f"  Stems complets (RGB seul)   : {n_rgb}")
        w(f"  Stems complets (RGB + NIR)  : {n_full}")

        # Stems manquants dans chaque groupe
        only_rgb = info["stems_rgb"] - info["stems_nir"]
        no_qpcard = info["stems_rgb"] - info["stems_qpcard"]
        no_xml_rgb = info["stems_rgb"] - info["stems_xml_rgb"]
        no_xml_nir = info["stems_nir"] - info["stems_xml_nir"]

        if no_qpcard:
            w(f"  [!!] Masque qpcard manquant  : {len(no_qpcard)} image(s)")
        if no_xml_rgb:
            w(f"  [!!] XML RGB manquant        : {len(no_xml_rgb)} image(s)")
        if no_xml_nir:
            w(f"  [!!] XML NIR manquant        : {len(no_xml_nir)} image(s)")
        if only_rgb:
            w(f"  [!!] RGB sans NIR            : {len(only_rgb)} image(s)")

        if info["missing_dirs"]:
            w(f"  [!!] Dossiers absents        : {', '.join(info['missing_dirs'])}")

        if n_rgb == n_full == 0:
            w("  [!!] Aucune image exploitable !")
        elif n_full == n_rgb:
            w("  [OK] Toutes les images sont completes (RGB + NIR)")
        else:
            w(f"  [OK] {n_full}/{n_rgb} images completes avec NIR")

    # Recapitulatif global
    w()
    w("=" * 70)
    w("  RECAPITULATIF GLOBAL")
    w("=" * 70)
    w(f"  Dossiers analyses       : {len(infos)}")
    w(f"  Total images RGB seul   : {total_complete_rgb}")
    w(f"  Total images RGB + NIR  : {total_complete_full}")
    w("=" * 70)

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[LOG] Diagnostic sauvegarde dans {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path(__file__).parent.parent / "data")
    args = p.parse_args()

    folders = sorted(args.data_dir.glob("for_spad_*"))
    if not folders:
        print("[WARN] Aucun dossier for_spad_* trouve dans", args.data_dir)
        sys.exit(1)

    infos    = [check_folder(f) for f in folders]
    log_path = Path(__file__).parent / "diagnostic.txt"
    print_report(infos, log_path)


if __name__ == "__main__":
    main()

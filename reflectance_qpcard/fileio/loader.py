"""
Chargement d'images, détection de triplets et sauvegarde des résultats.
"""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from config import BANDS


def parse_xml_metadata(xml_path) -> dict:
    """Extrait ExposureTime (TI en secondes) et gain = ISOSpeedRatings/100 depuis un XML Exif."""
    root = ET.parse(str(xml_path)).getroot()
    ti   = float(root.findtext("ExposureTime")) / 1_000_000  # microsecondes → secondes
    iso  = float(root.findtext("ISOSpeedRatings"))
    return {"TI": ti, "gain": iso / 100.0}


def find_triplets(test_dir, mode: str = "RGB+NIR") -> list:
    """
    Retourne les stems communs aux dossiers attendus selon le mode.

    mode="RGB+NIR" (défaut) : rgb/, nir/, xml_rgb/, xml_nir/, masks/qpcard/
    mode="RGB"              : rgb/, xml_rgb/, masks/qpcard/ uniquement
    """
    test_dir = Path(test_dir)

    rgb_stems      = {f.stem for f in (test_dir / "rgb").glob("*.png")}
    xml_rgb_stems  = {f.stem for f in (test_dir / "xml_rgb").glob("*.xml")}
    mask_rgb_stems = {f.stem for f in (test_dir / "masks" / "qpcard").glob("*.png")}

    if mode == "RGB+NIR":
        nir_stems     = {f.stem for f in (test_dir / "nir").glob("*.png")}
        xml_nir_stems = {f.stem for f in (test_dir / "xml_nir").glob("*.xml")}
        common = sorted(
            rgb_stems & nir_stems & xml_rgb_stems & xml_nir_stems & mask_rgb_stems
        )
    else:
        common = sorted(rgb_stems & xml_rgb_stems & mask_rgb_stems)

    return common


def find_site_pairs(
    data_root,
    rgb_dir: str = "images_rgb_rect",
    nir_dir: str = "images_nir",
    nir_rect_dir: str = "images_nir_rect",
    xml_rgb_dir: str = "xml_rgb_rect",
    xml_nir_dir: str = "xml_nir_rect",
    mask_dir: str = "qpcard",
    rgb_camera_tag: str = "Camera1",
    nir_camera_tag: str = "Camera3",
) -> list:
    """
    Associe chaque image RGB (caméra `rgb_camera_tag`, ex. Camera1) à son image
    NIR (caméra `nir_camera_tag`, ex. Camera3) du même plot/tir, en substituant
    le tag caméra dans le nom de fichier (RGB et NIR sont deux capteurs
    physiques distincts, donc des noms de fichiers différents pour un même tir).

    nir_dir (brute) sert à la régression ELM, nir_rect_dir (rectifiée) sert à
    produire la carte de réflectance finale — les deux doivent exister.

    Il n'y a qu'un masque QPCard par stem (masks/<mask_dir>/, en géométrie
    RGB) : le masque NIR est dérivé à la volée en le décalant avec la valeur
    (dx, dy) sauvegardée par processing/decalage.py dans masks/qpcard_nir_shift.json (voir
    process_site_dataset), donc aucun masque NIR par image n'est requis ici.

    Retourne les stems RGB pour lesquels l'image NIR brute, l'image NIR
    rectifiée, les deux XML et le masque QPCard existent.
    """
    data_root = Path(data_root)

    rgb_stems = {
        f.stem for f in (data_root / rgb_dir).glob("*.png")
        if rgb_camera_tag in f.stem
    }
    nir_stems      = {f.stem for f in (data_root / nir_dir).glob("*.png")}
    nir_rect_stems = {f.stem for f in (data_root / nir_rect_dir).glob("*.png")}
    xml_rgb_stems  = {f.stem for f in (data_root / xml_rgb_dir).glob("*.xml")}
    xml_nir_stems  = {f.stem for f in (data_root / xml_nir_dir).glob("*.xml")}
    mask_stems     = {f.stem for f in (data_root / "masks" / mask_dir).glob("*.png")}

    complete = []
    for stem in sorted(rgb_stems):
        nir_stem = stem.replace(rgb_camera_tag, nir_camera_tag)
        if (
            nir_stem in nir_stems
            and nir_stem in nir_rect_stems
            and stem in xml_rgb_stems
            and nir_stem in xml_nir_stems
            and stem in mask_stems
        ):
            complete.append(stem)

    return complete


def save_image_metadata(result: dict, stem: str, metadata_dir, violations: list = None) -> Path:
    """
    Sauvegarde en JSON les métadonnées d'une image traitée : EXIF (TI, gain)
    utilisées pour la calibration, résumé des régressions ELM par bande
    (slope, intercept, r2, rmse, n), et les éventuelles violations de
    contraintes (image alors non calibrée, mais diagnostic conservé).
    """
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    models = {
        band: {k: m[k] for k in ("slope", "intercept", "r2", "rmse", "n")}
        for band, m in result["_models"].items() if m is not None
    }

    payload = {
        "stem":       stem,
        "exif":       result["_meta"],
        "models":     models,
        "violations": violations or [],
    }

    path = metadata_dir / f"{stem}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[IO] Métadonnées sauvegardées : {path}")
    return path


def save_site_metadata(site_metadata: dict, metadata_dir, filename: str = "metadata.json") -> Path:
    """
    Sauvegarde dans un fichier JSON unique l'ensemble des métadonnées de toutes les images d'un site :
    EXIF (TI, gain) par caméra, modèles de calibration ELM par bande (slope, intercept, r2, rmse, n),
    et violations éventuelles.
    """
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(site_metadata, f, indent=2, ensure_ascii=False)

    print(f"[IO] Métadonnées globales du site sauvegardées dans un seul fichier : {path}")
    return path


def save_reflectance(
    result: dict,
    stem: str,
    save_dir,
    mode: str = "RGB+NIR",
) -> None:
    """
    Sauvegarde les cartes de réflectance en TIF séparés par bande.

    mode="RGB"     : R, G, B — 8 bits  (0 % → 0, 50 % → 255)
    mode="RGB+NIR" : R, G, B, NIR — 16 bits (0 % → 0, 100 % → 65535)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == "RGB":
        bands_to_save = ("R", "G", "B")
        scale = 255.0 / 0.5
        dtype = np.uint8
        effective_depth = 8
    else:
        bands_to_save = ("R", "G", "B", "NIR")
        scale = 65535.0
        dtype = np.uint16
        effective_depth = 16

    for band in bands_to_save:
        if band not in result:
            continue
        band_dir = save_dir / "reflectance" / band
        band_dir.mkdir(parents=True, exist_ok=True)
        arr  = result[band]
        out  = np.clip(arr * scale, 0, np.iinfo(dtype).max).astype(dtype)
        path = band_dir / f"{stem}.tif"
        cv2.imwrite(str(path), out)

    print(f"[IO] TIF {effective_depth}-bits ({mode}) sauvegardés dans {save_dir}/reflectance/[R|G|B] pour {stem}")


def save_metrics_csv(
    results: dict,
    out_dir,
    mode: str = "RGB+NIR",
) -> Path:
    """
    Sauvegarde les métriques de calibration ELM (pente, intercept, R², RMSE)
    pour chaque image et chaque bande dans un CSV unique.

    results : dict stem → résultat process_single (doit contenir "_models")
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics_calibration.csv"

    active_bands = [b for b in BANDS if b != "NIR" or mode == "RGB+NIR"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "band", "slope", "intercept", "r2", "rmse", "n"])
        for stem, res in results.items():
            for band in active_bands:
                m = res["_models"].get(band)
                if m is None:
                    continue
                writer.writerow(
                    [stem, band, m["slope"], m["intercept"], m["r2"], m["rmse"], m["n"]]
                )

    print(f"[IO] Métriques de calibration (R², RMSE) sauvegardées : {path}")
    return path

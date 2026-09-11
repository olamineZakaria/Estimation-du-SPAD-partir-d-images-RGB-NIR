"""
Traitement haut niveau : d'un triplet (RGB, NIR, masques) vers les cartes de réflectance.
"""
import json
import warnings
from pathlib import Path
from typing import Optional

import shutil
from datetime import datetime
import cv2
import numpy as np
import yaml

from config import BANDS, PATCH_ORDER, PATCH_CENTERS, YAML_PATH, SAT_HIGH
from .extractor  import warp_to_template, extract_all_patches
from .calibration import fit_all_bands, apply_calibration
from fileio.loader import (
    find_triplets, find_site_pairs, save_reflectance, parse_xml_metadata,
    save_image_metadata, save_site_metadata,
)
from viz.plots import plot_regression, plot_warped_qpcard


def _load_refl_ref(yaml_path=YAML_PATH) -> dict:
    """Charge les réflectances de référence depuis le YAML."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    patches = data["qpcard"]["patches"]
    return {band: [p["reflectance"][band] for p in patches] for band in BANDS}


def process_single(
    rgb_path,
    nir_path       = None,
    xml_rgb_path   = None,
    xml_nir_path   = None,
    mask_rgb_path  = None,
    nir_shift             = None,
    nir_output_path       = None,
    refl_ref:      dict = None,
    force_zero:    bool = True,
    save_dir             = None,
    metadata_dir          = None,
    garbage_dir:    Optional[Path] = None,
    n_patches_rgb: int  = 2,
    mode:          str  = "RGB+NIR",
    save_plots:    bool = True,
    save_individual_json: bool = False,
    metadata_collector: Optional[dict] = None,
) -> Optional[dict]:
    """
    Traite un triplet (ou paire RGB seule) et retourne les cartes de réflectance.

    mode="RGB+NIR" : traite RGB + NIR → 4 bandes 16 bits (R, G, B, NIR)
    mode="RGB"     : traite RGB uniquement → 3 bandes 8 bits (R, G, B)

    nir_path sert à la régression ELM (DN mesurés sur la QPCard). Si
    nir_output_path est fourni, la calibration résultante est appliquée à cette
    image à la place de nir_path (ex. image NIR brute pour la régression,
    rectifiée pour la carte de réflectance finale) ; sinon nir_path sert aux deux.

    nir_shift = (dx, dy) optionnel : décale mask_rgb à la volée (voir
    processing.mask_shift.shift_mask) pour l'utiliser comme masque NIR — évite
    d'avoir à stocker un masque NIR par image. Si None, mask_rgb est réutilisé
    tel quel comme masque NIR.

    Si metadata_dir est fourni, le JSON de métadonnées est sauvegardé
    systématiquement — même si l'image est ensuite rejetée pour contrainte non
    respectée (diagnostic). Le plot de régression et le plot de la QPCard
    redressée (matplotlib, coûteux à l'unité) ne sont générés en plus que si
    save_plots=True.

    Retourne dict ou None si contrainte non respectée / masque manquant.
    """
    rgb_path      = Path(rgb_path)
    mask_rgb_path = Path(mask_rgb_path)
    if garbage_dir is not None:
        garbage_dir = Path(garbage_dir)
        garbage_dir.mkdir(parents=True, exist_ok=True)

    if not mask_rgb_path.exists():
        reason = f"masque RGB introuvable : {mask_rgb_path.name}"
        print(f"[SKIP] {reason}")
        if garbage_dir is not None:
            # move related files to garbage and write a JSON with reason
            files_to_move = [rgb_path,]
            if nir_path is not None:
                files_to_move.append(Path(nir_path))
            if xml_rgb_path is not None:
                files_to_move.append(Path(xml_rgb_path))
            if xml_nir_path is not None:
                files_to_move.append(Path(xml_nir_path))
            if mask_rgb_path.exists():
                files_to_move.append(mask_rgb_path)
            moved = []
            for p in files_to_move:
                try:
                    if p is None:
                        continue
                    if Path(p).exists():
                        dest = garbage_dir / Path(p).name
                        shutil.move(str(p), str(dest))
                        moved.append(str(dest.name))
                except Exception:
                    pass
            meta = {
                "reason": reason,
                "moved_files": moved,
                "time": datetime.now().isoformat(),
            }
            json_path = garbage_dir / f"{rgb_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(meta, jf, ensure_ascii=False, indent=2)
        return None

    if refl_ref is None:
        refl_ref = _load_refl_ref()

    img_bgr  = cv2.imread(str(rgb_path),      cv2.IMREAD_UNCHANGED)
    mask_rgb = cv2.imread(str(mask_rgb_path), cv2.IMREAD_GRAYSCALE)

    if img_bgr  is None: raise FileNotFoundError(f"Image RGB introuvable : {rgb_path}")
    if mask_rgb is None: raise FileNotFoundError(f"Masque RGB introuvable : {mask_rgb_path}")

    meta_rgb  = parse_xml_metadata(xml_rgb_path)
    scale_rgb = meta_rgb["TI"] * meta_rgb["gain"]

    warped_rgb = warp_to_template(img_bgr, mask_rgb)

    if mode == "RGB+NIR":
        img_nir = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
        if img_nir is None: raise FileNotFoundError(f"Image NIR introuvable : {nir_path}")
        meta_nir  = parse_xml_metadata(xml_nir_path)
        scale_nir = meta_nir["TI"] * meta_nir["gain"]

        if nir_shift is not None:
            from processing.mask_shift import shift_mask
            dx, dy = nir_shift
            mask_nir = shift_mask(mask_rgb, dx, dy)
        else:
            mask_nir = mask_rgb  # pas de décalage fourni : on réutilise le masque RGB tel quel

        warped_nir = warp_to_template(img_nir, mask_nir)

        if nir_output_path is not None:
            img_nir_output = cv2.imread(str(nir_output_path), cv2.IMREAD_GRAYSCALE)
            if img_nir_output is None:
                raise FileNotFoundError(f"Image NIR (sortie) introuvable : {nir_output_path}")
        else:
            img_nir_output = img_nir
    else:
        img_nir = warped_nir = meta_nir = scale_nir = img_nir_output = None

    dn_record, extract_violations = extract_all_patches(warped_rgb, warped_nir)

    # Normalisation DN / (TI × gain) avant régression ELM
    for band in ("R", "G", "B"):
        dn_record[band] = [
            v / scale_rgb if (v is not None and not np.isnan(v)) else np.nan
            for v in dn_record[band]
        ]
    if "NIR" in dn_record:
        for p_name, v in zip(PATCH_ORDER, dn_record["NIR"]):
            if v is None or np.isnan(v):
                print(f"  [INFO] [NIR] Patch '{p_name}' exclu de la calibration (saturé DN >= {SAT_HIGH} ou sous-exposé)")
        dn_record["NIR"] = [
            v / scale_nir if (v is not None and not np.isnan(v)) else np.nan
            for v in dn_record["NIR"]
        ]

    models, calib_violations = fit_all_bands(
        dn_record, refl_ref, force_zero, n_patches_rgb, mode=mode
    )

    all_violations = extract_violations + calib_violations

    meta_full = {"rgb": meta_rgb}
    if mode == "RGB+NIR":
        meta_full["nir"] = meta_nir

    models_summary = {
        band: {k: m[k] for k in ("slope", "intercept", "r2", "rmse", "n")}
        for band, m in models.items() if m is not None
    } if models else {}

    meta_entry = {
        "stem": rgb_path.stem,
        "exif": meta_full,
        "models": models_summary,
        "violations": all_violations or [],
    }

    if metadata_collector is not None:
        metadata_collector[rgb_path.stem] = meta_entry

    if metadata_dir is not None:
        if save_individual_json:
            diag = {"_meta": meta_full, "_models": models}
            save_image_metadata(diag, rgb_path.stem, metadata_dir, violations=all_violations)
        if save_plots:
            plot_regression(models, rgb_path.stem, save_dir=metadata_dir)
            plot_warped_qpcard(warped_rgb, warped_nir, rgb_path.stem, save_dir=metadata_dir)

    if all_violations:
        reason = "; ".join(all_violations)
        print(f"[SKIP] {rgb_path.stem} — contraintes non respectées :")
        for msg in all_violations:
            print(msg)
        if garbage_dir is not None:
            # move related files to garbage and write a JSON with violations
            files_to_move = [rgb_path,]
            if nir_path is not None:
                files_to_move.append(Path(nir_path))
            if nir_output_path is not None:
                files_to_move.append(Path(nir_output_path))
            if xml_rgb_path is not None:
                files_to_move.append(Path(xml_rgb_path))
            if xml_nir_path is not None:
                files_to_move.append(Path(xml_nir_path))
            if mask_rgb_path.exists():
                files_to_move.append(mask_rgb_path)
            moved = []
            for p in files_to_move:
                try:
                    if p is None:
                        continue
                    if Path(p).exists():
                        dest = garbage_dir / Path(p).name
                        shutil.move(str(p), str(dest))
                        moved.append(str(dest.name))
                except Exception:
                    pass
            meta = {
                "reason": "violations",
                "messages": all_violations,
                "moved_files": moved,
                "time": datetime.now().isoformat(),
            }
            json_path = garbage_dir / f"{rgb_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(meta, jf, ensure_ascii=False, indent=2)
        return None

    # Application : normaliser l'image entière avant d'appliquer ρ = a × DN_norm
    img_bgr_norm = img_bgr.astype(np.float32) / scale_rgb
    rgb_ref = apply_calibration(img_bgr_norm, models, mode="RGB")

    result = {
        "R":        np.clip(rgb_ref[:, :, 2], 0.0, 1.0).astype(np.float32),
        "G":        np.clip(rgb_ref[:, :, 1], 0.0, 1.0).astype(np.float32),
        "B":        np.clip(rgb_ref[:, :, 0], 0.0, 1.0).astype(np.float32),
        "_dn":      dn_record,
        "_models":  models,
        "_warpeds": (warped_rgb, warped_nir),
        "_meta":    meta_full,
    }

    if mode == "RGB+NIR":
        img_nir_norm    = img_nir_output.astype(np.float32) / scale_nir
        nir_ref         = apply_calibration(img_nir_norm, models, mode="NIR")
        result["NIR"]   = np.clip(nir_ref, 0.0, 1.0).astype(np.float32)

    if save_dir is not None:
        save_reflectance(result, rgb_path.stem, save_dir, mode=mode)

    return result


def process_batch(
    test_dir,
    force_zero:    bool = True,
    save_dir            = None,
    predict_masks: bool = False,
    model_path          = None,
    n_patches_rgb: int  = 2,
    mode:          str  = "RGB+NIR",
) -> dict:
    """
    Traite tous les triplets/paires trouvés dans test_dir.

    mode="RGB+NIR" : attend rgb/, nir/, xml_rgb/, xml_nir/,
                     masks/qpcard/, masks/qpcard_nir/
    mode="RGB"     : attend rgb/, xml_rgb/, masks/qpcard/ uniquement

    Retourne dict stem → résultat process_single
    """
    test_dir = Path(test_dir)
    refl_ref = _load_refl_ref()

    stems = find_triplets(test_dir, mode=mode)

    if not stems:
        print("[WARN] Aucun fichier complet trouvé dans", test_dir)
        if mode == "RGB+NIR":
            print("       Requis : rgb/, nir/, xml_rgb/, xml_nir/, masks/qpcard/")
        else:
            print("       Requis : rgb/, xml_rgb/, masks/qpcard/")
        return {}

    print(f"[OK] {len(stems)} image(s) trouvée(s)")

    predictor = None
    if predict_masks:
        if model_path is None:
            raise ValueError("model_path requis quand predict_masks=True")
        from core.predictor import QPCardMaskPredictor
        predictor = QPCardMaskPredictor(model_path=model_path)

    active_bands = [b for b in BANDS if b != "NIR" or mode == "RGB+NIR"]

    results = {}
    for stem in stems:
        print(f"\n-- {stem}")
        try:
            rgb_p      = test_dir / "rgb"                  / f"{stem}.png"
            xml_rgb_p  = test_dir / "xml_rgb"              / f"{stem}.xml"
            mask_rgb_p = test_dir / "masks" / "qpcard" / f"{stem}.png"

            if mode == "RGB+NIR":
                nir_p     = test_dir / "nir"     / f"{stem}.png"
                xml_nir_p = test_dir / "xml_nir" / f"{stem}.xml"
            else:
                nir_p = xml_nir_p = None
            mask_nir_p = None  # masque NIR = masque RGB

            if not mask_rgb_p.exists() and predictor is not None:
                mask = predictor(rgb_p)
                predictor.save(mask, mask_rgb_p)

            res = process_single(
                rgb_p, nir_p,
                xml_rgb_p, xml_nir_p,
                mask_rgb_p, mask_nir_p,
                refl_ref=refl_ref,
                force_zero=force_zero,
                save_dir=save_dir,
                n_patches_rgb=n_patches_rgb,
                mode=mode,
            )
            if res is not None:
                results[stem] = res
                for band in active_bands:
                    arr = res[band]
                    print(f"  {band}: min={arr.min():.4f}  max={arr.max():.4f}  "
                          f"mean={arr.mean():.4f}")
        except Exception as exc:
            warnings.warn(f"[ERREUR] {stem} : {exc}")

    print(f"\nTotal traité : {len(results)} image(s)")
    return results


def process_site_dataset(
    data_root,
    force_zero:    bool = True,
    save_dir             = None,
    metadata_dir          = None,
    n_patches_rgb: int  = 2,
    max_images:    int  = None,
    rgb_dir:       str  = "images_rgb_rect",
    nir_dir:       str  = "images_nir",
    nir_rect_dir:  str  = "images_nir_rect",
    xml_rgb_dir:   str  = "xml_rgb_rect",
    xml_nir_dir:   str  = "xml_nir_rect",
    mask_dir:      str  = "qpcard",
    nir_shift_path: str = "masks/qpcard_nir_shift.json",
    rgb_camera_tag: str = "Camera1",
    nir_camera_tag: str = "Camera3",
    save_plots:    bool = True,
) -> dict:
    """
    Traite un dossier de site (layout images_rgb_rect/ + images_nir/ +
    images_nir_rect/ + xml_rgb_rect/ + xml_nir_rect/ + masks/qpcard/), où RGB
    et NIR sont deux caméras physiques distinctes (Camera1 vs Camera3) avec des
    noms de fichiers différents pour un même tir.

    Toujours en mode RGB+NIR : masks/qpcard/ sert au warp RGB ; pour le NIR, ce
    même masque est décalé à la volée avec (dx, dy) lu dans nir_shift_path (
    produit par processing/decalage.py). La régression ELM est
    calculée sur l'image NIR brute (images_nir/), puis appliquée à l'image NIR
    rectifiée (images_nir_rect/) pour produire la carte finale.

    Si metadata_dir est fourni, sauvegarde pour chaque image traitée un JSON
    (EXIF + résumé des régressions ELM) ; si en plus save_plots=True, génère
    aussi le plot de régression DN→Réflectance et le plot de la QPCard
    redressée (RGB + NIR) avec les fenêtres d'extraction (matplotlib, coûteux
    à l'unité — a désactiver pour les runs de production ne nécessitant pas
    ce diagnostic visuel par image).

    Retourne dict stem RGB → {"_models": ...} (résumé des régressions ELM par
    bande). Les cartes de réflectance pleine résolution ne sont pas conservées
    en mémoire au-delà de chaque image — elles sont déjà écrites sur disque par
    save_reflectance (voir save_dir) au fur et à mesure du traitement.
    """
    data_root = Path(data_root)
    refl_ref  = _load_refl_ref()

    nir_shift = None
    shift_path = data_root / nir_shift_path
    if shift_path.exists():
        with open(shift_path) as f:
            shift_data = json.load(f)
        nir_shift = (shift_data["dx"], shift_data["dy"])
        print(f"[INFO] Décalage NIR chargé : dx={nir_shift[0]}px  dy={nir_shift[1]}px")
    else:
        print(f"[WARN] Pas de fichier de décalage ({shift_path}) — masque RGB non décalé utilisé pour le NIR.")

    stems = find_site_pairs(
        data_root,
        rgb_dir=rgb_dir, nir_dir=nir_dir, nir_rect_dir=nir_rect_dir,
        xml_rgb_dir=xml_rgb_dir, xml_nir_dir=xml_nir_dir,
        mask_dir=mask_dir,
        rgb_camera_tag=rgb_camera_tag, nir_camera_tag=nir_camera_tag,
    )
    if max_images is not None:
        stems = stems[:max_images]

    if not stems:
        print(f"[WARN] Aucune paire RGB/NIR complète trouvée dans {data_root}")
        print(f"       Requis : {rgb_dir}/ ({rgb_camera_tag}), {nir_dir}/ + {nir_rect_dir}/ ({nir_camera_tag}), "
              f"{xml_rgb_dir}/, {xml_nir_dir}/, masks/{mask_dir}/")
        return {}

    print(f"[OK] {len(stems)} paire(s) RGB/NIR trouvée(s)")

    site_metadata = {}
    results = {}
    for stem in stems:
        print(f"\n-- {stem}")
        try:
            nir_stem = stem.replace(rgb_camera_tag, nir_camera_tag)

            rgb_p      = data_root / rgb_dir          / f"{stem}.png"
            nir_p      = data_root / nir_dir           / f"{nir_stem}.png"
            nir_rect_p = data_root / nir_rect_dir      / f"{nir_stem}.png"
            xml_rgb_p  = data_root / xml_rgb_dir       / f"{stem}.xml"
            xml_nir_p  = data_root / xml_nir_dir       / f"{nir_stem}.xml"
            mask_rgb_p = data_root / "masks" / mask_dir / f"{stem}.png"

            res = process_single(
                rgb_p, nir_p,
                xml_rgb_p, xml_nir_p,
                mask_rgb_p,
                nir_shift=nir_shift,
                nir_output_path=nir_rect_p,
                refl_ref=refl_ref,
                force_zero=force_zero,
                save_dir=save_dir,
                metadata_dir=metadata_dir,
                garbage_dir=data_root / "garbage",
                n_patches_rgb=n_patches_rgb,
                mode="RGB+NIR",
                save_plots=save_plots,
                save_individual_json=False,
                metadata_collector=site_metadata,
            )
            if res is not None:
                for band in BANDS:
                    arr = res[band]
                    print(f"  {band}: min={arr.min():.4f}  max={arr.max():.4f}  "
                          f"mean={arr.mean():.4f}")
                # On ne garde que les modeles ELM (deja ecrits sur disque par
                # save_reflectance) : conserver les cartes pleine resolution de
                # toutes les images du site en RAM jusqu'a la fin ferait gonfler
                # la memoire pour rien, puisque save_metrics_csv n'utilise que ca.
                results[stem] = {"_models": res["_models"]}
        except Exception as exc:
            warnings.warn(f"[ERREUR] {stem} : {exc}")

    if metadata_dir is not None and site_metadata:
        save_site_metadata(site_metadata, metadata_dir)

    print(f"\nTotal traité : {len(results)} image(s)")
    return results

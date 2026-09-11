"""
Redressement perspective de la QPCard et extraction des DN médians par patch/bande.
"""
import numpy as np
import cv2

from config import (
    QPCARD_W, QPCARD_H,
    PATCH_CENTERS, PATCH_ORDER,
    HALF,
    SAT_HIGH, SAT_LOW,
)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 coins : TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_template(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Redresse `img` dans l'espace template QPCard (QPCARD_W × QPCARD_H px)
    en utilisant le masque binaire pour localiser la cible.

    Lève ValueError si le masque est vide.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Masque vide — aucune QPCard détectée")

    cnt            = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)

    if rw < rh:
        rw, rh, angle = rh, rw, angle + 90

    src_pts = _order_points(
        cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.float32)
    )
    dst_pts = np.array(
        [[0, 0], [QPCARD_W, 0], [QPCARD_W, QPCARD_H], [0, QPCARD_H]],
        dtype=np.float32,
    )

    M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (QPCARD_W, QPCARD_H))
    return warped


def extract_patch_dn(
    warped: np.ndarray,
    center: tuple,
    half: int = HALF,
) -> tuple:
    """
    Extrait le DN médian de tous les pixels d'une fenêtre 2*half × 2*half.

    Retourne (médiane, MAD, n_pixels).
    """
    px, py  = center
    crop    = warped[py - half:py + half, px - half:px + half].astype(float)
    flat    = crop.ravel()
    med     = np.median(flat)
    mad     = np.median(np.abs(flat - med))
    return med, mad, flat.size


def extract_all_patches(
    warped_rgb: np.ndarray,
    warped_nir: np.ndarray = None,
) -> tuple:
    """
    Extrait les DN médians des 3 patchs × 4 bandes depuis les images redressées.

    warped_nir peut être None (mode RGB uniquement) ; dans ce cas la bande NIR est omise.

    Retourne (dn, violations) :
        dn         : dict band → [dn_noir, dn_gris, dn_blanc]  (np.nan si invalide)
        violations : list de messages décrivant chaque contrainte violée
    """
    dn = {"R": [], "G": [], "B": []}
    if warped_nir is not None:
        dn["NIR"] = []

    for patch_name in PATCH_ORDER:
        cx, cy = PATCH_CENTERS[patch_name]

        for band_key, ch_idx in [("R", 2), ("G", 1), ("B", 0)]:
            med, _, _ = extract_patch_dn(warped_rgb[:, :, ch_idx], (cx, cy))
            dn[band_key].append(med)

        if warped_nir is not None:
            med_nir, _, _ = extract_patch_dn(warped_nir, (cx, cy))
            if med_nir >= SAT_HIGH:
                dn["NIR"].append(np.nan)
            else:
                dn["NIR"].append(med_nir)

    return dn, []

import cv2
import numpy as np


def get_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Calcule le centroïde d'un masque binaire."""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        raise ValueError("Masque vide — aucun pixel blanc détecté")
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def shift_mask(mask_arr: np.ndarray, shift_x: int, shift_y: int = 0) -> np.ndarray:
    """Décale mask_arr de (shift_x, shift_y) pixels."""
    shifted = np.zeros_like(mask_arr)

    if shift_x > 0:
        shifted[:, shift_x:] = mask_arr[:, :-shift_x]
    elif shift_x < 0:
        shifted[:, :shift_x] = mask_arr[:, -shift_x:]
    else:
        shifted = mask_arr.copy()

    tmp = shifted.copy()
    shifted = np.zeros_like(tmp)

    if shift_y > 0:
        shifted[shift_y:, :] = tmp[:-shift_y, :]
    elif shift_y < 0:
        shifted[:shift_y, :] = tmp[-shift_y:, :]
    else:
        shifted = tmp.copy()

    return shifted


class MaskShift:
    """Calcule le décalage entre deux masques et recale le masque RGB sur le masque NIR."""

    def __init__(self, mask_rgb: np.ndarray, mask_nir: np.ndarray):
        self.mask_rgb = mask_rgb
        self.mask_nir = mask_nir

        self.cx_rgb, self.cy_rgb = get_centroid(mask_rgb)
        self.cx_nir, self.cy_nir = get_centroid(mask_nir)

        self.dx = self.cx_nir - self.cx_rgb
        self.dy = self.cy_nir - self.cy_rgb

    def get_shift(self) -> tuple[int, int]:
        """Retourne le décalage entier (dx, dy)."""
        return int(round(self.dx)), int(round(self.dy))

    def shift_mask(self) -> np.ndarray:
        """Retourne le masque RGB recalé sur le masque NIR."""
        shift_x, shift_y = self.get_shift()
        return shift_mask(self.mask_rgb, shift_x, shift_y)

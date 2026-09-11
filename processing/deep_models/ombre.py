from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from ._base import BaseMaskPredictor, DEFAULT_DATA_ROOT, DEFAULT_MODELS_DIR, run_dataset


DEFAULT_MODEL  = DEFAULT_MODELS_DIR / "ombres_final.pt"
DEFAULT_DEVICE = "cuda"
DEFAULT_STRIDE = 32


class OmbreMaskPredictor(BaseMaskPredictor):
    pass


@lru_cache(maxsize=1)
def _get_predictor(model_path: Path, tile_size: int = 512, tile_stride: int = 256) -> OmbreMaskPredictor:
    return OmbreMaskPredictor(model_path=model_path, tile_size=tile_size, tile_stride=tile_stride)


def _apply_vegetation_mask(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    veg_mask_path = image_path.parent.parent / "masks" / "vegetation" / image_path.name
    veg_mask = cv2.imread(str(veg_mask_path), cv2.IMREAD_GRAYSCALE)
    if veg_mask is None:
        raise FileNotFoundError(f"Cannot read vegetation mask: {veg_mask_path}")

    rgb[veg_mask == 0] = 255  # background → white before shadow inference
    return rgb


def process_dataset(data_root: Path = DEFAULT_DATA_ROOT, model_path: Path = DEFAULT_MODEL, max_images: int = None, tile_size: int = 512, tile_stride: int = 256) -> None:
    predictor = _get_predictor(model_path, tile_size, tile_stride)
    run_dataset(predictor, data_root, mask_subdir="ombres", get_input=_apply_vegetation_mask, max_images=max_images)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ombre mask predictor — génère un masque binaire depuis une image RGB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image",          type=Path, help="Chemin de l'image RGB d'entrée")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Chemin de sortie du masque (.png). "
                             "Par défaut : même dossier que l'image, suffixe _mask.png")
    parser.add_argument("--model",  "-m", type=Path, default=DEFAULT_MODEL,
                        help="Chemin du modèle TorchScript (.pt)")
    parser.add_argument("--device", "-d", type=str,  default=DEFAULT_DEVICE,
                        choices=["cuda", "cpu"], help="Device d'inférence")
    parser.add_argument("--stride", "-s", type=int,  default=DEFAULT_STRIDE,
                        help="Stride de padding (doit correspondre au modèle)")

    args = parser.parse_args()
    output = args.output or args.image.with_name(args.image.stem + "_mask.png")

    predictor = OmbreMaskPredictor(
        model_path=args.model,
        device=args.device,
        stride=args.stride,
    )
    mask = predictor(args.image)
    saved = predictor.save(mask, output)
    print(f"[SAVED] {saved}")

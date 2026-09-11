from functools import lru_cache
from pathlib import Path

from ._base import BaseMaskPredictor, DEFAULT_DATA_ROOT, DEFAULT_MODELS_DIR, run_dataset
from ..qpcard_postprocessing import perfect_mask_bottom_polygon


DEFAULT_MODEL  = DEFAULT_MODELS_DIR / "qpcard_final.pt"
DEFAULT_DEVICE = "cuda"
DEFAULT_STRIDE = 32


class QPCardMaskPredictor(BaseMaskPredictor):
    pass


@lru_cache(maxsize=1)
def _get_predictor(model_path: Path, tile_size: int = 512, tile_stride: int = 256) -> QPCardMaskPredictor:
    return QPCardMaskPredictor(model_path=model_path, tile_size=tile_size, tile_stride=tile_stride)


def _postprocess(mask_path: Path, bottom_fraction: float) -> None:
    try:
        perfect_mask_bottom_polygon(str(mask_path), str(mask_path), bottom_fraction=bottom_fraction)
    except ValueError as e:
        print(f"[POST]  {mask_path.name} — {e} (mask inchangé)")


def process_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    model_path: Path = DEFAULT_MODEL,
    max_images: int = None,
    tile_size: int = 512,
    tile_stride: int = 256,
) -> None:
    predictor = _get_predictor(model_path, tile_size, tile_stride)
    run_dataset(
        predictor,
        data_root,
        mask_subdir="qpcard",
        max_images=max_images,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="QPCard mask predictor — génère un masque binaire depuis une image RGB",
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
    parser.add_argument("--bottom-fraction", type=float, default=0.25,
                        help="Fraction du bas de l'image pour le post-traitement")

    args = parser.parse_args()
    output = args.output or args.image.with_name(args.image.stem + "_mask.png")

    predictor = QPCardMaskPredictor(
        model_path=args.model,
        device=args.device,
        stride=args.stride,
    )
    mask = predictor(args.image)
    saved = predictor.save(mask, output)
    _postprocess(saved, args.bottom_fraction)
    print(f"[SAVED] {saved}")

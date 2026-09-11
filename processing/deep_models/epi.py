from functools import lru_cache
from pathlib import Path
from typing import Union

from ._base import BaseMaskPredictor, DEFAULT_DATA_ROOT, DEFAULT_MODELS_DIR, run_dataset


DEFAULT_MODEL  = DEFAULT_MODELS_DIR / "MAnet_pvt_v2_b2_ble_epi_v2_2026_04_23.torchscript"
DEFAULT_DEVICE = "cuda"
DEFAULT_STRIDE = 32


class EpiMaskPredictor(BaseMaskPredictor):
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        stride: int = DEFAULT_STRIDE,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            stride=stride,
            padding_mode="reflect",
            use_amp=True,
        )


@lru_cache(maxsize=1)
def _get_predictor(model_path: Path) -> EpiMaskPredictor:
    return EpiMaskPredictor(model_path=model_path)


def process_dataset(data_root: Path = DEFAULT_DATA_ROOT, model_path: Path = DEFAULT_MODEL, max_images: int = None) -> None:
    predictor = _get_predictor(model_path)
    run_dataset(predictor, data_root, mask_subdir="epi", max_images=max_images)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Epi mask predictor — génère un masque binaire depuis une image RGB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image",          type=Path, help="Chemin de l'image RGB d'entrée")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Chemin de sortie du masque (.png). "
                             "Par défaut : même dossier que l'image, suffixe _mask.png")
    parser.add_argument("--model",  "-m", type=Path, default=DEFAULT_MODEL,
                        help="Chemin du modèle TorchScript (.torchscript)")
    parser.add_argument("--device", "-d", type=str,  default=DEFAULT_DEVICE,
                        choices=["cuda", "cpu"], help="Device d'inférence")
    parser.add_argument("--stride", "-s", type=int,  default=DEFAULT_STRIDE,
                        help="Stride de padding (doit correspondre au modèle)")

    args = parser.parse_args()
    output = args.output or args.image.with_name(args.image.stem + "_mask.png")

    predictor = EpiMaskPredictor(
        model_path=args.model,
        device=args.device,
        stride=args.stride,
    )
    mask = predictor(args.image)
    saved = predictor.save(mask, output)
    print(f"[SAVED] {saved}")

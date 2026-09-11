from pathlib import Path
from typing import Union
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_STRIDE = 32


class QPCardMaskPredictor:
    """
    Prédit le masque binaire de la QPCard depuis une image RGB avec un modèle TorchScript.

    Utilisation :
        predictor = QPCardMaskPredictor("models/qpcard_final.pt")
        mask = predictor(rgb_path)                   # np.ndarray uint8, 0/255
        predictor.save(mask, "out/mask.png")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        stride: int = _STRIDE,
    ):
        self.stride = stride
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        self.mean = _MEAN.to(self.device)
        self.std  = _STD.to(self.device)

        print(f"[INFO] Device : {self.device}")
        t0 = time.perf_counter()
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        print(f"[INFO] Modèle chargé en {time.perf_counter() - t0:.3f}s")

    def _preprocess(self, image_path: Union[str, Path]):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .to(self.device, dtype=torch.float32)
            .div_(255.0)
            .sub_(self.mean)
            .div_(self.std)
            .unsqueeze_(0)
        )

        ph = (self.stride - orig_h % self.stride) % self.stride
        pw = (self.stride - orig_w % self.stride) % self.stride
        if ph or pw:
            tensor = F.pad(tensor, (0, pw, 0, ph))

        return tensor, orig_h, orig_w

    def predict(self, image_path: Union[str, Path]) -> np.ndarray:
        tensor, orig_h, orig_w = self._preprocess(image_path)

        with torch.inference_mode():
            logits = self.model(tensor)

        mask = (
            logits.squeeze(0)
            .argmax(dim=0)[:orig_h, :orig_w]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return np.where(mask, 255, 0).astype(np.uint8)

    def save(self, mask: np.ndarray, output_path: Union[str, Path]) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), mask):
            raise IOError(f"Impossible d'écrire : {output_path}")
        return output_path

    def __call__(self, image_path: Union[str, Path]) -> np.ndarray:
        t0 = time.perf_counter()
        print(f"[INFO] Prédiction masque : {Path(image_path).name}")
        mask = self.predict(image_path)
        print(f"[DONE] {time.perf_counter() - t0:.3f}s  "
              f"(foreground {mask.mean() / 255 * 100:.1f}%)")
        return mask

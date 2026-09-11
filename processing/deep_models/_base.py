import math
from pathlib import Path
from typing import Callable, Optional, Union
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DEFAULT_DATA_ROOT  = Path.cwd() / "data"
DEFAULT_MODELS_DIR = Path.cwd() / "models"


class BaseMaskPredictor:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        stride: int = 32,
        padding_mode: str = "constant",
        use_amp: bool = False,
        tile_size: Optional[int] = None,
        tile_stride: int = 256,
        tile_batch_size: int = 4,
    ):
        self.model_path      = Path(model_path)
        self.stride          = stride
        self.padding_mode    = padding_mode
        self.use_amp         = use_amp
        self.tile_size       = tile_size
        self.tile_stride     = tile_stride
        self.tile_batch_size = tile_batch_size
        self.device          = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        if self.device == "cuda":
            # cuDNN v8 returns CUDNN_STATUS_NOT_SUPPORTED for depthwise convs with
            # large group counts (PVT-v2 uses groups=N where N can be 512+), then
            # crashes the CUDA context. PyTorch's native CUDA kernels handle these
            # configs correctly. benchmark=False alone does not prevent the failure.
            torch.backends.cudnn.enabled = False

        self.mean = _MEAN.to(self.device)
        self.std  = _STD.to(self.device)

        print(f"[INFO] Device      : {self.device}")
        print(f"[INFO] Model       : {self.model_path}")
        if self.tile_size:
            print(f"[INFO] Mode        : tiling {self.tile_size}px / stride {self.tile_stride}px / batch {self.tile_batch_size}")

        t0 = time.perf_counter()
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        print(f"[INFO] Model loaded in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Full-image path (no tiling)
    # ------------------------------------------------------------------

    def preprocess(self, image: Union[str, Path, np.ndarray]):
        if isinstance(image, np.ndarray):
            rgb = image
        else:
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
            tensor = F.pad(tensor, (0, pw, 0, ph), mode=self.padding_mode)

        return tensor, orig_h, orig_w

    def predict(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if self.tile_size is not None:
            return self._predict_tiled(image)

        tensor, orig_h, orig_w = self.preprocess(image)

        with torch.inference_mode():
            if self.use_amp and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)

        mask = (
            logits.squeeze(0)
            .argmax(dim=0)[:orig_h, :orig_w]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return np.where(mask, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Tiling path
    # ------------------------------------------------------------------

    def _predict_tiled(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        # 1. Load
        if isinstance(image, np.ndarray):
            rgb = image
        else:
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = rgb.shape[:2]
        tile   = self.tile_size
        stride = self.tile_stride

        # 2. Compute grid & padding so every pixel is covered
        def _n_tiles(dim: int) -> int:
            return 1 if dim <= tile else math.ceil((dim - tile) / stride) + 1

        n_h = _n_tiles(orig_h)
        n_w = _n_tiles(orig_w)
        pad_h = (n_h - 1) * stride + tile - orig_h
        pad_w = (n_w - 1) * stride + tile - orig_w

        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        padded_h, padded_w = rgb.shape[:2]

        # Accumulation buffers (float64 for numerical stability)
        full_prob = np.zeros((padded_h, padded_w), dtype=np.float64)
        count_map = np.zeros((padded_h, padded_w), dtype=np.float64)

        # 3. Extract all patches
        patches, positions = [], []
        for row in range(n_h):
            for col in range(n_w):
                y, x = row * stride, col * stride
                patches.append(rgb[y:y + tile, x:x + tile])
                positions.append((y, x))

        # 4. Batch inference
        for start in range(0, len(patches), self.tile_batch_size):
            batch_patches = patches[start:start + self.tile_batch_size]
            batch_pos     = positions[start:start + self.tile_batch_size]

            tensors = [
                torch.from_numpy(p)
                .permute(2, 0, 1)
                .to(self.device, dtype=torch.float32)
                .div_(255.0)
                .sub_(self.mean)
                .div_(self.std)
                for p in batch_patches
            ]
            batch_t = torch.stack(tensors)  # (B, 3, tile, tile)

            with torch.inference_mode():
                if self.use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch_t)  # (B, C, tile, tile)
                else:
                    logits = self.model(batch_t)

            # Probability of positive class
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits[:, 0])          # (B, tile, tile)
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]   # (B, tile, tile)

            probs_np = probs.cpu().numpy()

            for i, (y, x) in enumerate(batch_pos):
                full_prob[y:y + tile, x:x + tile] += probs_np[i]
                count_map[y:y + tile, x:x + tile] += 1.0

        # 5. Average → binarize → crop to original size
        prob_map = full_prob / count_map
        binary   = (prob_map[:orig_h, :orig_w] >= 0.5).astype(np.uint8)
        return np.where(binary, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------

    def save(self, mask: np.ndarray, output_path: Union[str, Path]) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), mask):
            raise IOError(f"Cannot write: {output_path}")
        return output_path

    def __call__(self, image_path: Union[str, Path]) -> np.ndarray:
        t0 = time.perf_counter()
        print(f"[INFO] Image : {image_path}")
        mask = self.predict(image_path)
        print(f"[INFO] Foreground: {mask.mean() / 255 * 100:.2f}%")
        print(f"[DONE] Time     : {time.perf_counter() - t0:.3f}s")
        return mask


def iter_rgb_images(data_root: Path):
    """Yields all RGB images under data_root, handling both flat and nested layouts."""
    rgb_dir = data_root / "images_rgb_rect"
    if rgb_dir.is_dir():
        yield from (p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        return

    for directory in data_root.iterdir():
        if not directory.is_dir():
            continue
        rgb_dir = directory / "images_rgb_rect"
        if not rgb_dir.is_dir():
            continue
        for image_path in rgb_dir.iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path


def run_dataset(
    predictor: BaseMaskPredictor,
    data_root: Path,
    mask_subdir: str,
    get_input: Optional[Callable[[Path], Union[str, Path, np.ndarray]]] = None,
    post_process: Optional[Callable[[Path], None]] = None,
    max_images: Optional[int] = None,
) -> None:
    """Process all images in data_root, saving masks to masks/<mask_subdir>/."""
    images = list(iter_rgb_images(data_root))
    if max_images is not None:
        images = images[:max_images]
    print(f"[INFO] Found {len(images)} images\n")

    total = skipped = failed = 0
    for image_path in images:
        total += 1
        mask_save_path = image_path.parent.parent / "masks" / mask_subdir / image_path.name

        if mask_save_path.exists():
            print(f"[SKIP] ({total}/{len(images)}) {image_path.name}")
            skipped += 1
            continue

        try:
            print(f"[RUN]  ({total}/{len(images)}) {image_path.name}")
            inp = get_input(image_path) if get_input else image_path
            mask = predictor.predict(inp)
            predictor.save(mask, mask_save_path)
            if post_process is not None:
                post_process(mask_save_path)
            print(f"[OK]   Saved → {mask_save_path}")
        except RuntimeError as e:
            print(f"[ERR]  {image_path.name} — {e}")
            failed += 1
            # A CUDA launch failure poisons the entire device context; re-raising
            # lets Airflow mark the task as failed rather than silently skipping
            # all remaining images with guaranteed GPU errors.
            if "CUDA error" in str(e):
                raise
        except Exception as e:
            print(f"[ERR]  {image_path.name} — {e}")
            failed += 1

    ok = total - skipped - failed
    print(f"\n[DONE] Total: {total} | OK: {ok} | Skipped: {skipped} | Failed: {failed}")

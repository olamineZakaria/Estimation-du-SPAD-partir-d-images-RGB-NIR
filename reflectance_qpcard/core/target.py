import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import yaml


class Patch:
    CHANNELS = ("R", "G", "B", "NIR")

    def __init__(self, center=None, size=None, name=""):
        self._center = tuple(center) if center else None
        self._size   = tuple(size)   if size   else None
        self._name   = name

        self._reflectance = {ch: 0 for ch in self.CHANNELS}
        self._luminance   = {ch: 0 for ch in self.CHANNELS}

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    def set_geometry(self, center, size):
        self._center = tuple(center)
        self._size   = tuple(size)

    def set_reflectance(self, ch, val):
        self._reflectance[ch] = val

    def set_luminance(self, ch, val):
        self._luminance[ch] = val

    @property
    def reflectance(self):
        return self._reflectance

    @property
    def luminance(self):
        return self._luminance

    def __repr__(self):
        return f"Patch(name={self._name!r}, center={self._center})"


class QPCard:
    def __init__(self, width, height, center_dark, center_gray, center_white, patch_size):
        self._width   = width
        self._height  = height
        self._corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

        self._patches = {
            1: Patch(center_dark,  patch_size, "dark"),
            2: Patch(center_gray,  patch_size, "gray"),
            3: Patch(center_white, patch_size, "white"),
        }

        self._force_zero       = False
        self.calibration_model = {}

    @property
    def patches(self):
        return self._patches

    @property
    def patch_centers(self):
        return [self._patches[k].center for k in self._patches]

    def load_reflectance_from_yaml(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        for patch_data in data["qpcard"]["patches"]:
            pid  = patch_data["patch_id"]
            refl = patch_data["reflectance"]
            for ch, val in refl.items():
                self._patches[pid].set_reflectance(ch, val)

    def reset_luminances(self):
        for patch in self._patches.values():
            for ch in Patch.CHANNELS:
                patch.set_luminance(ch, 0)

    def extract_patch_values(self, camera_img, projected_points, exposure=1.0, iso=1.0):
        """
        Extrait les luminances des patchs depuis l'image (déjà redressée dans l'espace template).

        camera_img        : np.ndarray BGR (H,W,3) ou grayscale (H,W)
        projected_points  : array (N,2) des centres des patchs dans l'espace image
        exposure, iso     : facteurs de normalisation
        """
        img  = camera_img
        mode = "NIR" if len(img.shape) == 2 else "RGB"

        max_value = 254 if img.dtype == np.uint8 else 4094

        patches = list(self._patches.values())

        for i, (x, y) in enumerate(projected_points):
            if i >= len(patches):
                break
            patch = patches[i]
            if patch.size is None:
                continue

            w, h = map(int, patch.size)
            x, y = int(x), int(y)
            roi  = img[max(0, y - h // 2):y + h // 2, max(0, x - w // 2):x + w // 2]

            if roi.size == 0:
                continue

            if mode == "RGB":
                b, g, r = cv.split(roi)
                R = float(np.median(r))
                G = float(np.median(g))
                B = float(np.median(b))

                if R < max_value and G < max_value and B < max_value:
                    patch.set_luminance("R", R / (exposure * iso))
                    patch.set_luminance("G", G / (exposure * iso))
                    patch.set_luminance("B", B / (exposure * iso))
            else:
                NIR = float(np.median(roi))
                if NIR < max_value:
                    patch.set_luminance("NIR", NIR / (exposure * iso))

    def fit_calibration_model(self, force_zero=True):
        self._force_zero       = force_zero
        self.calibration_model = {}

        for ch in Patch.CHANNELS:
            X, Y = [], []
            for patch in self._patches.values():
                lum  = patch.luminance.get(ch, 0)
                refl = patch.reflectance.get(ch)
                if lum <= 0 or refl is None:
                    continue
                X.append(lum)
                Y.append(refl)

            if len(X) < 2:
                continue

            X = np.array(X)
            Y = np.array(Y)

            if force_zero:
                a = np.sum(X * Y) / np.sum(X * X)
                b = 0.0
            else:
                A    = np.vstack([X, np.ones(len(X))]).T
                a, b = np.linalg.lstsq(A, Y, rcond=None)[0]

            self.calibration_model[ch] = (a, float(b))

        return self.calibration_model

    def calibrate_image(self, img):
        if not self.calibration_model:
            raise ValueError("Modèle non calculé — appeler fit_calibration_model() d'abord")

        out = img.astype(np.float32).copy()

        if len(img.shape) == 3:
            for ch, idx in [("B", 0), ("G", 1), ("R", 2)]:
                if ch in self.calibration_model:
                    a, b = self.calibration_model[ch]
                    out[:, :, idx] = a * out[:, :, idx] + b
        else:
            if "NIR" in self.calibration_model:
                a, b = self.calibration_model["NIR"]
                out  = a * out + b

        return out

    def plot_calibration(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, ch in enumerate(Patch.CHANNELS):
            ax = axes[idx]
            X, Y = [], []

            for patch in self._patches.values():
                lum  = patch.luminance.get(ch, 0)
                refl = patch.reflectance.get(ch)
                if lum <= 0 or refl is None:
                    continue
                X.append(lum)
                Y.append(refl)

            if len(X) < 2 or ch not in self.calibration_model:
                ax.set_title(f"{ch} (données insuffisantes)")
                ax.axis("off")
                continue

            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            a, b = self.calibration_model[ch]

            x_line = np.linspace(0 if self._force_zero else X.min(), X.max(), 100)
            Y_pred = a * X + b
            r2     = 1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - Y.mean()) ** 2)

            ax.scatter(X, Y)
            ax.plot(x_line, a * x_line + b, lw=2)
            if self._force_zero:
                ax.scatter([0], [0], marker="x", color="gray")

            ax.set_title(f"Canal {ch}")
            ax.set_xlabel("Luminance")
            ax.set_ylabel("Réflectance")
            ax.grid(True, alpha=0.3)
            ax.text(0.05, 0.95, f"y={a:.5f}x+{b:.5f}\nR²={r2:.4f}",
                    transform=ax.transAxes, va="top", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

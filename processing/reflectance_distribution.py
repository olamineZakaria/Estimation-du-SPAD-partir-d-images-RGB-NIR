"""
Distribution des Bandes de Reflectance
=======================================
Pour chaque image (RGB+NIR calibree en reflectance) d'un site, trace la
distribution des valeurs de pixel des 4 bandes (B, G, R, NIR) avec la
moyenne, la mediane et le mode. Les pixels a 0 de la bande NIR (hors zone
valide apres warp/decalage) sont exclus, comme dans processing/spad_pixel_stats.py.

Sauvegarde dans data/<site>/distribution/<stem>.png (meme nom que l'image
source, une bande venant de data/<site>/reflectance/{B,G,R,NIR}/<stem>.tif).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

from processing.spad_pixel_stats import REFLECTANCE_FULL_SCALE, load_reflectance_bands

DEFAULT_DATA_ROOT = Path.cwd() / "data"

# nom_affiche -> (band_key, couleur, exclure_zeros)
BAND_PLOTS = {
    "Bleu (B)":  ("B",   "blue",       False),
    "Vert (G)":  ("G",   "green",      False),
    "Rouge (R)": ("R",   "red",        False),
    "NIR":       ("NIR", "darkorange", True),
}


def plot_band_distributions(bands: dict[str, np.ndarray], title: str, save_path: Path) -> bool:
    """bands: valeurs de pixel par bande (image complete ou deja restreinte a
    un masque). Retourne False (sans rien sauvegarder) si aucune bande n'a de
    pixel a tracer, par ex. un scenario dont le masque est vide."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    has_data = False

    for ax, (name, (band_key, color, remove_zeros)) in zip(axes, BAND_PLOTS.items()):
        arr = bands[band_key].flatten()
        if remove_zeros:
            arr = arr[arr > 0]
        # valeurs converties en pourcentage de reflectance (DN / 2**16 * 100)
        arr = arr.astype(np.float64) / REFLECTANCE_FULL_SCALE * 100

        plot_title = f"Distribution - {name}"
        if band_key == "NIR":
            plot_title += " (0 exclus)"

        if arr.size == 0:
            ax.set_title(f"{plot_title} (aucun pixel)", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue
        has_data = True

        mean_v = arr.mean()
        median_v = np.median(arr)

        if arr.size > 1 and np.ptp(arr) > 0:
            kde = gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 512)
            density = kde(xs)
            ax.plot(xs, density, color=color, linewidth=1.5)
            ax.fill_between(xs, density, color=color, alpha=0.4)
            mode_v = xs[np.argmax(density)]
        else:
            ax.axvline(arr[0], color=color, linewidth=2)
            mode_v = arr[0]

        ax.axvline(mean_v, color="black", linestyle="--", linewidth=1.5, label=f"Moyenne = {mean_v:.2f}%")
        ax.axvline(median_v, color="purple", linestyle="-.", linewidth=1.5, label=f"Mediane = {median_v:.2f}%")
        ax.axvline(mode_v, color="dimgray", linestyle=":", linewidth=2, label=f"Mode = {mode_v:.2f}%")

        ax.set_title(plot_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Reflectance (%)")
        ax.set_ylabel("Densite")
        ax.legend(fontsize=9)

    if not has_data:
        plt.close(fig)
        return False

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def process_one_image(site_root: Path, stem: str, save_dir: Path) -> None:
    bands = load_reflectance_bands(site_root, stem)
    plot_band_distributions(bands, stem, save_dir / f"{stem}.png")


def process_site(site_root: Path, max_images: int | None = None) -> None:
    site_root = Path(site_root)
    stems = sorted(p.stem for p in (site_root / "reflectance" / "B").glob("*.tif"))
    if max_images is not None:
        stems = stems[:max_images]

    save_dir = site_root / "distribution"
    total = len(stems)
    skipped = 0
    for i, stem in enumerate(stems, start=1):
        try:
            process_one_image(site_root, stem, save_dir)
            print(f"[OK]   ({i}/{total}) {site_root.name} — {stem}")
        except FileNotFoundError as e:
            skipped += 1
            print(f"[SKIP] ({i}/{total}) {site_root.name} — {e}")

    print(f"[DONE] {total - skipped}/{total} images traitees, {skipped} ignorees — {site_root.name}")

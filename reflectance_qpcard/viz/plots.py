"""
Visualisations : QPCard redressée, courbes de régression ELM, cartes de réflectance.
"""
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    PATCH_CENTERS, PATCH_ORDER, HALF,
    BAND_COLORS, PATCH_COLORS, BANDS,
)


def plot_warped_qpcard(
    warped_rgb: np.ndarray,
    warped_nir: np.ndarray,
    stem:       str,
    save_dir    = None,
):
    """
    Affiche (et sauvegarde) la QPCard redressée en RGB (et NIR si disponible)
    avec les fenêtres d'extraction des patchs.
    """
    panels = [(cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2RGB), "RGB")]
    if warped_nir is not None:
        panels.append((warped_nir, "NIR"))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4), squeeze=False)
    fig.suptitle(f"QPCard redressée — {stem}", fontsize=11, fontweight="bold")

    for ax, (img, title) in zip(axes[0], panels):
        cmap = None if title == "RGB" else "gray"
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Bande {title}", fontweight="bold")

        color = "lime" if title == "NIR" else "red"
        for name, (cx, cy) in PATCH_CENTERS.items():
            rect = plt.Rectangle(
                (cx - HALF, cy - HALF), 2 * HALF, 2 * HALF,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(cx, cy - HALF - 5, name.upper(),
                    ha="center", va="bottom", fontsize=8,
                    color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_qpcard_warped.png")


def plot_regression(
    fit_results: dict,
    stem:        str,
    save_dir     = None,
):
    """
    Affiche (et sauvegarde) les courbes de régression ELM DN → Réflectance
    pour les bandes disponibles.

    fit_results : dict band → résultat fit_elm (ou None)
    """
    available = [b for b in BANDS if fit_results.get(b) is not None]
    if not available:
        print(f"[VIZ] Aucune bande disponible pour {stem}")
        return

    ncols = min(len(available), 2)
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"Régression ELM DN → Réflectance\n{stem}", fontsize=12, fontweight="bold")

    axes_flat = axes.flat
    for ax, band in zip(axes_flat, available):
        r   = fit_results[band]
        col = BAND_COLORS[band]
        dn  = r["dn"]
        ref = r["refl"]
        fz  = r

        x_range = np.linspace(0, max(dn) * 1.15, 300)
        ax.plot(x_range, fz["slope"] * x_range + fz["intercept"],
                color=col, lw=2.2,
                label=f"R²={fz['r2']:.4f}  RMSE={fz['rmse']:.5f}")

        for i, (d, rv) in enumerate(zip(dn, ref)):
            pc = list(PATCH_COLORS.values())[i]
            ec = "white" if i == 0 else "black"
            ax.scatter(d, rv, s=120, color=pc, edgecolors=ec, lw=1.5, zorder=6)
            ax.annotate(PATCH_ORDER[i], (d, rv),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=8.5, color="#333")

        for d, rv, p in zip(dn, ref, fz["pred"]):
            ax.plot([d, d], [rv, p], color="gray", lw=1, ls=":", alpha=0.6)

        ax.set_xlim(left=-2)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("DN (Digital Number)", fontsize=10)
        ax.set_ylabel("Réflectance [0,1]", fontsize=10)
        ax.set_title(f"Bande {band}", color=col, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.text(0.97, 0.10,
                f"$a={fz['slope']:.5f}$",
                transform=ax.transAxes, ha="right", fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    for ax in list(axes_flat)[len(available):]:
        ax.set_visible(False)

    legend_patches = [
        mpatches.Patch(fc=pc, ec="black", label=pn.capitalize())
        for pn, pc in PATCH_COLORS.items()
    ]
    fig.legend(handles=legend_patches, title="Patchs QPCard",
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_regression.png")


def plot_reflectance_maps(
    result: dict,
    stem:   str,
    save_dir = None,
):
    """
    Affiche (et sauvegarde) les 4 cartes de réflectance (R, G, B, NIR).
    """
    available = [b for b in BANDS if b in result]
    ncols = min(len(available), 2)
    nrows = (len(available) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle(f"Cartes de réflectance — {stem}", fontsize=12, fontweight="bold")

    axes_flat = axes.flat
    for ax, band in zip(axes_flat, available):
        arr  = result[band]
        col  = BAND_COLORS[band]
        cmap = "RdYlGn" if band != "B" else "Blues"

        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Réflectance {band}", color=col, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in list(axes_flat)[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_reflectance_maps.png")


def _save_or_show(fig, save_dir, filename: str):
    if save_dir is not None:
        path = Path(save_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Sauvegardé : {path}")
    else:
        plt.show()

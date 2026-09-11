from pathlib import Path

import cv2
import numpy as np


def perfect_mask_bottom_polygon(image_path, output_path, bottom_fraction=0.25):
    """
    Concentre le traitement sur les `bottom_fraction` (%) inférieurs de l'image,
    fusionne tous les blobs blancs trouvés dans cette zone en UN SEUL polygone
    (convex hull), et retourne un masque binaire "parfait" de la même taille
    que l'image d'origine.

    Args:
        image_path (str): chemin du mask binaire d'entrée (0/255)
        output_path (str): chemin de sauvegarde du mask nettoyé
        bottom_fraction (float): fraction de la hauteur à considérer (0.25 = 25% du bas)

    Returns:
        np.ndarray: masque binaire final (uint8, 0/255)
        np.ndarray: points du polygone (N,2) dans le repère de l'image complète
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape
    y_start = int(h * (1 - bottom_fraction))

    # On isole la bande du bas
    band = img[y_start:h, :]
    _, binary_band = cv2.threshold(band, 127, 255, cv2.THRESH_BINARY)

    # Contours trouvés uniquement dans la bande
    contours, _ = cv2.findContours(binary_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Aucun objet trouvé dans les {}% inférieurs de l'image".format(int(bottom_fraction*100)))

    # Fusion de tous les points de tous les contours -> un seul polygone convexe
    all_points = np.vstack([c.reshape(-1, 2) for c in contours])
    hull = cv2.convexHull(all_points)

    # Repositionner le polygone dans le repère de l'image complète
    hull_full = hull.copy()
    hull_full[:, 0, 1] += y_start

    # Reconstruction du masque final (taille originale, tout le reste = 0)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out, [hull_full], 0, 255, -1)

    cv2.imwrite(output_path, out)
    return out, hull_full.reshape(-1, 2)


def process_dataset(
    data_root: Path,
    model_path: Path = None,
    max_images: int = None,
    mask_subdir: str = "qpcard",
    bottom_fraction: float = 0.25,
) -> None:
    """Nettoie tous les masks d'un dossier masks/<mask_subdir> avec perfect_mask_bottom_polygon."""
    mask_dir = Path(data_root) / "masks" / mask_subdir
    if not mask_dir.is_dir():
        print(f"[WARN] Pas de dossier de masks {mask_dir} — postprocessing ignoré.")
        return

    masks = sorted(mask_dir.glob("*.png"))
    if max_images is not None:
        masks = masks[:max_images]
    print(f"[INFO] {len(masks)} masks à post-traiter dans {mask_dir}\n")

    total = failed = 0
    for mask_path in masks:
        total += 1
        try:
            perfect_mask_bottom_polygon(str(mask_path), str(mask_path), bottom_fraction=bottom_fraction)
            print(f"[OK]   ({total}/{len(masks)}) {mask_path.name}")
        except ValueError as e:
            print(f"[SKIP] ({total}/{len(masks)}) {mask_path.name} — {e} (mask inchangé)")
        except Exception as e:
            print(f"[ERR]  ({total}/{len(masks)}) {mask_path.name} — {e}")
            failed += 1

    print(f"\n[DONE] Total: {total} | Failed: {failed}")


if __name__ == "__main__":
    mask, polygon = perfect_mask_bottom_polygon(
        "Plot1_13_8617WW0010049_Tricam1Camera1_1.png",
        "/mnt/user-data/outputs/Plot1_13_8617WW0010049_Tricam1Camera1_1_mask_final.png",
        bottom_fraction=0.25,
    )
    print("Polygone ({} sommets):".format(len(polygon)))
    print(polygon)

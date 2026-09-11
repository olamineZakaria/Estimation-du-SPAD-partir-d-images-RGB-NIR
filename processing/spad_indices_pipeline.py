"""Pipeline SPAD et indices spectraux, autonome et sans Airflow.

Pour chaque ligne de ``data/spad_dataset_merged.csv``, ce script :

1. charge les bandes R, G, B et NIR de l'image ;
2. charge le masque combine ;
3. calcule cinq indices sur les pixels du masque ;
4. calcule moyenne, mediane et mode des indices et des quatre bandes ;
5. calcule les intensites de la mire depuis les pentes de calibration ;
6. ajoute les mesures SPAD et les metadonnees XML ;
7. ecrit un CSV par site et un CSV global.

Aucun fichier ``.npy`` et aucun graphique ne sont produits.
"""

from __future__ import annotations

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CSV_ROOT = PROJECT_ROOT / "csv"
SPAD_CSV = DATA_ROOT / "spad_dataset_merged.csv"

REFLECTANCE_FULL_SCALE = 2**16
DENOMINATOR_EPSILON = 1e-2
MODE_SAMPLE_LIMIT = 50_000

INDEX_NAMES = (
    "rb_index",
    "br_bnir_index",
    "ci_green",
    "nir_b_gb_index",
    "nir_gb_ratio_index",
)

BAND_NAMES = ("R", "G", "B", "NIR")

PATCH_REFLECTANCE = {
    "noir": {"R": 0.071405, "G": 0.072839, "B": 0.072962, "NIR": 0.065163},
    "gris": {"R": 0.152181, "G": 0.156171, "B": 0.155624, "NIR": 0.132566},
    "blanc": {"R": 0.815291, "G": 0.806183, "B": 0.787966, "NIR": 0.872626},
}

METADATA_FIELDS = (
    "Site",
    "Plot",
    "spad_mean",
    "Variete",
    "Image",
    "Date",
    "DateTime",
    "ISOSpeedRatings_rgb",
    "ExposureTime_rgb",
    "ISOSpeedRatings_nir",
    "ExposureTime_nir",
    "Feuille_1",
    "Feuille_2",
    "Feuille_3",
    "Nb_pixels",
)

INDEX_STAT_FIELDS = tuple(
    field
    for index_name in INDEX_NAMES
    for field in (
        f"Mean_{index_name}",
        f"Median_{index_name}",
        f"Mode_{index_name}",
    )
)

BAND_STAT_FIELDS = tuple(
    field
    for band_name in BAND_NAMES
    for field in (
        f"Mean_{band_name}",
        f"Median_{band_name}",
        f"Mode_{band_name}",
    )
)

MIRE_FIELDS = (
    "intensite_mire_R",
    "intensite_mire_G",
    "intensite_mire_B",
    "intensite_mire_NIR",
    "DN_gris_R",
    "DN_gris_G",
    "DN_gris_B",
    "DN_gris_NIR",
    "intensite_mire_noir_R",
    "intensite_mire_noir_NIR",
    "intensite_mire_blanc_R",
    "intensite_mire_blanc_NIR",
    "rgb_TI",
    "rgb_gain",
    "nir_TI",
    "nir_gain",
)

OUTPUT_FIELDS = (
    METADATA_FIELDS + INDEX_STAT_FIELDS + BAND_STAT_FIELDS + MIRE_FIELDS
)

cv2 = None
np = None
gaussian_kde = None
MODE_RNG = None


def ensure_dependencies() -> None:
    """Charge les bibliotheques scientifiques seulement quand elles sont utiles."""
    global cv2, np, gaussian_kde, MODE_RNG

    if np is not None:
        return

    import cv2 as cv2_module
    import numpy as numpy_module
    from scipy.stats import gaussian_kde as gaussian_kde_function

    cv2 = cv2_module
    np = numpy_module
    gaussian_kde = gaussian_kde_function
    MODE_RNG = np.random.default_rng(0)


def read_csv(path: Path) -> list[dict]:
    """Lit un fichier CSV UTF-8."""
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(rows: list[dict], path: Path) -> Path:
    """Ecrit les lignes avec un ordre de colonnes stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_image(path: Path):
    """Charge une image sans modifier sa profondeur de bits."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return image


def load_bands(site_root: Path, stem: str) -> dict:
    """Charge et normalise les quatre bandes en reflectance 0..1."""
    return {
        band: load_image(site_root / "reflectance" / band / f"{stem}.tif").astype(
            np.float64
        )
        / REFLECTANCE_FULL_SCALE
        for band in ("R", "G", "B", "NIR")
    }


def load_combined_mask(site_root: Path, stem: str):
    """Charge le masque combine sous forme booleenne."""
    path = site_root / "masks" / "combined" / f"{stem}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Masque combine introuvable : {path}")
    return mask > 0


def safe_divide(numerator, denominator):
    """Divise en excluant les denominateurs trop proches de zero."""
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > DENOMINATOR_EPSILON
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def compute_indices(bands: dict) -> dict:
    """Calcule les cinq indices demandes sur l'image complete."""
    red = bands["R"]
    green = bands["G"]
    blue = bands["B"]
    nir = bands["NIR"]

    return {
        "rb_index": safe_divide(red - blue, red + blue),
        "br_bnir_index": safe_divide(blue - red, blue + nir),
        "ci_green": safe_divide(nir, green) - 1.0,
        "nir_b_gb_index": safe_divide(nir - blue, green - blue),
        "nir_gb_ratio_index": safe_divide(nir, green - blue),
    }


def continuous_mode(values) -> float:
    """Estime le mode continu par KDE, avec histogramme de secours."""
    if values.size == 0:
        return float("nan")
    if values.size == 1 or np.ptp(values) == 0:
        return float(values[0])

    sample = values
    if values.size > MODE_SAMPLE_LIMIT:
        sample = MODE_RNG.choice(values, size=MODE_SAMPLE_LIMIT, replace=False)

    try:
        kde = gaussian_kde(sample)
        positions = np.linspace(values.min(), values.max(), 512)
        return float(positions[np.argmax(kde(positions))])
    except Exception:
        counts, limits = np.histogram(values, bins=100)
        index = int(np.argmax(counts))
        return float((limits[index] + limits[index + 1]) / 2.0)


def calculate_stats(indices: dict, mask) -> dict:
    """Calcule moyenne, mediane et mode des pixels conserves par le masque."""
    stats = {}

    for name in INDEX_NAMES:
        values = indices[name][mask]
        values = values[np.isfinite(values)]

        if values.size == 0:
            mean = median = mode = float("nan")
        else:
            mean = float(np.mean(values))
            median = float(np.median(values))
            mode = continuous_mode(values)

        stats[f"Mean_{name}"] = mean
        stats[f"Median_{name}"] = median
        stats[f"Mode_{name}"] = mode

    return stats


def calculate_band_stats(bands: dict, mask) -> dict:
    """Calcule les statistiques des bandes en pourcentage de reflectance."""
    stats = {}

    for name in BAND_NAMES:
        values = bands[name][mask]

        # Les pixels NIR nuls peuvent provenir du decalage de l'image NIR.
        if name == "NIR":
            values = values[values != 0]
        values = values[np.isfinite(values)] * 100.0

        if values.size == 0:
            mean = median = mode = float("nan")
        else:
            mean = float(np.mean(values))
            median = float(np.median(values))
            mode = continuous_mode(values)

        stats[f"Mean_{name}"] = mean
        stats[f"Median_{name}"] = median
        stats[f"Mode_{name}"] = mode

    return stats


def read_xml_value(path: Path, tag: str) -> str:
    """Lit un tag XML ; retourne une chaine vide si le tag est absent."""
    if not path.exists():
        return ""
    try:
        node = ET.parse(path).getroot().find(tag)
        return node.text.strip() if node is not None and node.text else ""
    except (ET.ParseError, OSError) as exc:
        print(f"[WARNING] XML illisible {path} : {exc}")
        return ""


def image_metadata(site_root: Path, stem: str) -> dict:
    """Lit les metadonnees RGB et NIR associees a l'image."""
    nir_stem = stem.replace("Camera1", "Camera3")
    rgb_xml = site_root / "xml_rgb_rect" / f"{stem}.xml"
    nir_xml = site_root / "xml_nir_rect" / f"{nir_stem}.xml"

    return {
        "DateTime": read_xml_value(rgb_xml, "DateTime"),
        "ISOSpeedRatings_rgb": read_xml_value(rgb_xml, "ISOSpeedRatings"),
        "ExposureTime_rgb": read_xml_value(rgb_xml, "ExposureTime"),
        "ISOSpeedRatings_nir": read_xml_value(nir_xml, "ISOSpeedRatings"),
        "ExposureTime_nir": read_xml_value(nir_xml, "ExposureTime"),
    }


def optional_float(value: str) -> float:
    """Convertit une valeur en nombre, ou retourne NaN si elle est absente."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@lru_cache(maxsize=None)
def load_calibration_slopes(site_root: Path) -> dict[tuple[str, str], float]:
    """Charge les pentes ELM d'un site, indexees par (stem, bande)."""
    path = site_root / "reflectance" / "metrics_calibration.csv"
    if not path.exists():
        print(f"[WARNING] Calibration absente : {path}")
        return {}

    slopes = {}
    for row in read_csv(path):
        slope = optional_float(row.get("slope", ""))
        if np.isfinite(slope) and slope != 0:
            slopes[(row.get("stem", ""), row.get("band", ""))] = slope
    return slopes


def calculate_mire_values(site_root: Path, stem: str, metadata: dict) -> dict:
    """Calcule les intensites et DN estimes des patchs de la QPCard."""
    slopes = load_calibration_slopes(site_root)

    rgb_ti = optional_float(metadata["ExposureTime_rgb"]) / 1_000_000.0
    rgb_gain = optional_float(metadata["ISOSpeedRatings_rgb"]) / 100.0
    nir_ti = optional_float(metadata["ExposureTime_nir"]) / 1_000_000.0
    nir_gain = optional_float(metadata["ISOSpeedRatings_nir"]) / 100.0

    result = {
        "rgb_TI": rgb_ti,
        "rgb_gain": rgb_gain,
        "nir_TI": nir_ti,
        "nir_gain": nir_gain,
    }

    for band in BAND_NAMES:
        slope = slopes.get((stem, band), float("nan"))
        if np.isfinite(slope) and slope != 0:
            intensity = PATCH_REFLECTANCE["gris"][band] / slope
        else:
            intensity = float("nan")

        result[f"intensite_mire_{band}"] = intensity
        ti = nir_ti if band == "NIR" else rgb_ti
        gain = nir_gain if band == "NIR" else rgb_gain
        result[f"DN_gris_{band}"] = intensity * ti * gain

    for patch_name in ("noir", "blanc"):
        for band in ("R", "NIR"):
            slope = slopes.get((stem, band), float("nan"))
            value = (
                PATCH_REFLECTANCE[patch_name][band] / slope
                if np.isfinite(slope) and slope != 0
                else float("nan")
            )
            result[f"intensite_mire_{patch_name}_{band}"] = value

    return result


def process_row(row: dict) -> dict:
    """Traite une ligne SPAD et son image, sans fichier intermediaire."""
    site = row["Site"]
    stem = Path(row["Image"]).stem
    site_root = DATA_ROOT / site

    bands = load_bands(site_root, stem)
    mask = load_combined_mask(site_root, stem)

    if mask.shape != bands["R"].shape:
        raise ValueError(
            f"Dimensions incompatibles pour {site}/{stem} : "
            f"masque={mask.shape}, image={bands['R'].shape}"
        )

    result = {
        "Site": site,
        "Plot": row.get("Plot", ""),
        "spad_mean": row.get("spad_mean", ""),
        "Variete": row.get("Variete", ""),
        "Image": row.get("Image", ""),
        "Date": row.get("Date", ""),
        "Feuille_1": row.get("Feuille_1", ""),
        "Feuille_2": row.get("Feuille_2", ""),
        "Feuille_3": row.get("Feuille_3", ""),
        "Nb_pixels": int(np.count_nonzero(mask)),
    }
    metadata = image_metadata(site_root, stem)
    result.update(metadata)
    result.update(calculate_stats(compute_indices(bands), mask))
    result.update(calculate_band_stats(bands, mask))
    result.update(calculate_mire_values(site_root, stem, metadata))
    return result


def process_site(
    site: str,
    rows: list[dict],
    max_images: int | None = None,
    max_workers: int | None = None,
) -> list[dict]:
    """Traite en parallele les images SPAD d'un site."""
    site_rows = [row for row in rows if row.get("Site") == site]
    if max_images is not None:
        site_rows = site_rows[:max_images]
    if not site_rows:
        raise ValueError(f"Aucune ligne SPAD pour le site '{site}'.")

    workers = max_workers or min(os.cpu_count() or 1, len(site_rows))
    results = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(process_row, row): row for row in site_rows}

        for done, future in enumerate(as_completed(futures), start=1):
            source_row = futures[future]
            try:
                results.append(future.result())
                print(
                    f"[OK]   ({done}/{len(site_rows)}) "
                    f"{site}/{source_row.get('Plot', '')} - {source_row['Image']}"
                )
            except (FileNotFoundError, ValueError) as exc:
                skipped += 1
                print(
                    f"[SKIP] ({done}/{len(site_rows)}) "
                    f"{site}/{source_row.get('Plot', '')} - {exc}"
                )

    results.sort(key=lambda item: (item["Plot"], item["Image"]))
    output_path = DATA_ROOT / site / "spad_indices_stats.csv"
    write_csv(results, output_path)
    print(
        f"[DONE] {len(results)}/{len(site_rows)} images traitees, "
        f"{skipped} ignorees - {output_path}"
    )
    return results


def save_merged_results(rows: list[dict]) -> tuple[Path, Path]:
    """Sauvegarde un CSV global fixe et une copie horodatee."""
    rows.sort(key=lambda item: (item["Site"], item["Plot"], item["Image"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = CSV_ROOT / f"SPAD_indices_{timestamp}.csv"
    merged_path = CSV_ROOT / "SPAD_indices_merged.csv"
    write_csv(rows, history_path)
    write_csv(rows, merged_path)
    return history_path, merged_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcule directement les indices et les associe aux mesures SPAD."
    )
    parser.add_argument(
        "sites",
        nargs="*",
        help="Sites a traiter. Sans argument, utilise tous les sites du CSV SPAD.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if not SPAD_CSV.exists():
        print(f"[ERROR] CSV SPAD introuvable : {SPAD_CSV}")
        return 1

    rows = read_csv(SPAD_CSV)
    sites = args.sites or sorted(
        {row["Site"] for row in rows if row.get("Site")}
    )
    if not sites:
        print("[ERROR] Aucun site trouve dans le CSV SPAD.")
        return 1

    ensure_dependencies()
    all_results = []

    try:
        for site in sites:
            print(f"\n=== {site} ===")
            all_results.extend(
                process_site(site, rows, args.max_images, args.workers)
            )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    history_path, merged_path = save_merged_results(all_results)
    print(f"\n[DONE] CSV global : {merged_path}")
    print(f"[DONE] Copie horodatee : {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

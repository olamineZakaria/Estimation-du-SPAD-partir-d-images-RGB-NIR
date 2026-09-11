"""Cree les correspondances SPAD puis le dataset SPAD global.

Ce script unique remplace l'enchainement de ``spad.py`` et
``merge_spad_dataset.py`` :

1. association des images aux mesures SPAD pour chaque site ;
2. creation des ``spad_mapping.csv`` ;
3. normalisation et fusion dans ``data/spad_dataset_merged.csv``.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SPAD_SOURCE_ROOT = DATA_ROOT / "ground_truth_spad"
OUTPUT_PATH = DATA_ROOT / "spad_dataset_merged.csv"

CLERMONT_XLSX = SPAD_SOURCE_ROOT / "SPAD_Clermont_20-05-26_PhB.xlsx"
TOULOUSE_XLSX = SPAD_SOURCE_ROOT / "SPAD_Auzeville_23_04_26.xlsx"
TOULOUSE_PLAN_CSV = SPAD_SOURCE_ROOT / "26TO_Phenet_Auz_Plan_long.csv"
MAUGUIO_CSV = SPAD_SOURCE_ROOT / "SPAD_Mauguio_16_04_26.csv"
ZURICH_XLSX = SPAD_SOURCE_ROOT / "Yara.xlsx"

ZURICH_DATES = (
    "2026-03-23",
    "2026-04-07",
    "2026-04-21",
    "2026-05-05",
    "2026-06-01",
)

CLERMONT_PATTERN = re.compile(r"^Plot.*?(A2|E2)([A-Z]\d+)_Tricam")
TOULOUSE_PATTERN = re.compile(r"Plot26TO25_Y0(\d+)X0(\d+)_Tricam")
MAUGUIO_PATTERN = re.compile(r"^Plot(\d+)_Tricam")
ZURICH_PATTERN = re.compile(r"^Plot(\d+)_(\d+)_.*?8617WW00100(\d+)")
PLOT_PATTERN = re.compile(r"Plot(.+?)_Tricam")

HNT_TO_SPAD_SLOPE = 0.0639
HNT_TO_SPAD_INTERCEPT = 5.84

LEAF_COLUMNS = ["Feuille_1", "Feuille_2", "Feuille_3"]
SPAD_MEAN_ALIASES = ("SPAD_Value_Mean", "spad_mean", "SPAD_Value")
LEAF_COLUMN_ALIASES = (
    ("Feuille_1", "Feuille_2", "Feuille_3"),
    ("SPAD_Value_1", "SPAD_Value_2", "SPAD_Value_3"),
)

pd = None


def ensure_pandas() -> None:
    """Charge pandas seulement au lancement du traitement."""
    global pd
    if pd is None:
        import pandas as pandas_module

        pd = pandas_module


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def image_records(images_dir: Path, parser, key_names: tuple[str, ...]):
    """Extrait les cles de jointure depuis les noms des images PNG."""
    images = sorted(images_dir.glob("*.png"))
    records = []
    skipped = 0

    for image in images:
        parsed = parser(image.stem)
        if parsed is None:
            skipped += 1
            continue
        if not isinstance(parsed, tuple):
            parsed = (parsed,)
        records.append({"image": image.name, **dict(zip(key_names, parsed))})

    return pd.DataFrame.from_records(records, columns=("image", *key_names)), images, skipped


def create_clermont_mapping() -> Path:
    """Associe les images de Clermont par la cle (bloc, plot_code)."""
    require_file(CLERMONT_XLSX)
    site_root = DATA_ROOT / "clermont-ferrand"
    output_path = site_root / "spad_mapping.csv"

    spad = pd.read_excel(CLERMONT_XLSX, sheet_name="Valeur SPAD")
    not_measured = spad["PLOTn0"].isna()
    if not_measured.any():
        print(f"[INFO] Clermont : {int(not_measured.sum())} plots non mesures ecartes")
    spad = spad[~not_measured].rename(
        columns={"BLOCK": "bloc", "PLOTn0": "plot_code"}
    )

    def parse(stem: str):
        match = CLERMONT_PATTERN.match(stem)
        return (match.group(1), match.group(2)) if match else None

    images_df, images, skipped = image_records(
        site_root / "images_rgb_rect", parse, ("bloc", "plot_code")
    )
    keep = [
        "bloc", "plot_code", "ID", "Nom_complet", "Variete", "Repetition",
        "Feuille_1", "Feuille_2", "Feuille_3",
    ]
    merged = images_df.merge(spad[keep], on=["bloc", "plot_code"], how="left")
    missing = merged["Nom_complet"].isna()
    if missing.any():
        print(f"[WARN] Clermont : {int(missing.sum())} images sans mesure SPAD")
    merged = merged[~missing].copy()
    merged["Date"] = "2026-05-20"
    merged.to_csv(output_path, index=False)
    print(f"[DONE] Clermont : {len(merged)} lignes, {skipped} images ignorees")
    return output_path


def create_toulouse_mapping() -> Path:
    """Associe les images de Toulouse par (numero_planche, numero)."""
    require_file(TOULOUSE_XLSX)
    site_root = DATA_ROOT / "toulouse"
    output_path = site_root / "spad_mapping.csv"

    spad = pd.read_excel(TOULOUSE_XLSX, sheet_name="Valeur SPAD")
    if TOULOUSE_PLAN_CSV.exists():
        plan = pd.read_csv(TOULOUSE_PLAN_CSV)
        spad = spad.merge(
            plan[["N°_planche", "N°", "Variete"]],
            on=["N°_planche", "N°"],
            how="left",
        )
    else:
        print(f"[WARN] Plan Toulouse absent : {TOULOUSE_PLAN_CSV}")
        spad["Variete"] = float("nan")

    def parse(stem: str):
        match = TOULOUSE_PATTERN.match(stem)
        return (int(match.group(1)), int(match.group(2))) if match else None

    images_df, images, skipped = image_records(
        site_root / "images_rgb_rect", parse, ("N°_planche", "N°")
    )
    merged = images_df.merge(spad, on=["N°_planche", "N°"], how="left")
    missing = merged["NOM"].isna()
    if missing.any():
        print(f"[WARN] Toulouse : {int(missing.sum())} images sans mesure SPAD")
    merged = merged[~missing].copy()
    merged["Date"] = "2026-04-23"
    merged.to_csv(output_path, index=False)
    print(f"[DONE] Toulouse : {len(merged)} lignes, {skipped} images ignorees")
    return output_path


def create_mauguio_mapping() -> Path:
    """Associe les images de Mauguio par leur numero de plot."""
    require_file(MAUGUIO_CSV)
    site_root = DATA_ROOT / "mauguio"
    output_path = site_root / "spad_mapping.csv"

    spad = pd.read_csv(MAUGUIO_CSV, sep=";", decimal=",")
    keep = [
        "Plot", "PlotAlias", "Variety", "ExperimentModalities", "Replication",
        "SPAD_Value_1", "SPAD_Value_2", "SPAD_Value_3", "SPAD_Value_Mean",
        "Date_SPAD",
    ]

    def parse(stem: str):
        match = MAUGUIO_PATTERN.match(stem)
        return int(match.group(1)) if match else None

    images_df, images, skipped = image_records(
        site_root / "images_rgb_rect", parse, ("Plot",)
    )
    merged = images_df.merge(spad[keep], on="Plot", how="left")
    merged["Variete"] = merged["Variety"]
    merged["Date"] = "2026-04-16"
    merged = merged.drop(columns=["Variety"], errors="ignore")
    missing = merged["SPAD_Value_Mean"].isna()
    if missing.any():
        print(f"[WARN] Mauguio : {int(missing.sum())} images sans mesure SPAD")
    merged = merged[~missing].copy()
    merged.to_csv(output_path, index=False)
    print(f"[DONE] Mauguio : {len(merged)} lignes, {skipped} images ignorees")
    return output_path


def create_zurich_mappings() -> list[Path]:
    """Cree un mapping par date pour Zurich."""
    require_file(ZURICH_XLSX)
    spad_base = pd.read_excel(ZURICH_XLSX, sheet_name="Tabelle1")
    output_paths = []

    for date_string in ZURICH_DATES:
        site_root = DATA_ROOT / "zurich" / date_string
        images_dir = site_root / "images_rgb_rect"
        if not images_dir.is_dir():
            print(f"[WARN] Zurich {date_string} : dossier images absent")
            continue

        date_dot = datetime.strptime(date_string, "%Y-%m-%d").strftime("%d.%m.%Y")
        yara_columns = [
            column
            for column in spad_base.columns
            if str(column).startswith(f"Yara_{date_dot}")
        ]
        if not yara_columns:
            print(f"[WARN] Zurich {date_string} : colonne Yara absente")
            continue
        yara_column = yara_columns[0]

        spad = spad_base.copy()
        spad["SPAD_Value"] = (
            HNT_TO_SPAD_SLOPE * spad[yara_column] + HNT_TO_SPAD_INTERCEPT
        )

        def parse(stem: str):
            match = ZURICH_PATTERN.match(stem)
            return (
                (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                if match else None
            )

        images_df, images, skipped = image_records(
            images_dir, parse, ("Wiederholung", "Position", "Nr.")
        )
        merged = images_df.merge(
            spad[["Nr.", "Buchstabe", yara_column, "SPAD_Value"]],
            on="Nr.",
            how="left",
        )
        missing = merged["SPAD_Value"].isna()
        if missing.any():
            print(
                f"[WARN] Zurich {date_string} : "
                f"{int(missing.sum())} images sans mesure SPAD"
            )
        merged["Variete"] = merged["Buchstabe"]
        merged["Date"] = date_string

        output_path = site_root / "spad_mapping.csv"
        merged.to_csv(output_path, index=False)
        output_paths.append(output_path)
        print(
            f"[DONE] Zurich {date_string} : {len(merged)} lignes, "
            f"{skipped} images ignorees"
        )

    return output_paths


def create_all_mappings() -> list[Path]:
    """Genere les mappings de tous les sites pris en charge."""
    paths = [
        create_clermont_mapping(),
        create_toulouse_mapping(),
        create_mauguio_mapping(),
    ]
    paths.extend(create_zurich_mappings())
    return paths


def find_image_column(frame) -> str:
    for candidate in ("image", "image_name"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Aucune colonne image dans {list(frame.columns)}")


def extract_plot(filename: str) -> str | None:
    match = PLOT_PATTERN.search(str(filename))
    return match.group(1) if match else None


def compute_spad_mean(frame):
    if all(column in frame.columns for column in LEAF_COLUMNS):
        return frame[LEAF_COLUMNS].mean(axis=1)
    for alias in SPAD_MEAN_ALIASES:
        if alias in frame.columns:
            return frame[alias]
    raise ValueError(f"Valeur SPAD introuvable dans {list(frame.columns)}")


def extract_leaf_values(frame):
    for aliases in LEAF_COLUMN_ALIASES:
        if all(column in frame.columns for column in aliases):
            return frame[list(aliases)].set_axis(LEAF_COLUMNS, axis=1)
    return pd.DataFrame(
        {column: float("nan") for column in LEAF_COLUMNS}, index=frame.index
    )


def image_is_processed(name: str | None, site_root: Path) -> bool:
    """Verifie que l'image representative est prete pour le pipeline d'indices."""
    if not isinstance(name, str) or not name:
        return False
    stem = Path(name).stem
    return (
        (site_root / "reflectance" / "R" / f"{stem}.tif").exists()
        and (site_root / "masks" / "combined" / f"{stem}.png").exists()
    )


def normalize_site_mapping(site_root: Path):
    """Convertit un spad_mapping.csv vers le schema commun."""
    frame = pd.read_csv(site_root / "spad_mapping.csv")
    image_column = find_image_column(frame)
    if image_column != "image":
        frame = frame.rename(columns={image_column: "image"})

    site_name = site_root.relative_to(DATA_ROOT).as_posix()
    output = pd.DataFrame(index=frame.index)
    output["Site"] = site_name
    output["Plot"] = frame["image"].map(extract_plot)
    output["spad_mean"] = compute_spad_mean(frame)
    output["Variete"] = frame.get("Variete", float("nan"))
    output["Date"] = frame.get("Date", float("nan"))

    leaf_values = extract_leaf_values(frame)
    for column in LEAF_COLUMNS:
        output[column] = leaf_values[column]

    images_dir = site_root / "images_rgb_rect"
    output["Image"] = frame["image"].apply(
        lambda name: name if (images_dir / str(name)).exists() else None
    )
    output["IsProcessed"] = output["Image"].apply(
        lambda name: image_is_processed(name, site_root)
    )
    return output


def merge_mappings() -> Path:
    """Fusionne tous les mappings disponibles dans le dataset final."""
    site_roots = sorted(
        {path.parent for path in DATA_ROOT.rglob("spad_mapping.csv")}
    )
    if not site_roots:
        raise FileNotFoundError(f"Aucun spad_mapping.csv sous {DATA_ROOT}")

    all_rows = pd.concat(
        [normalize_site_mapping(site_root) for site_root in site_roots],
        ignore_index=True,
    )
    all_rows = all_rows.dropna(subset=["Plot"])
    all_rows = all_rows.sort_values("IsProcessed", ascending=False, kind="stable")

    grouped = all_rows.groupby(["Site", "Plot"], as_index=False)
    result = grouped["spad_mean"].mean()
    result["spad_mean"] = result["spad_mean"].round(2)

    for column in ("Image", "Variete", "Date", *LEAF_COLUMNS):
        result = result.merge(
            grouped[column].first(), on=["Site", "Plot"], how="left"
        )

    result = result.sort_values(["Site", "Plot"]).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"[DONE] {len(result)} plots ecrits dans {OUTPUT_PATH}")
    return OUTPUT_PATH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cree les mappings SPAD et le dataset SPAD fusionne."
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Fusionne les mappings existants sans les regenerer.",
    )
    parser.add_argument(
        "--mappings-only",
        action="store_true",
        help="Cree les mappings sans produire le dataset global.",
    )
    args = parser.parse_args()

    if args.merge_only and args.mappings_only:
        parser.error("--merge-only et --mappings-only sont incompatibles")

    ensure_pandas()

    try:
        if not args.merge_only:
            create_all_mappings()
        if not args.mappings_only:
            merge_mappings()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

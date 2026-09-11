"""Interface web Flask pour explorer les images par site et par nom."""

from functools import lru_cache
from pathlib import Path
import json
import re

import cv2
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CSV_PATH = PROJECT_ROOT / "csv" / "SPAD_indices_merged.csv"

REFLECTANCE_FULL_SCALE = 2**16
DENOMINATOR_EPSILON = 1e-2


def load_reflectance_bands(site_root: Path, stem: str) -> dict[str, np.ndarray]:
    """Charge les quatre bandes de reflectance d'une image."""
    bands = {}
    for band in ("R", "G", "B", "NIR"):
        path = site_root / "reflectance" / band / f"{stem}.tif"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Bande {band} introuvable : {path}")
        bands[band] = image
    return bands


def normalize_bands(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convertit les DN 16 bits en fraction de reflectance 0..1."""
    return {
        name: values.astype(np.float64) / REFLECTANCE_FULL_SCALE
        for name, values in bands.items()
    }


def _safe_array_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > DENOMINATOR_EPSILON
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def compute_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calcule les cinq indices utilises par le nouveau pipeline."""
    red, green, blue, nir = bands["R"], bands["G"], bands["B"], bands["NIR"]
    return {
        "rb_index": _safe_array_ratio(red - blue, red + blue),
        "br_bnir_index": _safe_array_ratio(blue - red, blue + nir),
        "ci_green": _safe_array_ratio(nir, green) - 1.0,
        "nir_b_gb_index": _safe_array_ratio(nir - blue, green - blue),
        "nir_gb_ratio_index": _safe_array_ratio(nir, green - blue),
    }


# Indices calculables a la volee sous forme de carte 2D.
HEATMAP_INDEX_KEYS = (
    "rb_index", "br_bnir_index", "ci_green",
    "nir_b_gb_index", "nir_gb_ratio_index",
)

SCENARIO_ORDER = ["Combined"]

# Indices calcules a partir de R/G/B/NIR (memes formules que RB_index,
# verifiees par retro-ingenierie sur data/clermont-ferrand/pixel_median_stats_clean.csv).
# Ils peuvent etre calcules a partir de la moyenne, la mediane ou le mode des pixels.
INDEX_DEFS = {
    "rb_index": {"label": "(R-B)/(R+B)", "base": "index_rb"},
    "br_bnir_index": {"label": "(B-R)/(B+NIR)", "base": "br_bnir_index"},
    "ci_green": {"label": "CIgreen = NIR/G - 1", "base": "ci_green"},
    "nir_b_gb_index": {"label": "(NIR-B)/(G-B)", "base": "nir_b_gb_index"},
    "nir_gb_ratio_index": {"label": "NIR/(G-B)", "base": "nir_gb_ratio_index"},
}

CUSTOM_INDICES_FILE = Path(__file__).resolve().parent / "custom_indices.json"

def load_custom_indices() -> dict:
    if CUSTOM_INDICES_FILE.is_file():
        try:
            return json.loads(CUSTOM_INDICES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_custom_indices(indices: dict):
    try:
        CUSTOM_INDICES_FILE.write_text(json.dumps(indices, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def get_all_indices() -> dict:
    customs = load_custom_indices()
    merged = INDEX_DEFS.copy()
    for k, v in customs.items():
        merged[k] = {
            "label": f"{v['name']} = {v['formula']}",
            "base": f"custom_{k}",
            "formula": v["formula"],
            "custom": True
        }
    return merged

def compute_formula(df: pd.DataFrame, prefix: str, formula_str: str) -> pd.Series:
    # Remplacer les tirets/signes moins Unicode (ex. en-dash, em-dash, minus sign) par des traits d'union standards
    normalized = formula_str.replace('–', '-').replace('—', '-').replace('−', '-')
    cleaned = re.sub(r'\s+', '', normalized)
    allowed_pattern = re.compile(r'^(?:R|G|B|NIR|[0-9\.\+\-\*/\(\)])+$', re.IGNORECASE)
    if not allowed_pattern.match(cleaned):
        raise ValueError("Formule contient des caractères non autorisés.")
    
    local_dict = {
        "R": df[f"{prefix}_R"],
        "G": df[f"{prefix}_G"],
        "B": df[f"{prefix}_B"],
        "NIR": df[f"{prefix}_NIR"],
    }
    local_dict["r"] = local_dict["R"]
    local_dict["g"] = local_dict["G"]
    local_dict["b"] = local_dict["B"]
    local_dict["nir"] = local_dict["NIR"]
    local_dict["NIR"] = local_dict["NIR"]
    
    try:
        result = pd.eval(normalized, local_dict=local_dict, engine="python")
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result
    except Exception as e:
        raise ValueError(f"Erreur dans la formule : {e}")

STAT_PREFIXES = {"mean": "Mean", "median": "Median", "mode": "Mode"}
STAT_LABELS = {"mean": "Moyenne", "median": "Mediane", "mode": "Mode"}
DEFAULT_STAT = "median"

import math
from datetime import datetime

def calculate_solar_zenith(date_str: str, site_name: str) -> float:
    if not isinstance(date_str, str) or not date_str:
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y/%m/%d %H:%M:%S")
    except Exception:
        try:
            dt = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    coords = {
        "clermont-ferrand": (45.7772, 3.0870),
        "mauguio": (43.6163, 3.9783),
        "toulouse": (43.6047, 1.4442),
        "zurich": (47.3769, 8.5417)
    }
    
    s_norm = site_name.strip().lower()
    lat, lon = None, None
    if s_norm in coords:
        lat, lon = coords[s_norm]
    else:
        for k, v in coords.items():
            if k in s_norm or s_norm in k:
                lat, lon = v
                break
        if lat is None:
            return None
            
    day_of_year = dt.timetuple().tm_yday
    declination = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (284 + day_of_year))))
    b = math.radians(360.0 / 365.0 * (day_of_year - 81))
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    
    offset = 1
    if 3 < dt.month < 10:
        offset = 2
    elif dt.month == 3:
        if dt.day >= 29:
            offset = 2
    elif dt.month == 10:
        if dt.day < 25:
            offset = 2
            
    lstm = 15 * offset
    tc = 4 * (lon - lstm) + eot
    lst_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + tc / 60.0
    h = 15 * (lst_hours - 12.0)
    h_rad = math.radians(h)
    lat_rad = math.radians(lat)
    
    cos_zenith = math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(declination) * math.cos(h_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    
    zenith_deg = math.degrees(math.acos(cos_zenith))
    return round(zenith_deg, 2)

app = Flask(__name__)


def _load_stats() -> pd.DataFrame:
    if not CSV_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, comment="#")
    df["name"] = df["Image"].str.replace(".png", "", regex=False)
    df["Site"] = df["Site"].str.strip()

    if "Feuille_1" in df.columns and "Feuille_2" in df.columns:
        df["F1_F2"] = (df["Feuille_1"] + df["Feuille_2"]) / 2

    if "DateTime" in df.columns:
        df["solar_zenith"] = df.apply(
            lambda r: calculate_solar_zenith(r["DateTime"], r["Site"]) if pd.notna(r["DateTime"]) and pd.notna(r["Site"]) else None,
            axis=1
        )
        def format_time_hm(dt_str):
            if not isinstance(dt_str, str) or not dt_str:
                return None
            try:
                parts = dt_str.strip().split()
                if len(parts) >= 2:
                    hm_parts = parts[1].split(":")
                    if len(hm_parts) >= 2:
                        return f"{hm_parts[0]}:{hm_parts[1]}"
            except Exception:
                pass
            return None
        df["time_hm"] = df["DateTime"].apply(format_time_hm)

    required_columns = {
        "Mean_rb_index", "Median_rb_index", "Mode_rb_index",
        "Mean_br_bnir_index", "Median_br_bnir_index", "Mode_br_bnir_index",
        "Mean_ci_green", "Median_ci_green", "Mode_ci_green",
        "Mean_nir_b_gb_index", "Median_nir_b_gb_index", "Mode_nir_b_gb_index",
        "Mean_nir_gb_ratio_index", "Median_nir_gb_ratio_index", "Mode_nir_gb_ratio_index",
        "Mean_R", "Median_R", "Mode_R", "Mean_G", "Median_G", "Mode_G",
        "Mean_B", "Median_B", "Mode_B", "Mean_NIR", "Median_NIR", "Mode_NIR",
    }
    if not required_columns.issubset(df.columns):
        missing = sorted(required_columns - set(df.columns))
        raise ValueError(f"Colonnes absentes de {CSV_PATH}: {', '.join(missing)}")

    df["Scenario"] = "Combined"
    for stat_key, prefix in STAT_PREFIXES.items():
        df[f"index_rb_{stat_key}"] = df[f"{prefix}_rb_index"]
        df[f"br_bnir_index_{stat_key}"] = df[f"{prefix}_br_bnir_index"]
        df[f"ci_green_{stat_key}"] = df[f"{prefix}_ci_green"]
        df[f"nir_b_gb_index_{stat_key}"] = df[f"{prefix}_nir_b_gb_index"]
        df[f"nir_gb_ratio_index_{stat_key}"] = df[f"{prefix}_nir_gb_ratio_index"]

    # Le nouveau CSV conserve le nombre de pixels masques, mais pas le nombre
    # total de pixels de l'image : le pourcentage ne peut donc pas etre deduit.
    df["Pixel_percentage"] = np.nan
    df["scenario_order"] = 0
    return df


STATS_DF = _load_stats()

ORIGINAL_COLUMNS = [
    c for c in STATS_DF.columns
    if c != "scenario_order"
    and not c.startswith("index_rb_")
    and not c.startswith("br_bnir_index_")
    and not c.startswith("ci_green_")
    and not c.startswith("nir_b_gb_index_")
    and not c.startswith("nir_gb_ratio_index_")
]


def list_sites() -> list[str]:
    """Retourne uniquement les sites presents dans le nouveau CSV final."""
    if STATS_DF.empty or "Site" not in STATS_DF.columns:
        return []
    return sorted(
        site
        for site in STATS_DF["Site"].dropna().unique()
        if (DATA_ROOT / site / "images_rgb_rect").is_dir()
    )


def get_site_dir(site: str) -> Path:
    if site not in list_sites():
        abort(404, description=f"Site inconnu: {site}")
    return DATA_ROOT / site


def list_names(site: str) -> list[str]:
    """Noms d'images presents a la fois dans le CSV et sur le disque."""
    site_dir = get_site_dir(site)
    images_dir = site_dir / "images_rgb_rect"
    csv_names = STATS_DF.loc[STATS_DF["Site"] == site, "name"].dropna().unique()
    return sorted(name for name in csv_names if (images_dir / f"{name}.png").is_file())


def list_scenarios(site: str) -> list[str]:
    if STATS_DF.empty:
        return []
    site_rows = STATS_DF[STATS_DF["Site"] == site]
    scenarios = set(site_rows["Scenario"].dropna().unique())
    
    # Exclusions
    scenarios.discard("Vegetation-no-specular-no-shadow-no-epi-no-tige-no-glaucousness")
    scenarios.discard("Senescent_veg")
    scenarios.discard("Vegetation-no-Senescent_veg")
    
    return sorted(list(scenarios), key=lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 999)


def list_all_scenarios() -> list[str]:
    """Union des scenarios disponibles, tous sites confondus (pour le scatter multi-sites)."""
    if STATS_DF.empty:
        return []
    scenarios = set(STATS_DF["Scenario"].dropna().unique())
    
    # Exclusions
    scenarios.discard("Vegetation-no-specular-no-shadow-no-epi-no-tige-no-glaucousness")
    scenarios.discard("Senescent_veg")
    scenarios.discard("Vegetation-no-Senescent_veg")
    
    return sorted(list(scenarios), key=lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 999)


def get_name_path(site: str, name: str) -> Path:
    """Valide que name existe bien pour ce site et renvoie le chemin de base."""
    if name not in list_names(site):
        abort(404, description=f"Nom inconnu pour {site}: {name}")
    return get_site_dir(site)


@app.route("/")
def index():
    return render_template("index.html", sites=list_sites())


@app.route("/api/sites")
def api_sites():
    return jsonify(list_sites())


@app.route("/api/names")
def api_names():
    site = request.args.get("site", "")
    q = request.args.get("q", "").strip().lower()
    class_filter = request.args.get("class_filter", "all")
    names = list_names(site)
    if str(class_filter) in ("0", "1") and not STATS_DF.empty and "class" in STATS_DF.columns:
        valid_names = set(STATS_DF[(STATS_DF["Site"] == site) & (STATS_DF["class"] == int(class_filter))]["name"])
        names = [n for n in names if n in valid_names]
    if q:
        names = [n for n in names if q in n.lower()]
    return jsonify(names[:200])


@app.route("/api/scenarios")
def api_scenarios():
    site = request.args.get("site", "")
    return jsonify(list_scenarios(site))


@app.route("/api/scenarios_all")
def api_scenarios_all():
    return jsonify(list_all_scenarios())


@app.route("/api/image_stats")
def api_image_stats():
    site = request.args.get("site", "")
    name = request.args.get("name", "")
    stat_key = request.args.get("stat", DEFAULT_STAT)
    if stat_key not in STAT_PREFIXES:
        abort(404, description=f"Statistique inconnue: {stat_key}")
    prefix = STAT_PREFIXES[stat_key]
    if STATS_DF.empty:
        return jsonify([])
    rows = STATS_DF[(STATS_DF["Site"] == site) & (STATS_DF["name"] == name)]
    rows = rows.sort_values("scenario_order")
    cols = [
        "Scenario", "spad_mean", "Nb_pixels",
        f"{prefix}_R", f"{prefix}_G", f"{prefix}_B", f"{prefix}_NIR",
    ]
    if "class" in rows.columns:
        cols.append("class")
    rows = rows[cols].rename(columns={
        f"{prefix}_R": "stat_R", f"{prefix}_G": "stat_G",
        f"{prefix}_B": "stat_B", f"{prefix}_NIR": "stat_NIR",
    })
    records = rows.replace({np.nan: None}).to_dict(orient="records")
    return jsonify(records)


@app.route("/api/indices")
def api_indices():
    all_indices = get_all_indices()
    return jsonify([{"key": k, "label": v["label"], "custom": v.get("custom", False)} for k, v in all_indices.items()])


@app.route("/api/indices/add", methods=["POST"])
def api_add_index():
    data = request.json or {}
    name = data.get("name", "").strip()
    formula = data.get("formula", "").strip()
    if not name or not formula:
        return jsonify({"error": "Nom et formule requis."}), 400
    
    # Normaliser la formule (remplacer les tirets Unicode)
    formula = formula.replace('–', '-').replace('—', '-').replace('−', '-')
    
    key = re.sub(r'[^a-zA-Z0-9_]', '', name.lower().replace(" ", "_"))
    if not key:
        return jsonify({"error": "Nom d'indice invalide (caractères alphabétiques requis)."}), 400
        
    all_indices = get_all_indices()
    if key in INDEX_DEFS:
        return jsonify({"error": "Cet indice est prédéfini et ne peut pas être modifié."}), 400
        
    # Validate formula syntax on dummy data
    try:
        dummy_df = pd.DataFrame({
            "Median_R": [1.0], "Median_G": [1.0], "Median_B": [1.0], "Median_NIR": [1.0]
        })
        compute_formula(dummy_df, "Median", formula)
    except Exception as e:
        return jsonify({"error": f"Formule invalide : {e}"}), 400
        
    customs = load_custom_indices()
    customs[key] = {"name": name, "formula": formula}
    save_custom_indices(customs)
    
    return jsonify({"success": True, "key": key, "label": f"{name} = {formula}"})


@app.route("/api/indices/delete/<key>", methods=["DELETE"])
def api_delete_index(key):
    customs = load_custom_indices()
    if key in customs:
        del customs[key]
        save_custom_indices(customs)
        return jsonify({"success": True})
    return jsonify({"error": "Indice introuvable."}), 404


@app.route("/api/stats_options")
def api_stats_options():
    return jsonify([{"key": k, "label": v} for k, v in STAT_LABELS.items()])


def _iqr_outliers(series: pd.Series) -> pd.Series:
    if len(series) < 4:
        return pd.Series(False, index=series.index)
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < low) | (series > high)


def _corr(rows: pd.DataFrame, column: str) -> tuple[float | None, float | None]:
    if len(rows) < 2:
        return None, None
    r = rows["spad_mean"].corr(rows[column])
    rho = rows["spad_mean"].corr(rows[column], method="spearman")
    return (None if r is None or np.isnan(r) else r), (None if rho is None or np.isnan(rho) else rho)


from scipy.cluster.vq import kmeans2

def compute_zenith_clusters(df: pd.DataFrame, n_clusters: int = 2) -> pd.Series:
    labels = pd.Series(index=df.index, dtype=object)
    for site, group in df.groupby("Site"):
        zeniths = group["solar_zenith"].dropna()
        if len(zeniths) < n_clusters or n_clusters < 2:
            for idx in group.index:
                labels.loc[idx] = "Zenith Unique"
            continue
        
        data = zeniths.values.astype(float)
        initial_centroids = np.linspace(data.min(), data.max(), n_clusters)
        centroids, assignments = kmeans2(data, initial_centroids, minit='matrix')
        
        sorted_indices = np.argsort(centroids)
        cluster_names = []
        for rank, c_idx in enumerate(sorted_indices):
            c_val = centroids[c_idx]
            if n_clusters == 2:
                name = f"Zenith Bas ({c_val:.1f}°)" if rank == 0 else f"Zenith Haut ({c_val:.1f}°)"
            elif n_clusters == 3:
                name = "Zenith Bas" if rank == 0 else ("Zenith Moyen" if rank == 1 else "Zenith Haut")
                name = f"{name} ({c_val:.1f}°)"
            else:
                name = f"Cluster {rank+1} ({c_val:.1f}°)"
            cluster_names.append(name)
            
        rank_map = {c_idx: rank for rank, c_idx in enumerate(sorted_indices)}
        for idx, assign_val in zip(zeniths.index, assignments):
            rank = rank_map[assign_val]
            labels.loc[idx] = cluster_names[rank]
            
    return labels.fillna("Zenith N/A")


def _scatter_data(
    sites: list[str], scenario: str, index_key: str, stat_key: str,
    exclude_spad_outliers: bool, exclude_index_outliers: bool,
    zenith_clusters: int = 0, class_filter: str = "all"
):
    """Renvoie (rows_used, records, outliers_spad, outliers_index, r, rho, column, prefix, correlations)."""
    all_indices = get_all_indices()
    if index_key not in all_indices:
        abort(404, description=f"Indice inconnu: {index_key}")
    if stat_key not in STAT_PREFIXES:
        abort(404, description=f"Statistique inconnue: {stat_key}")
    
    idx_def = all_indices[index_key]
    column = f"{idx_def['base']}_{stat_key}"
    prefix = STAT_PREFIXES[stat_key]

    if STATS_DF.empty:
        return STATS_DF, [], [], [], None, None, column, prefix, {}

    # Dynamic calculation for custom index if not present in STATS_DF
    if column not in STATS_DF.columns:
        if "formula" in idx_def:
            try:
                STATS_DF[column] = compute_formula(STATS_DF, prefix, idx_def["formula"])
            except Exception:
                STATS_DF[column] = np.nan
        else:
            abort(404, description=f"Formule introuvable pour {index_key}")

    rows = STATS_DF[(STATS_DF["Site"].isin(sites)) & (STATS_DF["Scenario"] == scenario)]
    if "class" in rows.columns:
        if str(class_filter) == "0":
            rows = rows[rows["class"] == 0]
        elif str(class_filter) == "1":
            rows = rows[rows["class"] == 1]
    rows = rows.dropna(subset=["spad_mean", column])

    is_outlier_spad = _iqr_outliers(rows["spad_mean"])
    is_outlier_index = _iqr_outliers(rows[column])

    cols = ["Site", "name", "Plot", "spad_mean", "DateTime", column, f"{prefix}_R", f"{prefix}_G", f"{prefix}_B", f"{prefix}_NIR"]
    for f_col in ["Feuille_1", "Feuille_2", "Feuille_3", "F1_F2"]:
        if f_col in rows.columns:
            cols.append(f_col)
    for extra_col in ["solar_zenith", "time_hm"]:
        if extra_col in rows.columns:
            cols.append(extra_col)
    if "Variete" in rows.columns:
        cols.append("Variete")
    if "class" in rows.columns:
        cols.append("class")
    for mire_col in [
        "intensite_mire_R", "intensite_mire_G", "intensite_mire_B", "intensite_mire_NIR",
        "DN_gris_R", "DN_gris_G", "DN_gris_B", "DN_gris_NIR",
        "intensite_mire_noir_R", "intensite_mire_noir_NIR",
        "intensite_mire_blanc_R", "intensite_mire_blanc_NIR",
        "rgb_TI", "rgb_gain", "nir_TI", "nir_gain"
    ]:
        if mire_col in rows.columns:
            cols.append(mire_col)

    display = rows[cols].rename(columns={
        column: "value", f"{prefix}_R": "stat_R", f"{prefix}_G": "stat_G",
        f"{prefix}_B": "stat_B", f"{prefix}_NIR": "stat_NIR",
    })
    display["outlier_spad"] = is_outlier_spad
    display["outlier_index"] = is_outlier_index
    all_records = display.replace({np.nan: None}).to_dict(orient="records")
    outliers_spad = [rec for rec in all_records if rec["outlier_spad"]]
    outliers_index = [rec for rec in all_records if rec["outlier_index"]]

    keep = pd.Series(True, index=rows.index)
    if exclude_spad_outliers:
        keep &= ~is_outlier_spad
    if exclude_index_outliers:
        keep &= ~is_outlier_index

    rows_used = rows[keep].copy()
    records = [rec for rec, k in zip(all_records, keep) if k]

    if zenith_clusters > 0 and "solar_zenith" in rows_used.columns:
        cluster_labels = compute_zenith_clusters(rows_used, zenith_clusters)
        rows_used["zenith_cluster"] = cluster_labels
        updated_records = []
        for rec, (_, row) in zip(records, rows_used.iterrows()):
            rec["zenith_cluster"] = row.get("zenith_cluster", None)
            updated_records.append(rec)
        records = updated_records
    else:
        for rec in records:
            rec["zenith_cluster"] = None

    r = rows_used["spad_mean"].corr(rows_used[column]) if len(rows_used) >= 2 else None
    if r is not None and np.isnan(r):
        r = None
    rho = rows_used["spad_mean"].corr(rows_used[column], method="spearman") if len(rows_used) >= 2 else None
    if rho is not None and np.isnan(rho):
        rho = None

    correlations = {}
    for var in ["spad_mean", "Feuille_1", "Feuille_2", "Feuille_3", "F1_F2"]:
        if var in rows_used.columns:
            valid_rows = rows_used.dropna(subset=[var, column])
            if len(valid_rows) >= 2:
                r_val = valid_rows[var].corr(valid_rows[column])
                rho_val = valid_rows[var].corr(valid_rows[column], method="spearman")
                correlations[var] = {
                    "r": None if np.isnan(r_val) else float(r_val),
                    "rho": None if np.isnan(rho_val) else float(rho_val),
                    "n": int(len(valid_rows))
                }
            else:
                correlations[var] = {"r": None, "rho": None, "n": len(valid_rows)}
        else:
            correlations[var] = {"r": None, "rho": None, "n": 0}

    return rows_used, records, outliers_spad, outliers_index, r, rho, column, prefix, correlations


def _parse_sites_param() -> list[str]:
    sites_param = request.args.get("sites", "")
    return [s for s in sites_param.split(",") if s]


@app.route("/api/scatter")
def api_scatter():
    sites = _parse_sites_param()
    scenario = request.args.get("scenario", "")
    index_key = request.args.get("index", "rb_index")
    stat_key = request.args.get("stat", DEFAULT_STAT)
    exclude_spad_outliers = request.args.get("exclude_spad_outliers", "0") == "1"
    exclude_index_outliers = request.args.get("exclude_index_outliers", "0") == "1"
    class_filter = request.args.get("class_filter", "all")
    try:
        zenith_clusters = int(request.args.get("zenith_clusters", "0"))
    except ValueError:
        zenith_clusters = 0

    _, records, outliers_spad, outliers_index, r, rho, _, _, correlations = _scatter_data(
        sites, scenario, index_key, stat_key, exclude_spad_outliers, exclude_index_outliers, zenith_clusters, class_filter
    )

    all_indices = get_all_indices()
    return jsonify({
        "points": records,
        "outliers_spad": outliers_spad,
        "outliers_index": outliers_index,
        "r": r,
        "rho": rho,
        "correlations": correlations,
        "n": len(records),
        "label": all_indices[index_key]["label"],
    })


@app.route("/api/spad_distribution")
def api_spad_distribution():
    sites = _parse_sites_param()
    scenario = request.args.get("scenario", "")
    x_var = request.args.get("x_var", "spad_mean")
    class_filter = request.args.get("class_filter", "all")

    if STATS_DF.empty:
        return jsonify({"sites": []})

    results = []
    for site in sites:
        sub = STATS_DF[
            (STATS_DF["Site"] == site) & (STATS_DF["Scenario"] == scenario)
        ]
        if "class" in sub.columns:
            if str(class_filter) == "0":
                sub = sub[sub["class"] == 0]
            elif str(class_filter) == "1":
                sub = sub[sub["class"] == 1]
        spad_values = sub[x_var].dropna()
        if len(spad_values) < 2:
            continue
        mu, sigma = norm.fit(spad_values)
        x = np.linspace(spad_values.min(), spad_values.max(), 200)
        y = norm.pdf(x, mu, sigma)
        results.append({
            "site": site, "x": x.tolist(), "y": y.tolist(), "mu": mu, "sigma": sigma,
        })

    return jsonify({"sites": results})


@app.route("/api/scatter/export")
def api_scatter_export():
    sites = _parse_sites_param()
    scenario = request.args.get("scenario", "")
    index_key = request.args.get("index", "rb_index")
    stat_key = request.args.get("stat", DEFAULT_STAT)
    exclude_spad_outliers = request.args.get("exclude_spad_outliers", "0") == "1"
    exclude_index_outliers = request.args.get("exclude_index_outliers", "0") == "1"
    class_filter = request.args.get("class_filter", "all")
    try:
        zenith_clusters = int(request.args.get("zenith_clusters", "0"))
    except ValueError:
        zenith_clusters = 0

    rows_used, _, _, _, _, _, _, _, _ = _scatter_data(
        sites, scenario, index_key, stat_key, exclude_spad_outliers, exclude_index_outliers, zenith_clusters, class_filter
    )

    if STATS_DF.empty:
        rows_used = pd.DataFrame(columns=ORIGINAL_COLUMNS)

    export_cols = ORIGINAL_COLUMNS.copy()
    if "zenith_cluster" in rows_used.columns:
        export_cols.append("zenith_cluster")
    csv_data = rows_used[export_cols].to_csv(index=False)

    suffix = ""
    if exclude_spad_outliers:
        suffix += "_sans_aberrants_spad"
    if exclude_index_outliers:
        suffix += "_sans_aberrants_indice"
    if str(class_filter) in ("0", "1"):
        suffix += f"_class{class_filter}"
    filename = f"{'+'.join(sites)}_{scenario}{suffix}.csv"
    return app.response_class(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/img/rgb/<path:site>/<name>")
def img_rgb(site, name):
    site_dir = get_name_path(site, name)
    path = site_dir / "images_rgb_rect" / f"{name}.png"
    if not path.is_file():
        abort(404)
    return send_file(path)


@app.route("/img/metadata/<path:site>/<name>")
def img_metadata(site, name):
    site_dir = get_name_path(site, name)
    path = site_dir / "metadata" / f"{name}_qpcard_warped.png"
    if not path.is_file():
        abort(404)
    return send_file(path)


@app.route("/img/regression/<path:site>/<name>")
def img_regression(site, name):
    site_dir = get_name_path(site, name)
    path = site_dir / "metadata" / f"{name}_regression.png"
    if not path.is_file():
        abort(404)
    return send_file(path)


@app.route("/img/mask_combined/<path:site>/<name>")
def img_mask_combined(site, name):
    site_dir = get_name_path(site, name)
    path = site_dir / "masks" / "combined" / f"{name}.png"
    if not path.is_file():
        abort(404)
    return send_file(path)


@lru_cache(maxsize=8)
def _index_map_2d(site: str, name: str, index_key: str) -> np.ndarray:
    """Carte 2D (H, W) de la valeur de l'indice, NaN hors du masque combiné.
    Recalculee avec les cinq formules du pipeline puis mise en cache pour les
    huit dernieres combinaisons site/image/indice."""
    site_dir = get_site_dir(site)
    bands = load_reflectance_bands(site_dir, name)
    norm_bands = normalize_bands(bands)
    indices = compute_indices(norm_bands)
    if index_key not in indices:
        abort(404, description=f"Indice inconnu: {index_key}")

    mask_path = site_dir / "masks" / "combined" / f"{name}.png"
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        abort(404, description="Masque combiné introuvable pour cette image")
    mask = mask_img > 0

    arr = indices[index_key].astype(np.float32)
    return np.where(mask, arr, np.nan)


@app.route("/img/index_heatmap/<path:site>/<name>")
def img_index_heatmap(site, name):
    get_name_path(site, name)
    index_key = request.args.get("index", "rb_index")
    if index_key not in HEATMAP_INDEX_KEYS:
        abort(404, description=f"Indice inconnu: {index_key}")

    arr = _index_map_2d(site, name, index_key)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        abort(404, description="Aucun pixel valide sous le masque combiné")

    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max()) + 1e-6
    normed = np.clip((arr - lo) / (hi - lo), 0, 1)
    gray = (np.nan_to_num(normed, nan=0.0) * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)  # BGR, uint8
    alpha = np.where(np.isfinite(arr), 255, 0).astype(np.uint8)
    bgra = np.dstack([color, alpha])

    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        abort(500, description="Échec de l'encodage PNG")
    return app.response_class(buf.tobytes(), mimetype="image/png")


@app.route("/api/pixel_value/<path:site>/<name>")
def api_pixel_value(site, name):
    get_name_path(site, name)
    index_key = request.args.get("index", "rb_index")
    if index_key not in HEATMAP_INDEX_KEYS:
        abort(404, description=f"Indice inconnu: {index_key}")
    try:
        x = int(request.args.get("x", -1))
        y = int(request.args.get("y", -1))
    except ValueError:
        abort(400, description="Coordonnées x/y invalides")

    arr = _index_map_2d(site, name, index_key)
    h, w = arr.shape
    if not (0 <= x < w and 0 <= y < h):
        return jsonify({"value": None, "width": w, "height": h})

    v = arr[y, x]
    return jsonify({"value": None if not np.isfinite(v) else float(v), "width": w, "height": h})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

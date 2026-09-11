"""
Configuration centrale du projet reflectance_qpcard.
Modifier ce fichier pour adapter les chemins et paramètres.
"""
from pathlib import Path

# ── Racine du projet ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
YAML_PATH    = DATA_DIR / "reflectances_qpcard.yaml"

# ── Dossier d'entrée par défaut (test/) ──────────────────────────────────────
TEST_DIR = PROJECT_ROOT.parent / "data" / "for_spad_iveuro"

# ── Dossier de sortie par défaut ──────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "results_calib"

# ── Modèle QPCard mask predictor ─────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT.parent / "models" / "qpcard_final.pt"

# ── Gabarit QPCard 101 (espace template 500×140 px) ──────────────────────────
QPCARD_W = 500
QPCARD_H = 140

PATCH_CENTERS = {
    "noir":  (124, 70),
    "gris":  (272, 70),
    "blanc": (424, 70),
}
PATCH_ORDER = ["noir", "gris", "blanc"]
PATCH_SIZE  = (135, 135)   # utilisé par QPCard (target.py)

HALF = 40          # demi-fenêtre extraction en pixels (espace template)

# ── Seuils de saturation ──────────────────────────────────────────────────────
SAT_LOW  = 1      # pixels ≤ sat_low exclus (bouchés)
SAT_HIGH = 250    # pixels ≥ sat_high exclus (saturés) — 8 bits
MIN_VALID_PX = 10 # pixels valides minimum pour une médiane fiable

# ── Couleurs pour les plots ───────────────────────────────────────────────────
BAND_COLORS  = {"R": "#e63946", "G": "#2a9d8f", "B": "#457b9d", "NIR": "#9b5de5"}
PATCH_COLORS = {"noir": "#2c2c2c", "gris": "#888888", "blanc": "#f0f0f0"}

BANDS = ["R", "G", "B", "NIR"]

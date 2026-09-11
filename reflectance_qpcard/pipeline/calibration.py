"""
Régression ELM (Empirical Line Method) DN → Réflectance et application aux images.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from config import BANDS, PATCH_ORDER, SAT_HIGH


def fit_elm(
    dn_vals:   list,
    refl_vals: list,
    force_zero: bool = True,
) -> dict:
    """
    Ajuste un modèle linéaire Réflectance = a·DN + b.
    Les NaN sont filtrés ; les contraintes sont vérifiées par fit_all_bands.
    """
    raw_dn   = np.array(dn_vals,   dtype=float)
    raw_refl = np.array(refl_vals, dtype=float)

    valid      = ~np.isnan(raw_dn)
    dn_clean   = raw_dn[valid].reshape(-1, 1)
    refl_clean = raw_refl[valid]

    model = LinearRegression(fit_intercept=not force_zero)
    model.fit(dn_clean, refl_clean)

    pred      = model.predict(dn_clean)
    slope     = float(model.coef_[0])
    intercept = 0.0 if force_zero else float(model.intercept_)
    r2        = float(r2_score(refl_clean, pred))
    rmse      = float(np.sqrt(mean_squared_error(refl_clean, pred)))

    return dict(
        slope=slope, intercept=intercept,
        r2=r2, rmse=rmse, n=int(valid.sum()),
        dn=dn_clean.ravel(), refl=refl_clean, pred=pred,
    )


def fit_all_bands(
    dn_record:    dict,
    refl_ref:     dict,
    force_zero:   bool = True,
    n_patches_rgb: int = 2,
    mode:          str = "RGB+NIR",
) -> tuple:
    """
    Vérifie les contraintes et ajuste le modèle ELM pour les bandes disponibles.

    n_patches_rgb : nombre de patches utilisés pour toutes les bandes, y compris NIR
                    (1=noir, 2=noir+gris, 3=tous). Par défaut 2 (noir+gris) : le
                    patch blanc sature souvent sur les 4 bandes et est exclu.
    mode          : "RGB+NIR" (défaut) ou "RGB" (ignore la bande NIR).

    Retourne (results, violations) :
        results    : dict band → résultat fit_elm (ou None si contrainte violée)
        violations : list de messages décrivant chaque contrainte violée
    """
    results    = {}
    violations = []

    active_bands = [b for b in BANDS if b != "NIR" or mode == "RGB+NIR"]
    for band in active_bands:
        n_use  = 3 if band == "NIR" else n_patches_rgb
        raw_dn = np.array(dn_record[band][:n_use], dtype=float)

        # Si des DN bruts (<= 255) sont passés directement, exclure les patchs saturés (DN >= SAT_HIGH)
        valid_finite = raw_dn[~np.isnan(raw_dn)]
        if len(valid_finite) > 0 and np.max(valid_finite) <= 255.0:
            raw_dn[raw_dn >= SAT_HIGH] = np.nan

        n_valid = int(np.sum(~np.isnan(raw_dn)))

        if n_valid < 2:
            violations.append(
                f"  [{band}] Seulement {n_valid}/3 patchs valides — minimum 2 requis"
            )
            results[band] = None
            continue

        valid_mask = ~np.isnan(raw_dn)
        valid_dn   = raw_dn[valid_mask]
        if not np.all(np.diff(valid_dn) > 0):
            labels     = np.array(["noir", "gris", "blanc"])[:n_use][valid_mask]
            dn_str     = "  ".join(f"DN({l})={d:.1f}" for l, d in zip(labels, valid_dn))
            violations.append(
                f"  [{band}] Ordre physique invalide — {dn_str} "
                f"(attendu : noir < gris < blanc)"
            )
            results[band] = None
            continue

        results[band] = fit_elm(
            dn_record[band][:n_use], refl_ref[band][:n_use], force_zero
        )

    return results, violations


def apply_calibration(
    img:         np.ndarray,
    band_models: dict,
    mode:        str = "RGB",
) -> np.ndarray:
    """
    Applique les modèles ELM à une image entière.

    mode = "RGB" : img est BGR uint8/uint16, shape (H,W,3)
    mode = "NIR" : img est grayscale, shape (H,W)

    Retourne une image float32 [0,1] (non clippée).
    """
    out = img.astype(np.float32).copy()

    if mode == "RGB":
        for ch, idx in [("B", 0), ("G", 1), ("R", 2)]:
            m = band_models.get(ch)
            if m is None:
                continue
            out[:, :, idx] = m["slope"] * out[:, :, idx] + m["intercept"]
    else:
        m = band_models.get("NIR")
        if m is not None:
            out = m["slope"] * out + m["intercept"]

    return out

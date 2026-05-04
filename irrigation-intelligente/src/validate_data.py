import pandas as pd
import yaml
import sys
import os
import logging
import json

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# RÈGLES DE VALIDATION MÉTIER
# ─────────────────────────────────────────────
RANGE_RULES = {
   
    "Humidity":             (0.0,  100.0),
    "Temperature_C":        (-50.0, 60.0),
    "Rainfall_mm":          (0.0,  2000.0),
    "Wind_Speed_kmh":       (0.0,  300.0),
    "Field_Area_hectare":   (0.0,  100000.0),
    "Previous_Irrigation_mm":  (0.0, 5000.0),
}

EXPECTED_COLUMNS = [
    'Temperature_C', 'Humidity', 'Rainfall_mm',
    'Wind_Speed_kmh', 'Crop_Type', 'Crop_Growth_Stage',
    'Season', 'Irrigation_Type', 'Field_Area_hectare',
    'Mulching_Used', 'Previous_Irrigation_mm', 'Region', 'Irrigation_Need'
]

NUMERIC_COLS = [
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────
def validate_data(config_path="params.yaml"):

    errors   = []   # bloquants  → sys.exit(1)
    warnings = []   # non-bloquants

    # 0. Charger la config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    file_path = config['data']['raw_path']

    logger.info(f" Validation du fichier : {file_path}")

    # ── CHECK 1 : existence du fichier ────────────────────
    if not os.path.exists(file_path):
        logger.error(f" Fichier introuvable : {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    logger.info(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ── CHECK 2 : colonnes attendues ──────────────────────
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_cols:
        errors.append(f"Colonnes manquantes : {missing_cols}")
    if extra_cols:
        warnings.append(f"Colonnes supplémentaires (ignorées) : {extra_cols}")

    # ── CHECK 3 : types numériques ────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Type incorrect : '{col}' doit être numérique")

    # ── CHECK 4 : valeurs manquantes ──────────────────────
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        total_null = null_counts.sum()
        null_pct   = round(total_null / (df.shape[0] * df.shape[1]) * 100, 2)
        if null_pct > 10:
            errors.append(f"Trop de valeurs manquantes : {total_null} ({null_pct}%)")
        else:
            warnings.append(f"Valeurs manquantes (< 10%) : {null_cols.to_dict()}")

    # ── CHECK 5 : doublons ────────────────────────────────
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        dup_pct = round(n_duplicates / len(df) * 100, 2)
        warnings.append(f"Doublons détectés : {n_duplicates} lignes ({dup_pct}%)")

    # ── CHECK 6 : plages physiques métier ─────────────────
    for col, (min_val, max_val) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        out_of_range = (~df[col].between(min_val, max_val)).sum()
        if out_of_range > 0:
            pct = round(out_of_range / len(df) * 100, 2)
            msg = f"'{col}' : {out_of_range} valeurs hors [{min_val}, {max_val}] ({pct}%)"
            if pct > 5:
                errors.append(msg)
            else:
                warnings.append(msg)

    # ── CHECK 7 : distribution de la cible ────────────────
    if 'Irrigation_Need' in df.columns:
        dist = df['Irrigation_Need'].value_counts(normalize=True) * 100
        logger.info(f"Distribution cible :\n{dist.round(2).to_string()}")
        min_class_pct = dist.min()
        if min_class_pct < 5:
            warnings.append(
                f"Déséquilibre de classes : classe minoritaire = {min_class_pct:.1f}%. "
                "Considérer SMOTE ou class_weight='balanced'."
            )

    # ── CHECK 8 : taille minimale du dataset ──────────────
    if len(df) < 100:
        errors.append(f"Dataset trop petit : {len(df)} lignes (minimum requis : 100)")

    # ─────────────────────────────────────────────
    # RAPPORT FINAL
    # ─────────────────────────────────────────────
    report = {
        "file":          file_path,
        "rows":          int(df.shape[0]),
        "columns":       int(df.shape[1]),
        "nulls_total":   int(null_counts.sum()),
        "duplicates":    int(n_duplicates),
        "errors":        errors,
        "warnings":      warnings,
        "status":        "FAILED" if errors else "PASSED"
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    for w in warnings:
        logger.warning(f"  {w}")

    if errors:
        for e in errors:
            logger.error(f"{e}")
        logger.error(" Validation échouée. Pipeline arrêté.")
        sys.exit(1)

    logger.info(f" Validation réussie — {len(warnings)} avertissement(s), 0 erreur.")
    return report


if __name__ == "__main__":
    validate_data("params.yaml")
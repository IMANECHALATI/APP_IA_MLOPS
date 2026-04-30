import pandas as pd
import yaml
import os
import joblib
import sys
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
    'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
]

NUMERIC_COLS = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────
def preprocess(config_path="params.yaml"):

    # 0. Config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("⚙️  Début du preprocessing...")

    # 1. Chargement
    df = pd.read_csv(config['data']['raw_path'])
    df.columns = df.columns.str.strip()
    logger.info(f"📂 Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # 2. Vérification colonnes
    all_cols = CATEGORICAL_COLS + NUMERIC_COLS + ['Irrigation_Need']
    missing  = [c for c in all_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Colonnes manquantes : {missing}")
        sys.exit(1)

    # 3. Suppression doublons
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped_dup = n_before - len(df)
    if n_dropped_dup > 0:
        logger.warning(f"⚠️  {n_dropped_dup} doublons supprimés")

    # 4. Suppression NaN
    n_before = len(df)
    df = df.dropna(subset=all_cols)
    n_dropped_na = n_before - len(df)
    if n_dropped_na > 0:
        logger.warning(f"⚠️  {n_dropped_na} lignes supprimées (NaN)")

    logger.info(f"📊 Dataset propre : {len(df)} lignes restantes")

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 5. Encodage catégoriel
    logger.info("🔤 Encodage des variables catégorielles...")
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, f"models/encoder_{col}.pkl")
        logger.info(f"   ✅ {col} → {list(le.classes_)}")

    # 6. Normalisation numérique
    logger.info("📏 Normalisation des variables numériques...")
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    joblib.dump(scaler, "models/scaler.pkl")

    # 7. Encodage cible
    target_le = LabelEncoder()
    df['Irrigation_Need'] = target_le.fit_transform(df['Irrigation_Need'])
    joblib.dump(target_le, "models/target_encoder.pkl")
    logger.info(f"🎯 Classes cible : {list(target_le.classes_)}")

    # 8. Split — ✅ stratify=y pour équilibrer les classes
    X = df.drop('Irrigation_Need', axis=1)
    y = df['Irrigation_Need']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['base']['random_state'],
        stratify=y                         # ← CRITIQUE : garantit la distribution des classes
    )

    # 9. Sauvegarde CSV
    X_train.to_csv(config['data']['train_X_path'], index=False)
    X_test.to_csv(config['data']['test_X_path'],   index=False)
    y_train.to_csv(config['data']['train_y_path'], index=False)
    y_test.to_csv(config['data']['test_y_path'],   index=False)

    # 10. Rapport preprocessing
    report = {
        "total_rows_raw":       n_before + n_dropped_dup,
        "duplicates_removed":   n_dropped_dup,
        "nulls_removed":        n_dropped_na,
        "rows_final":           len(df),
        "train_size":           len(X_train),
        "test_size":            len(X_test),
        "n_features":           X.shape[1],
        "target_classes":       list(target_le.classes_),
        "class_distribution": {
            "train": y_train.value_counts(normalize=True).round(3).to_dict(),
            "test":  y_test.value_counts(normalize=True).round(3).to_dict()
        },
        "artifacts_saved": [
            f"models/encoder_{col}.pkl" for col in CATEGORICAL_COLS
        ] + ["models/scaler.pkl", "models/target_encoder.pkl"]
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info("─" * 50)
    logger.info(f"✅ Preprocessing terminé !")
    logger.info(f"   Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
    logger.info(f"   Features : {X.shape[1]} | Classes : {list(target_le.classes_)}")
    logger.info(f"   Artefacts : {len(CATEGORICAL_COLS) + 2} fichiers .pkl sauvegardés")

    return report


if __name__ == "__main__":
    preprocess("params.yaml")
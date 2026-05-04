import pandas as pd
import yaml
import os
import joblib
import sys
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION DES COLONNES
# ─────────────────────────────────────────────
# 1. Les colonnes à ignorer et supprimer (issues de l'analyse de validation)
COLS_TO_DROP = [
    'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 
    'Electrical_Conductivity', 'Sunlight_Hours', 'Water_Source'
]

# 2. Les colonnes conservées
CATEGORICAL_COLS = [
    'Crop_Type', 'Crop_Growth_Stage', 'Season', 
    'Irrigation_Type', 'Mulching_Used', 'Region'
]

NUMERIC_COLS = [
    'Temperature_C', 'Humidity', 'Rainfall_mm', 
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────
def preprocess(config_path="params.yaml"):

    # 0. Config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(" Début du preprocessing...")

    # 1. Chargement
    df = pd.read_csv(config['data']['raw_path'])
    df.columns = df.columns.str.strip()
    logger.info(f"Dataset initial : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # 2. NETTOYAGE : Suppression des colonnes ignorées
    cols_to_drop_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop_present)
    logger.info(f" Colonnes supprimées : {cols_to_drop_present}")

    # 3. NETTOYAGE : Traitement des valeurs aberrantes (Outliers)
    # Au lieu de supprimer 20% des données, on "plafonne" (clip) la pluie à 2000 mm.
    # Tout ce qui est > 2000 devient 2000.
    if 'Rainfall_mm' in df.columns:
        n_outliers = (df['Rainfall_mm'] > 2000.0).sum()
        df['Rainfall_mm'] = df['Rainfall_mm'].clip(upper=2000.0)
        if n_outliers > 0:
            logger.info(f" {n_outliers} valeurs de 'Rainfall_mm' ont été plafonnées à 2000.0 mm")

    # 4. Suppression doublons et NaN
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped_dup = n_before - len(df)
    
    n_before_nan = len(df)
    df = df.dropna()
    n_dropped_na = n_before_nan - len(df)
    
    if n_dropped_dup > 0 or n_dropped_na > 0:
        logger.warning(f" Nettoyage : {n_dropped_dup} doublons et {n_dropped_na} lignes avec NaN supprimés.")

    logger.info(f" Dataset propre avant encodage : {len(df)} lignes restantes")

    # Création des dossiers
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 5. Séparation X (Features) et y (Cible)
    X = df.drop('Irrigation_Need', axis=1)
    y = df['Irrigation_Need']

    # 6. Encodage de la Cible (y)
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    joblib.dump(target_le, "models/target_encoder.pkl")
    logger.info(f" Classes cible encodées : {list(target_le.classes_)}")

    # 7. SPLIT (Crucial : on divise AVANT de transformer pour éviter la fuite de données)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config['data']['test_size'],
        random_state=config['base']['random_state'],
        stratify=y_encoded  # Maintient l'équilibre des classes
    )

    # 8. PREPROCESSING : Normalisation Numérique (StandardScaler)
    logger.info(" Normalisation des variables numériques...")
    scaler = StandardScaler()
    
    # On fit uniquement sur le train
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[NUMERIC_COLS]), columns=NUMERIC_COLS, index=X_train.index)
    X_test_num  = pd.DataFrame(scaler.transform(X_test[NUMERIC_COLS]), columns=NUMERIC_COLS, index=X_test.index)
    joblib.dump(scaler, "models/scaler.pkl")

    # 9. PREPROCESSING : Encodage Catégoriel (OneHotEncoder)
    logger.info(" Encodage des variables catégorielles (OneHot)...")
    # sparse_output=False renvoie un tableau numpy classique au lieu d'une matrice creuse
    # handle_unknown='ignore' empêche le modèle de planter si une nouvelle catégorie apparaît en production
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # On fit uniquement sur le train
    X_train_cat_array = ohe.fit_transform(X_train[CATEGORICAL_COLS])
    X_test_cat_array  = ohe.transform(X_test[CATEGORICAL_COLS])
    
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLS)
    
    X_train_cat = pd.DataFrame(X_train_cat_array, columns=cat_feature_names, index=X_train.index)
    X_test_cat  = pd.DataFrame(X_test_cat_array, columns=cat_feature_names, index=X_test.index)
    joblib.dump(ohe, "models/onehot_encoder.pkl")

    # 10. Recombinaison des colonnes numériques et catégorielles
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final  = pd.concat([X_test_num, X_test_cat], axis=1)


    # 11. Sauvegarde CSV
    X_train_final.to_csv(config['data']['train_X_path'], index=False)
    X_test_final.to_csv(config['data']['test_X_path'],   index=False)
    
    # y_train et y_test sont des arrays numpy, on les convertit en DataFrame pour la sauvegarde
    pd.DataFrame(y_train, columns=['Irrigation_Need']).to_csv(config['data']['train_y_path'], index=False)
    pd.DataFrame(y_test, columns=['Irrigation_Need']).to_csv(config['data']['test_y_path'],  index=False)

    # 12. Rapport preprocessing
    report = {
        "total_rows_raw":       n_before + n_dropped_dup,
        "duplicates_removed":   n_dropped_dup,
        "nulls_removed":        n_dropped_na,
        "outliers_capped":      int(n_outliers) if 'Rainfall_mm' in df.columns else 0,
        "rows_final":           len(df),
        "train_size":           len(X_train_final),
        "test_size":            len(X_test_final),
        "n_features_original":  len(NUMERIC_COLS) + len(CATEGORICAL_COLS),
        "n_features_final":     X_train_final.shape[1], # Plus grand à cause du OneHotEncoding
        "target_classes":       list(target_le.classes_)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info("─" * 50)
    logger.info(f" Preprocessing terminé avec succès !")
    logger.info(f"  Train : {len(X_train_final)} lignes | Test : {len(X_test_final)} lignes")
    logger.info(f"  Features (après encodage) : {X_train_final.shape[1]} colonnes")
    logger.info("─" * 50)

    return report

if __name__ == "__main__":
    preprocess("params.yaml")

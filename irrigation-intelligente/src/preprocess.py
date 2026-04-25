import pandas as pd
import yaml
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
        
    df = pd.read_csv(config['data']['raw_path'])
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    
    # LISTE EXACTE BASÉE SUR TON DERNIER MESSAGE D'ERREUR
    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
    ]
    
    numeric_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours', 
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
    ]

    # Vérification de sécurité
    for col in categorical_cols + numeric_cols:
        if col not in df.columns:
            print(f" Erreur critique : La colonne '{col}' est introuvable.")
            sys.exit(1)

    df = df.dropna()
    os.makedirs('models', exist_ok=True)

    # 1. Encodage Catégoriel (LabelEncoding)
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, f"models/encoder_{col}.pkl")

    # 2. Normalisation Numérique (Scaling)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, "models/scaler.pkl")

    # 3. Encodage de la cible
    target_le = LabelEncoder()
    df['Irrigation_Need'] = target_le.fit_transform(df['Irrigation_Need'])
    joblib.dump(target_le, "models/target_encoder.pkl")

    # 4. Split et Sauvegarde
    X = df.drop('Irrigation_Need', axis=1)
    y = df['Irrigation_Need']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['base']['random_state']
    )
    
    os.makedirs('data', exist_ok=True)
    X_train.to_csv("data/train_X.csv", index=False)
    X_test.to_csv("data/test_X.csv", index=False)
    y_train.to_csv("data/train_y.csv", index=False)
    y_test.to_csv("data/test_y.csv", index=False)
    
    print(" Preprocessing terminé sans erreurs !")
    print(f"   Fichiers générés : {len(categorical_cols)} encodeurs, 1 scaler, et 4 fichiers CSV.")

if __name__ == "__main__":
    preprocess("params.yaml")
import pandas as pd
import yaml
import sys
import os

def validate_data(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    file_path = config['data']['raw_path']
    
    if not os.path.exists(file_path):
        print(f" Erreur : Le fichier {file_path} est introuvable.")
        sys.exit(1)
        
    df = pd.read_csv(file_path)

    # Liste synchronisée avec les colonnes réelles de ton CSV
    expected_columns = [
        'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh', 
        'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Water_Source', 
        'Field_Area_hectare', 'Mulching_Used', 'Previous_Irrigation_mm', 'Region', 'Irrigation_Need'
    ]

    # 1. Vérification de la présence des colonnes
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f" Erreur : Colonnes manquantes dans le dataset : {missing_cols}")
        sys.exit(1)

    # 2. Validation des types numériques (DataOps)
    numeric_to_check = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Humidity']
    for col in numeric_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f" Erreur : La colonne {col} doit être numérique.")
            sys.exit(1)

    # 3. Validation des plages physiques (Business Logic)
    if not df['Soil_pH'].between(0, 14).all():
        print(" Attention : Valeurs de Soil_pH hors de la plage standard [0, 14].")

    if (df['Soil_Moisture'] < 0).any():
        print(" Erreur : Soil_Moisture négative détectée.")
        sys.exit(1)

    print(" Validation DataOps réussie : Le schéma est prêt pour le pipeline.")

if __name__ == "__main__":
    validate_data("params.yaml")
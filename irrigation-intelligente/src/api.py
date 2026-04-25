from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel


app = FastAPI(title="Système d'Irrigation Intelligente")

# 1. Charger tous les artéfacts sauvegardés
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
target_le = joblib.load("models/target_encoder.pkl")

# On charge aussi les encodeurs pour les variables texte
categorical_cols = [
    'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
    'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
]
encoders = {col: joblib.load(f"models/encoder_{col}.pkl") for col in categorical_cols}

# 2. Définir le format des données d'entrée (Schéma Pydantic)
class IrrigationData(BaseModel):
    Soil_Type: str
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Temperature_C: float
    Humidity: float
    Rainfall_mm: float
    Sunlight_Hours: float
    Wind_Speed_kmh: float
    Crop_Type: str
    Crop_Growth_Stage: str
    Season: str
    Irrigation_Type: str
    Water_Source: str
    Field_Area_hectare: float
    Mulching_Used: str
    Previous_Irrigation_mm: float
    Region: str

@app.post("/predict")
def predict_irrigation(data: IrrigationData):
    # Convertir l'entrée en DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # A. Encodage des textes
    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])
        
    # B. Scaling des nombres
    numeric_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours', 
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
    ]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # C. Prédiction
    pred_code = model.predict(input_df)[0]
    
    # D. Traduction du résultat (Chiffre -> Label)
    prediction_label = target_le.inverse_transform([pred_code])[0]
    
    return {
        "prediction_code": int(pred_code),
        "irrigation_need": prediction_label
    }

@app.get("/")
def home():
    return {"status": "L'API d'irrigation est en ligne !"}
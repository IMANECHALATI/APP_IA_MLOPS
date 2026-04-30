from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging
import os
from pydantic import BaseModel, Field
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION DU LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log") if os.path.exists("logs") else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# INITIALISATION FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="🌿 Système d'Irrigation Intelligente",
    description="API MLOps pour la prédiction des besoins en irrigation agricole",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (pour autoriser le dashboard Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS ML
# ─────────────────────────────────────────────
try:
    model      = joblib.load("models/model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    target_le  = joblib.load("models/target_encoder.pkl")

    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
    ]
    encoders = {col: joblib.load(f"models/encoder_{col}.pkl") for col in categorical_cols}

    logger.info(" Tous les artefacts ML chargés avec succès.")

except FileNotFoundError as e:
    logger.error(f" Artefact ML introuvable : {e}")
    raise RuntimeError(f"Impossible de charger les artefacts ML : {e}")

# Colonnes numériques à scaler
numeric_cols = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

# Compteur de prédictions (monitoring simple)
prediction_counter = {"total": 0, "errors": 0}

# ─────────────────────────────────────────────
# SCHÉMA D'ENTRÉE (PYDANTIC)
# ─────────────────────────────────────────────
class IrrigationData(BaseModel):
    Soil_Type:               str   = Field(..., example="Sandy")
    Soil_pH:                 float = Field(..., ge=0.0, le=14.0, example=6.5)
    Soil_Moisture:           float = Field(..., ge=0.0, example=35.2)
    Organic_Carbon:          float = Field(..., ge=0.0, example=1.5)
    Electrical_Conductivity: float = Field(..., ge=0.0, example=0.8)
    Temperature_C:           float = Field(..., example=28.0)
    Humidity:                float = Field(..., ge=0.0, le=100.0, example=65.0)
    Rainfall_mm:             float = Field(..., ge=0.0, example=12.0)
    Sunlight_Hours:          float = Field(..., ge=0.0, le=24.0, example=8.5)
    Wind_Speed_kmh:          float = Field(..., ge=0.0, example=15.0)
    Crop_Type:               str   = Field(..., example="Wheat")
    Crop_Growth_Stage:       str   = Field(..., example="Vegetative")
    Season:                  str   = Field(..., example="Summer")
    Irrigation_Type:         str   = Field(..., example="Drip")
    Water_Source:            str   = Field(..., example="Groundwater")
    Field_Area_hectare:      float = Field(..., gt=0.0, example=2.5)
    Mulching_Used:           str   = Field(..., example="Yes")
    Previous_Irrigation_mm:  float = Field(..., ge=0.0, example=20.0)
    Region:                  str   = Field(..., example="North")

# ─────────────────────────────────────────────
# SCHÉMA DE RÉPONSE
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction_code:  int
    irrigation_need:  str
    confidence:       float
    timestamp:        str
    model_version:    str

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Status"])
def home():
    return {
        "status": "online",
        "message": "🌿 API d'irrigation intelligente en ligne",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Status"])
def health():
    """Vérifie que l'API et les modèles sont opérationnels."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": "1.0.0",
        "total_predictions": prediction_counter["total"],
        "total_errors": prediction_counter["errors"],
        "uptime": datetime.now().isoformat()
    }


@app.get("/model-info", tags=["MLOps"])
def model_info():
    """Retourne les informations sur le modèle en production."""
    return {
        "model_type":        type(model).__name__,
        "n_estimators":      getattr(model, "n_estimators", "N/A"),
        "max_depth":         getattr(model, "max_depth", "N/A"),
        "n_features":        getattr(model, "n_features_in_", "N/A"),
        "classes":           list(target_le.classes_),
        "categorical_cols":  categorical_cols,
        "numeric_cols":      numeric_cols,
        "version":           "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_irrigation(data: IrrigationData):
    """
    Prédit le besoin en irrigation à partir des données agronomiques et météo.
    Retourne le label, le code et le score de confiance.
    """
    prediction_counter["total"] += 1

    try:
        # A. Convertir en DataFrame
        input_df = pd.DataFrame([data.dict()])

        # B. Encodage des variables catégorielles
        for col in categorical_cols:
            if data.dict()[col] not in encoders[col].classes_:
                raise HTTPException(
                    status_code=422,
                    detail=f"Valeur inconnue pour '{col}': '{data.dict()[col]}'. "
                           f"Valeurs autorisées : {list(encoders[col].classes_)}"
                )
            input_df[col] = encoders[col].transform(input_df[col])

        # C. Scaling des colonnes numériques
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # D. Prédiction + Confiance
        pred_code   = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        confidence  = round(float(proba.max()), 4)

        # E. Décodage du label
        prediction_label = target_le.inverse_transform([pred_code])[0]

        # F. Logging
        logger.info(
            f" Prédiction | Label: {prediction_label} | "
            f"Confiance: {confidence:.2%} | "
            f"Région: {data.Region} | Crop: {data.Crop_Type}"
        )

        return PredictionResponse(
            prediction_code=int(pred_code),
            irrigation_need=prediction_label,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )

    except HTTPException:
        raise

    except Exception as e:
        prediction_counter["errors"] += 1
        logger.error(f" Erreur de prédiction : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de la prédiction : {str(e)}"
        )


@app.get("/metrics", tags=["MLOps"])
def get_metrics():
    """Endpoint de monitoring : statistiques d'utilisation de l'API."""
    total  = prediction_counter["total"]
    errors = prediction_counter["errors"]
    return {
        "total_predictions": total,
        "total_errors":      errors,
        "success_rate":      round((total - errors) / total, 4) if total > 0 else 1.0,
        "error_rate":        round(errors / total, 4) if total > 0 else 0.0,
        "timestamp":         datetime.now().isoformat()
    }
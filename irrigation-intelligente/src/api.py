from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging
import os
from pydantic import BaseModel, Field
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log") if os.path.exists("logs") else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Système d'Irrigation Intelligente",
    description="API MLOps pour la prédiction des besoins en irrigation agricole",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
    model     = joblib.load("models/model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    target_le = joblib.load("models/target_encoder.pkl")
    ohe       = joblib.load("models/onehot_encoder.pkl")  # ✅ OneHotEncoder

    logger.info("✅ Tous les artefacts ML chargés avec succès.")

except FileNotFoundError as e:
    logger.error(f"❌ Artefact ML introuvable : {e}")
    raise RuntimeError(f"Impossible de charger les artefacts ML : {e}")

# ✅ Cohérent avec preprocess.py
CATEGORICAL_COLS = [
    'Crop_Type', 'Crop_Growth_Stage', 'Season',
    'Irrigation_Type', 'Mulching_Used', 'Region'
]

NUMERIC_COLS = [
    'Temperature_C', 'Humidity', 'Rainfall_mm',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

prediction_counter = {"total": 0, "errors": 0}

# ─────────────────────────────────────────────
# SCHÉMA D'ENTRÉE
# ─────────────────────────────────────────────
class IrrigationData(BaseModel):
    Temperature_C:           float = Field(..., json_schema_extra={"example": 28.0})
    Humidity:                float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 65.0})
    Rainfall_mm:             float = Field(..., ge=0.0, json_schema_extra={"example": 12.0})
    Wind_Speed_kmh:          float = Field(..., ge=0.0, json_schema_extra={"example": 15.0})
    Field_Area_hectare:      float = Field(..., gt=0.0, json_schema_extra={"example": 2.5})
    Previous_Irrigation_mm:  float = Field(..., ge=0.0, json_schema_extra={"example": 20.0})
    Crop_Type:               str   = Field(..., json_schema_extra={"example": "Wheat"})      # ✅ ajouté
    Crop_Growth_Stage:       str   = Field(..., json_schema_extra={"example": "Vegetative"})
    Season:                  str   = Field(..., json_schema_extra={"example": "Summer"})
    Irrigation_Type:         str   = Field(..., json_schema_extra={"example": "Drip"})
    Mulching_Used:           str   = Field(..., json_schema_extra={"example": "Yes"})
    Region:                  str   = Field(..., json_schema_extra={"example": "North"})

class PredictionResponse(BaseModel):
    prediction_code: int
    irrigation_need: str
    confidence:      float
    timestamp:       str
    model_version:   str

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
    return {
        "model_type":       type(model).__name__,
        "n_estimators":     getattr(model, "n_estimators", "N/A"),
        "max_depth":        getattr(model, "max_depth", "N/A"),
        "n_features":       getattr(model, "n_features_in_", "N/A"),
        "classes":          list(target_le.classes_),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols":     NUMERIC_COLS,
        "version":          "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_irrigation(data: IrrigationData):
    prediction_counter["total"] += 1
    try:
        # ✅ Pydantic V2
        input_dict = data.model_dump()
        input_df   = pd.DataFrame([input_dict])

        # A. Scaling numérique
        input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

        # B. OneHotEncoding catégoriel ✅ cohérent avec preprocess.py
        cat_array        = ohe.transform(input_df[CATEGORICAL_COLS])
        cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLS)
        cat_df           = pd.DataFrame(cat_array, columns=cat_feature_names)

        # C. Recombinaison
        final_df = pd.concat([
            input_df[NUMERIC_COLS].reset_index(drop=True),
            cat_df.reset_index(drop=True)
        ], axis=1)

        # D. Prédiction
        pred_code  = model.predict(final_df)[0]
        proba      = model.predict_proba(final_df)[0]
        confidence = round(float(proba.max()), 4)

        # E. Décodage
        prediction_label = target_le.inverse_transform([pred_code])[0]

        logger.info(
            f"✅ Prédiction | Label: {prediction_label} | "
            f"Confiance: {confidence:.2%} | Région: {data.Region}"
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
        logger.error(f"❌ Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.get("/metrics", tags=["MLOps"])
def get_metrics():
    total  = prediction_counter["total"]
    errors = prediction_counter["errors"]
    return {
        "total_predictions": total,
        "total_errors":      errors,
        "success_rate":      round((total - errors) / total, 4) if total > 0 else 1.0,
        "error_rate":        round(errors / total, 4) if total > 0 else 0.0,
        "timestamp":         datetime.now().isoformat()
    }
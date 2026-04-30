# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# ─────────────────────────────────────────────
# MOCK DES ARTEFACTS ML (avant l'import de l'app)
# → évite de dépendre des vrais fichiers .pkl en CI/CD
# ─────────────────────────────────────────────
mock_model   = MagicMock()
mock_scaler  = MagicMock()
mock_target  = MagicMock()
mock_encoder = MagicMock()

mock_model.predict.return_value          = np.array([1])
mock_model.predict_proba.return_value    = np.array([[0.1, 0.8, 0.1]])
mock_model.n_estimators                  = 100
mock_model.max_depth                     = 10
mock_model.n_features_in_               = 19
mock_target.inverse_transform.return_value = ["Moderate"]
mock_target.classes_                     = ["High", "Low", "Moderate"]
mock_encoder.transform.side_effect       = lambda x: [0]
mock_encoder.classes_                    = ["Sandy", "Clay", "Loamy",
                                             "Wheat", "Rice", "Maize",
                                             "Vegetative", "Flowering",
                                             "Summer", "Winter", "Spring",
                                             "Drip", "Sprinkler", "Flood",
                                             "Groundwater", "River", "Rainwater",
                                             "Yes", "No",
                                             "North", "South", "East", "West"]

with patch("joblib.load", return_value=mock_model), \
     patch.dict("sys.modules", {}):
    mock_scaler.transform.return_value = np.zeros((1, 11))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────
# DONNÉES DE TEST VALIDES
# ─────────────────────────────────────────────
VALID_PAYLOAD = {
    "Soil_Type":               "Sandy",
    "Soil_pH":                 6.5,
    "Soil_Moisture":           35.2,
    "Organic_Carbon":          1.5,
    "Electrical_Conductivity": 0.8,
    "Temperature_C":           28.0,
    "Humidity":                65.0,
    "Rainfall_mm":             12.0,
    "Sunlight_Hours":          8.5,
    "Wind_Speed_kmh":          15.0,
    "Crop_Type":               "Wheat",
    "Crop_Growth_Stage":       "Vegetative",
    "Season":                  "Summer",
    "Irrigation_Type":         "Drip",
    "Water_Source":            "Groundwater",
    "Field_Area_hectare":      2.5,
    "Mulching_Used":           "Yes",
    "Previous_Irrigation_mm":  20.0,
    "Region":                  "North"
}

# ─────────────────────────────────────────────
# FIXTURE : CLIENT DE TEST
# ─────────────────────────────────────────────
@pytest.fixture
def client():
    """Crée un client de test FastAPI avec tous les artefacts mockés."""
    with patch("joblib.load") as mock_load:
        mock_load.side_effect = lambda path: (
            mock_model   if "model.pkl"          in path else
            mock_scaler  if "scaler.pkl"         in path else
            mock_target  if "target_encoder.pkl" in path else
            mock_encoder
        )
        from src.api import app
        with TestClient(app) as c:
            yield c


# ═════════════════════════════════════════════
# TESTS — ENDPOINTS DE BASE
# ═════════════════════════════════════════════

class TestBaseEndpoints:

    def test_root_returns_200(self, client):
        """GET / doit retourner 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_status(self, client):
        """GET / doit contenir le champ 'status'."""
        response = client.get("/")
        assert "status" in response.json()

    def test_health_returns_200(self, client):
        """GET /health doit retourner 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        """GET /health doit confirmer que le modèle est chargé."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_has_required_fields(self, client):
        """GET /health doit contenir tous les champs attendus."""
        response = client.get("/health")
        data = response.json()
        for field in ["status", "model_loaded", "model_version",
                      "total_predictions", "total_errors"]:
            assert field in data, f"Champ manquant dans /health : '{field}'"

    def test_model_info_returns_200(self, client):
        """GET /model-info doit retourner 200."""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_classes(self, client):
        """GET /model-info doit exposer les classes du modèle."""
        response = client.get("/model-info")
        data = response.json()
        assert "classes" in data
        assert isinstance(data["classes"], list)


# ═════════════════════════════════════════════
# TESTS — ENDPOINT /predict
# ═════════════════════════════════════════════

class TestPredictEndpoint:

    def test_predict_valid_payload_returns_200(self, client):
        """POST /predict avec données valides → 200."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client):
        """POST /predict doit retourner tous les champs du schéma."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        for field in ["prediction_code", "irrigation_need",
                      "confidence", "timestamp", "model_version"]:
            assert field in data, f"Champ manquant dans /predict : '{field}'"

    def test_predict_confidence_between_0_and_1(self, client):
        """La confiance doit être entre 0 et 1."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_predict_code_is_integer(self, client):
        """prediction_code doit être un entier."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(response.json()["prediction_code"], int)

    def test_predict_irrigation_need_is_string(self, client):
        """irrigation_need doit être une chaîne non vide."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        label = response.json()["irrigation_need"]
        assert isinstance(label, str)
        assert len(label) > 0

    def test_predict_missing_field_returns_422(self, client):
        """POST /predict avec un champ manquant → 422."""
        bad_payload = VALID_PAYLOAD.copy()
        del bad_payload["Soil_Type"]
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_invalid_ph_returns_422(self, client):
        """Soil_pH > 14 est physiquement impossible → 422."""
        bad_payload = VALID_PAYLOAD.copy()
        bad_payload["Soil_pH"] = 99.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_negative_moisture_returns_422(self, client):
        """Soil_Moisture négative → 422."""
        bad_payload = VALID_PAYLOAD.copy()
        bad_payload["Soil_Moisture"] = -5.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_negative_area_returns_422(self, client):
        """Field_Area_hectare ≤ 0 → 422."""
        bad_payload = VALID_PAYLOAD.copy()
        bad_payload["Field_Area_hectare"] = 0.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_humidity_over_100_returns_422(self, client):
        """Humidity > 100% est impossible → 422."""
        bad_payload = VALID_PAYLOAD.copy()
        bad_payload["Humidity"] = 150.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        """POST /predict sans body → 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_increments_counter(self, client):
        """Chaque appel à /predict doit incrémenter total_predictions."""
        before = client.get("/metrics").json()["total_predictions"]
        client.post("/predict", json=VALID_PAYLOAD)
        after  = client.get("/metrics").json()["total_predictions"]
        assert after == before + 1


# ═════════════════════════════════════════════
# TESTS — ENDPOINT /metrics
# ═════════════════════════════════════════════

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        """GET /metrics doit retourner 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_required_fields(self, client):
        """GET /metrics doit contenir tous les champs de monitoring."""
        response = client.get("/metrics")
        data = response.json()
        for field in ["total_predictions", "total_errors",
                      "success_rate", "error_rate"]:
            assert field in data, f"Champ manquant dans /metrics : '{field}'"

    def test_metrics_rates_between_0_and_1(self, client):
        """success_rate et error_rate doivent être entre 0 et 1."""
        client.post("/predict", json=VALID_PAYLOAD)
        data = client.get("/metrics").json()
        assert 0.0 <= data["success_rate"] <= 1.0
        assert 0.0 <= data["error_rate"]   <= 1.0

    def test_metrics_rates_sum_to_1(self, client):
        """success_rate + error_rate doit être égal à 1."""
        client.post("/predict", json=VALID_PAYLOAD)
        data = client.get("/metrics").json()
        assert abs(data["success_rate"] + data["error_rate"] - 1.0) < 0.01
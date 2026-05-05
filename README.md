#  AgroFlow — Système d'Irrigation Intelligente

> **Projet Académique S6 — IASD | AppIA & MILOPS**
> Pipeline complet MLOps + DataOps + DevOps pour la prédiction des besoins en irrigation agricole.

---

##  Table des Matières

- [Description](#description)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Lancement](#lancement)
- [API Endpoints](#api-endpoints)
- [Pipeline CI/CD](#pipeline-cicd)
- [Tests](#tests)
- [Structure du Projet](#structure-du-projet)
- [Dashboard](#dashboard)

---

##  Description

**AgroFlow** est un système d'irrigation intelligente basé sur le Machine Learning qui prédit le besoin en irrigation d'une parcelle agricole à partir de données météorologiques et agronomiques.

### Classes de prédiction

| Classe | Signification | Action recommandée |
|--------|--------------|-------------------|
| 🔴 **High** | Besoin critique | Irriguer 35–50 mm immédiatement |
| 🟡 **Medium** | Besoin modéré | Planifier 20–30 mm sous 24h |
| 🟢 **Low** | Niveau optimal | Aucune action requise |

---

##  Architecture

```
Capteurs / Données
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                    DataOps                          │
│  validate_data.py ──► preprocess.py                 │
│  (8 vérifications)    (OneHotEncoder + Scaler)      │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│                    MLOps                            │
│  train.py ──► evaluate.py ──► model.pkl             │
│  (RF + XGBoost)  (Quality Gate 75%)   (MLflow)      │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│                    DevOps                           │
│  FastAPI ──► Docker ──► GitHub Actions              │
│  (6 endpoints)  (3 containers)  (11 étapes CI/CD)   │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
                    Dashboard (React + GPS)
                    http://localhost:80
```

---

##  Technologies

| Catégorie | Technologies |
|-----------|-------------|
| **Langage** | Python 3.10 |
| **ML** | Scikit-learn, XGBoost, RandomForest |
| **MLOps** | MLflow, DVC |
| **API** | FastAPI, Uvicorn, Pydantic V2 |
| **DevOps** | Docker, Docker Compose, GitHub Actions |
| **Tests** | Pytest, HTTPX |
| **Dashboard** | React, HTML5, API wttr.in (météo GPS) |
| **Serveur** | Nginx (dashboard), Uvicorn (API) |

---

##  Installation

### Prérequis

- Python 3.10+
- Docker Desktop
- Git

### 1. Cloner le projet

```bash
git clone https://github.com/IMANECHALATI1/APP_IA_MLOPS.git
cd APP_IA_MLOPS/irrigation-intelligente
```

### 2. Créer un environnement virtuel

```bash
python -m venv env

# Windows
.\env\Scripts\Activate.ps1

# Linux/Mac
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

##  Lancement

### Option 1 — Avec Docker (recommandé)

```bash
# Lance les 3 services en une commande
docker-compose up --build
```

| Service | URL |
|---------|-----|
|  API FastAPI | http://localhost:8000 |
|  MLflow UI | http://localhost:5000 |
|  Dashboard | http://localhost:80 |

### Option 2 — Sans Docker (développement)

```bash
# Étape 1 — Valider les données
python src/validate_data.py params.yaml

# Étape 2 — Prétraiter les données
python src/preprocess.py params.yaml

# Étape 3 — Entraîner le modèle
python src/train.py params.yaml

# Étape 4 — Évaluer le modèle
python src/evaluate.py params.yaml

# Étape 5 — Lancer l'API
uvicorn src.api:app --reload --port 8000

# Étape 6 — Lancer MLflow (terminal séparé)
mlflow ui

# Étape 7 — Ouvrir le dashboard
# Ouvrir dashboard/index.html dans le navigateur
```

---

##  API Endpoints

### Base URL : `http://localhost:8000`

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Statut de l'API |
| `/health` | GET | Healthcheck (modèle chargé, compteurs) |
| `/predict` | POST | Prédiction d'irrigation |
| `/model-info` | GET | Informations sur le modèle en production |
| `/metrics` | GET | Monitoring (total prédictions, taux de succès) |
| `/docs` | GET | Documentation Swagger UI auto-générée |

### Exemple de requête `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature_C": 35.0,
    "Humidity": 40.0,
    "Rainfall_mm": 0.0,
    "Wind_Speed_kmh": 20.0,
    "Field_Area_hectare": 2.5,
    "Previous_Irrigation_mm": 10.0,
    "Crop_Type": "Wheat",
    "Crop_Growth_Stage": "Flowering",
    "Season": "Summer",
    "Irrigation_Type": "Drip",
    "Mulching_Used": "No",
    "Region": "South"
  }'
```

### Exemple de réponse

```json
{
  "prediction_code": 2,
  "irrigation_need": "High",
  "confidence": 0.9123,
  "timestamp": "2026-05-04T23:33:29.940",
  "model_version": "1.0.0"
}
```

---

##  Pipeline CI/CD

Le fichier `.github/workflows/train.yml` déclenche automatiquement 11 étapes à chaque `git push` :

```
git push → main
    │
    ├── 1. Checkout code
    ├── 2. Setup Python 3.10
    ├── 3. Install dependencies
    ├── 4. Lint (flake8)
    ├── 5. Validate data
    ├── 6. Preprocess data
    ├── 7. Train model
    ├── 8. Evaluate model ──── accuracy < 75% →  BLOQUÉ
    ├── 9. Upload metrics artifact
    ├── 10. Build Docker image
    └── 11. Test API in Docker (/, /health, /predict)
```

### Quality Gate

Le pipeline est automatiquement **bloqué** si l'accuracy du modèle est inférieure à **75%**, empêchant tout déploiement d'un modèle insuffisant.

---

##  Tests

```bash
# Lancer les 23 tests unitaires
python -m pytest tests/ -v

# Avec rapport de couverture
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Résultats

```
tests/test_api.py::TestBaseEndpoints::test_root_returns_200          PASSED
tests/test_api.py::TestBaseEndpoints::test_health_returns_200        PASSED
tests/test_api.py::TestBaseEndpoints::test_health_model_loaded       PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_valid_payload   PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_missing_field   PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_invalid_ph      PASSED
...
=============== 23 passed in 8.89s ===============
```

---

##  Structure du Projet

```
irrigation-intelligente/
│
├── .github/
│   └── workflows/
│       └── train.yml          # Pipeline CI/CD GitHub Actions
│
├── dashboard/
│   └── index.html             # Dashboard React avec géolocalisation GPS
│
├── data/                      # Données preprocessées (CSV)
│   ├── train_X.csv
│   ├── train_y.csv
│   ├── test_X.csv
│   └── test_y.csv
│
├── metrics/                   # Rapports JSON
│   ├── scores.json            # Métriques du modèle
│   ├── validation_report.json # Rapport DataOps
│   └── preprocessing_report.json
│
├── models/                    # Artefacts ML sauvegardés
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── onehot_encoder.pkl
│   └── target_encoder.pkl
│
├── src/                       # Code source Python
│   ├── api.py                 # API FastAPI
│   ├── train.py               # Entraînement ML + MLflow
│   ├── evaluate.py            # Évaluation + Quality Gate
│   ├── preprocess.py          # Pipeline DataOps
│   └── validate_data.py       # Validation des données
│
├── tests/
│   ├── __init__.py
│   └── test_api.py            # 23 tests unitaires
│
├── docker-compose.yml         # Orchestration 3 containers
├── Dockerfile                 # Image Docker API
├── params.yaml                # Paramètres centralisés
├── requirements.txt           # Dépendances Python
├── dvc.yaml                   # Pipeline DVC
└── README.md                  # Ce fichier
```

---

##  Dashboard

Le dashboard AgroFlow est une application web accessible depuis tout appareil sur le même réseau.

### Fonctionnalités

- **Auto-remplissage météo GPS** : Le bouton "Ma Position" détecte votre localisation et remplit automatiquement Température, Humidité, Vent et Précipitations via l'API wttr.in
- **4 onglets** : Prédiction, Monitoring, Historique, Modèle
- **Accessibilité réseau** : Accessible depuis mobile via `http://[IP_PC]:80`

### Accès depuis un autre appareil (même WiFi)

```bash
# 1. Trouver l'IP du PC
ipconfig  # Windows
# Chercher "Adresse IPv4" dans "Carte réseau sans fil Wi-Fi"
# Exemple : 192.168.100.15

# 2. Sur l'autre appareil, ouvrir :
# http://192.168.100.15:80
```

---

##  Paramètres du Modèle (params.yaml)

```yaml
train:
  n_estimators: 100      # Random Forest
  max_depth: 10
  xgb_n_estimators: 100  # XGBoost
  xgb_max_depth: 6
  xgb_learning_rate: 0.1

evaluate:
  accuracy_threshold: 0.75  # Quality Gate

mlflow:
  experiment_name: "s3id_ia"
```

---

##  Auteurs

Projet réalisé dans le cadre du module **AppIA & MILOPS**
Filière **IASD (Intelligence Artificielle et Science des Données)** — Semestre 6

---

##  Licence

Projet académique — Usage éducatif uniquement.


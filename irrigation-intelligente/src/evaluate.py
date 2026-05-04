import pandas as pd
import joblib
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import mlflow
import mlflow.sklearn

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SEUIL MINIMUM D'ACCEPTATION DU MODÈLE
# ─────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.75  # ← le CI/CD échoue si on est en dessous

def evaluate(config_path="params.yaml"):

    logger.info("Début de l'évaluation du modèle...")

    # 1. Charger les données de test
    if not os.path.exists("data/test_X.csv") or not os.path.exists("data/test_y.csv"):
        raise FileNotFoundError("❌ Fichiers de test introuvables dans data/")

    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv").values.ravel()

    # 2. Charger le modèle
    if not os.path.exists("models/model.pkl"):
        raise FileNotFoundError("❌ models/model.pkl introuvable. Lance d'abord train.py")
    model = joblib.load("models/model.pkl")

    # 3. Charger l'encodeur cible pour les vrais labels
    target_le = joblib.load("models/target_encoder.pkl")

    # 4. Prédictions
    predictions = model.predict(X_test)
    proba       = model.predict_proba(X_test)

    # 5. Métriques
    accuracy   = accuracy_score(y_test, predictions)
    f1         = f1_score(y_test, predictions, average='weighted')
    report     = classification_report(y_test, predictions, output_dict=True)
    cm         = confusion_matrix(y_test, predictions).tolist()

    logger.info(f" Accuracy  : {accuracy * 100:.2f}%")
    logger.info(f" F1-Score  : {f1:.4f}")
    logger.info(f" Precision : {report['weighted avg']['precision']:.4f}")
    logger.info(f" Recall    : {report['weighted avg']['recall']:.4f}")

    # 6. Construire le dictionnaire de métriques complet
    metrics = {
        "accuracy":                    round(accuracy, 4),
        "f1_score_weighted":           round(f1, 4),
        "weighted_avg_precision":      round(report['weighted avg']['precision'], 4),
        "weighted_avg_recall":         round(report['weighted avg']['recall'], 4),
        "confusion_matrix":            cm,
        "classes":                     list(target_le.classes_),
        "n_test_samples":              len(y_test),
        "threshold_passed":            accuracy >= ACCURACY_THRESHOLD
    }

    # 7. Sauvegarder metrics.json (pour DVC)
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/scores.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métriques sauvegardées dans {metrics_path}")

    # 8. Logger dans MLflow
    try:
        mlflow.set_experiment("irrigation-intelligente")
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metric("accuracy",               accuracy)
            mlflow.log_metric("f1_score_weighted",      f1)
            mlflow.log_metric("weighted_avg_precision", report['weighted avg']['precision'])
            mlflow.log_metric("weighted_avg_recall",    report['weighted avg']['recall'])
            mlflow.log_artifact(metrics_path)
        logger.info("Métriques envoyées à MLflow")
    except Exception as e:
        logger.warning(f"MLflow non disponible, métriques non envoyées : {e}")

    # 9. Gate qualité → fait échouer le CI/CD si modèle trop faible
    if not metrics["threshold_passed"]:
        logger.error(
            f"QUALITÉ INSUFFISANTE : accuracy={accuracy:.2%} "
            f"< seuil={ACCURACY_THRESHOLD:.0%}. Pipeline bloqué."
        )
        raise SystemExit(1)  # ← le CI/CD GitHub Actions détecte ce code de sortie

    logger.info("Évaluation terminée. Modèle validé et prêt pour le déploiement.")
    return metrics


if __name__ == "__main__":
    evaluate("params.yaml")
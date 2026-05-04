import pandas as pd
from xgboost import XGBClassifier          # ← 1. Remplace RandomForestClassifier
import joblib
import yaml
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import json

def train(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    X_train = pd.read_csv("data/train_X.csv")
    y_train = pd.read_csv("data/train_y.csv")
    X_test  = pd.read_csv("data/test_X.csv")
    y_test  = pd.read_csv("data/test_y.csv")

    mlflow.set_experiment("s3id_ia")

    with mlflow.start_run():

        # Log des paramètres
        mlflow.log_param("model_type",    "XGBoost")
        mlflow.log_param("n_estimators",  config['train']['xgb_n_estimators'])
        mlflow.log_param("max_depth",     config['train']['xgb_max_depth'])
        mlflow.log_param("learning_rate", config['train']['xgb_learning_rate'])
        mlflow.log_param("random_state",  config['base']['random_state'])

        model = XGBClassifier(             # ← 2. Remplace RandomForestClassifier
            n_estimators=config['train']['xgb_n_estimators'],
            max_depth=config['train']['xgb_max_depth'],
            learning_rate=config['train']['xgb_learning_rate'],
            random_state=config['base']['random_state'],
            eval_metric='mlogloss',
            verbosity=0
        )

        model.fit(X_train, y_train.values.ravel())

        # Log des métriques
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy : {accuracy:.4f}")

        # Log du modèle dans MLflow Registry
        mlflow.sklearn.log_model(
            model,
            name="xgboost_model",          # ← 3. Remplace "random_forest_model"
            registered_model_name="IrrigationModel"
        )

        # Sauvegarde locale (pour l'API)
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # Sauvegarder les métriques pour DVC
        os.makedirs('metrics', exist_ok=True)
        with open("metrics/scores.json", "w") as f:
            json.dump({"accuracy": accuracy}, f)

        print(f" Run MLflow enregistré avec accuracy={accuracy:.4f}")

if __name__ == "__main__":
    train("params.yaml")
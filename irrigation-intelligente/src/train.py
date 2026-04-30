import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    X_test  = pd.read_csv("data/test_X.csv")   # ← tu dois avoir ça
    y_test  = pd.read_csv("data/test_y.csv")

    #  MLflow : démarrer un run
    mlflow.set_experiment("irrigation-intelligente")

    with mlflow.start_run():

        # Log des paramètres
        mlflow.log_param("n_estimators", config['train']['n_estimators'])
        mlflow.log_param("max_depth",    config['train']['max_depth'])
        mlflow.log_param("random_state", config['base']['random_state'])

        model = RandomForestClassifier(
            n_estimators=config['train']['n_estimators'],
            max_depth=config['train']['max_depth'],
            random_state=config['base']['random_state']
        )
        model.fit(X_train, y_train.values.ravel())

        # Log des métriques
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f" Accuracy : {accuracy:.4f}")

        # Log du modèle dans MLflow Registry
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model",
            registered_model_name="IrrigationModel"  # ← versioning auto
        )

        # Sauvegarde locale aussi (pour l'API)
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # Sauvegarder les métriques pour DVC
        os.makedirs('metrics', exist_ok=True)
        with open("metrics/scores.json", "w") as f:
            json.dump({"accuracy": accuracy}, f)

        print(f" Run MLflow enregistré avec accuracy={accuracy:.4f}")

if __name__ == "__main__":
    train("params.yaml")
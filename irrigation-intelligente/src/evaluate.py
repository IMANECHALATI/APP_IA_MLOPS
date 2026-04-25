import pandas as pd
import joblib
import json
import yaml
from sklearn.metrics import accuracy_score, classification_report
import os

def evaluate(config_path):
    # 1. Charger les données de test
    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv")
    
    # 2. Charger le modèle entraîné
    model = joblib.load("models/model.pkl")
    
    # 3. Faire des prédictions
    predictions = model.predict(X_test)
    
    # 4. Calculer les métriques
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    print(f" Précision du modèle : {accuracy * 100:.2f}%")
    
    # 5. Sauvegarder les métriques pour DVC/MLOps
    metrics = {
        "accuracy": accuracy,
        "weighted_avg_precision": report['weighted avg']['precision'],
        "weighted_avg_recall": report['weighted avg']['recall']
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(" Métriques sauvegardées dans metrics.json")

if __name__ == "__main__":
    evaluate("params.yaml")
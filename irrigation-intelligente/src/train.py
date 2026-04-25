import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import os

def train(config_path):
    # Charger les paramètres
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
        
    # Charger les données prétraitées
    X_train = pd.read_csv("data/train_X.csv")
    y_train = pd.read_csv("data/train_y.csv")
    
    print(f"Entraînement sur {X_train.shape[0]} échantillons...")

    # Initialiser le modèle avec les paramètres du fichier params.yaml
    model = RandomForestClassifier(
        n_estimators=config['train']['n_estimators'],
        max_depth=config['train']['max_depth'],
        random_state=config['base']['random_state']
    )
    
    # Entraînement
    # .values.ravel() transforme le DataFrame y en vecteur simple attendu par sklearn
    model.fit(X_train, y_train.values.ravel())
    
    # Sauvegarder le modèle dans le dossier models/
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    print(" MLOps : Modèle entraîné et sauvegardé dans models/model.pkl")

if __name__ == "__main__":
    train("params.yaml")
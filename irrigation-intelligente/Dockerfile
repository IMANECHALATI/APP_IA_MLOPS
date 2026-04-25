# 1. Utiliser une image Python légère comme base
FROM python:3.9-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier de dépendances en premier (optimisation du cache)
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le reste du projet
COPY . .

# 6. Exposer le port que Flask va utiliser
EXPOSE 5000

# 7. Commande pour lancer l'API au démarrage du conteneur
CMD ["python", "src/api.py"]
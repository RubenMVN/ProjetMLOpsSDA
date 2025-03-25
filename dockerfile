# Étape 1 : Utiliser une image de base
FROM python:3.8-slim

# Étape 2 : Définir un répertoire de travail dans l'image
WORKDIR /app

# Étape 3 : Copier les fichiers locaux vers l'image
COPY . /app

# Étape 4 : Installer les dépendances du projet
RUN ls -l requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port sur lequel ton application va tourner
EXPOSE 8501

# Étape 6 : Définir la commande à exécuter pour démarrer l'application
CMD ["streamlit","run", "app.py"]

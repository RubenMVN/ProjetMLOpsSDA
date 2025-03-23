import streamlit as st
import joblib
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import joblib
import os

# Cnnect to the MLflow tracking server
try:
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:8080")
    print("Connected to MLflow server")
except:
    print(
        "MLflow server is not running!\nPlease execute the cell above in a terminal to launch the MLflow server."
    )

df = pd.read_csv(r"C:\Users\Sam\DU Data Analytics\ML Ops\Loan_Data.csv")
df.head()

X = df.drop(columns=["default"])  # Features
y = df["default"]  # Target


print("Features shape:", X.shape)
print("Target shape:", y.shape)

X = df.drop("default", axis=1) 
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import train_test_split

# Définir random_seed
random_seed = 42  # Vous pouvez utiliser n'importe quel entier

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Afficher les dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Modèle 1 : Random Forest
mlflow.set_experiment("Random_Forest_Experiment")

with mlflow.start_run(run_name="Random_Forest_Run"):
    # Entraîner le modèle
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Faire des prédictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Afficher les métriques dans le notebook
    print("Random Forest - Métriques :")
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"F1 Score: {f1}")
    
    # Log des métriques dans MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    
    # Log du modèle
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

# Modèle 2 : Régression Logistique
mlflow.set_experiment("Logistic_Regression_Experiment")

with mlflow.start_run(run_name="Logistic_Regression_Run"):
    # Entraîner le modèle
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Faire des prédictions
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Afficher les métriques dans le notebook
    print("Régression Logistique - Métriques :")
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"F1 Score: {f1}")
    
    # Log des métriques dans MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    
    # Log du modèle
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

import joblib

# Sauvegarder le modèle Random Forest
joblib.dump(rf_model, "random_forest_model.pkl")

# Sauvegarder le modèle de Régression Logistique
joblib.dump(lr_model, "logistic_regression_model.pkl")

# Sauvegarder le scaler (si vous l'avez utilisé)
joblib.dump(scaler, "scaler.pkl")

# Charger les modèles et le scaler avec des chemins absolus
rf_model = joblib.load(r"C:\Users\Sam\DU Data Analytics\ML Ops\models\random_forest_model.pkl")
lr_model = joblib.load(r"C:\Users\Sam\DU Data Analytics\ML Ops\models\logistic_regression_model.pkl")
scaler = joblib.load(r"C:\Users\Sam\DU Data Analytics\ML Ops\models\scaler.pkl")

# Titre de l'application
st.title("Prédiction de Défaut de Paiement")

# Formulaire pour saisir les données
st.write("Entrez les informations du client :")
credit_lines = st.number_input("Lignes de crédit", min_value=0)
loan_amt = st.number_input("Montant du prêt", min_value=0)
total_debt = st.number_input("Dette totale", min_value=0)
income = st.number_input("Revenu", min_value=0)
years_employed = st.number_input("Années d'emploi", min_value=0)
fico_score = st.number_input("Score FICO", min_value=300, max_value=850)

# Bouton pour faire la prédiction
if st.button("Prédire"):
    # Préparer les données d'entrée
    input_data = np.array([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]])
    input_data_scaled = scaler.transform(input_data)
    
    # Faire des prédictions avec les deux modèles
    rf_prediction = rf_model.predict_proba(input_data_scaled)[:, 1][0]
    lr_prediction = lr_model.predict_proba(input_data_scaled)[:, 1][0]
    
    # Afficher les résultats
    st.write(f"Probabilité de défaut (Random Forest) : {rf_prediction:.2f}")
    st.write(f"Probabilité de défaut (Régression Logistique) : {lr_prediction:.2f}")
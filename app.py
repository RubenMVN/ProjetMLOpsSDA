import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Charger les données
df = pd.read_csv(r"Loan_Data.csv")

# Séparation des caractéristiques et de la cible
X = df.drop(columns=["default"])  # Features
y = df["default"]  # Target

# Diviser les données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modèle 1 : Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Modèle 2 : Régression Logistique
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Sauvegarder les modèles et le scaler
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Charger les modèles et le scaler sauvegardés
rf_model = joblib.load(r"random_forest_model.pkl")
lr_model = joblib.load(r"logistic_regression_model.pkl")
scaler = joblib.load(r"scaler.pkl")

# Titre de l'application Streamlit
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
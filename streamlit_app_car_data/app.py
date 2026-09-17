"""
App Streamlit - Prédiction "Venant / D'occasion" pour une voiture (Car_Data)

Fichiers attendus dans le MÊME dossier que ce script (générés par le notebook
1Clas_KNN_on_Car_Data_complet.ipynb, section "Sélection automatique du meilleur modèle") :
    - xgb_model.json   (le meilleur modèle, quel que soit son algorithme réel)
    - scaler.joblib
    - encoders.joblib
    - uniques.joblib

Lancer en local :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
import os

st.set_page_config(page_title="Prédiction état d'une voiture", page_icon="🚗")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    # format natif XGBoost (JSON) : portable entre systèmes, contrairement au pickle/joblib
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    scaler = jb.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = jb.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    uniques = jb.load(os.path.join(MODEL_DIR, "uniques.joblib"))
    return model, scaler, encoders, uniques


try:
    model, scaler, encoders, uniques = load_artifacts()
except Exception as e:
    st.error(
        "Impossible de charger les fichiers du modèle (xgb_model.json, scaler.joblib, "
        "encoders.joblib, uniques.joblib). Vérifie qu'ils sont bien à côté de app.py.\n\n"
        "Détail : " + str(e)
    )
    st.stop()

# uniques / encoders : [Marque, Transmission, Quartier, Etat]
marques = uniques[0]
transmissions = uniques[1]
quartiers = uniques[2]
class_names = uniques[3]  # ["D'occasion", "Venant"]

st.title("🚗 Prédire si une voiture est neuve/venante ou d'occasion")
st.caption("Modèle entrainé sur des annonces de voitures à Dakar (expat-dakar.com).")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction par fichier CSV"])

# ---------------------------------------------------------------------------
# Prédiction simple
# ---------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.selectbox("Marque", marques)
            annee = st.number_input("Année", min_value=1990, max_value=2026, value=2015, step=1)
            transmission = st.selectbox("Transmission", transmissions)
        with col2:
            quartier = st.selectbox("Quartier", quartiers)
            prix = st.number_input("Prix (FCFA)", min_value=0, value=5_000_000, step=100_000)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        marque_enc = encoders[0].transform([marque])[0]
        transmission_enc = encoders[1].transform([transmission])[0]
        quartier_enc = encoders[2].transform([quartier])[0]

        # même ordre de colonnes que dans le notebook : Marque, Année, Transmission, Quartier, Prix
        x_new = np.array([[marque_enc, annee, transmission_enc, quartier_enc, prix]])
        x_new = scaler.transform(x_new)

        y_pred = model.predict(x_new)[0]
        proba = model.predict_proba(x_new)[0] if hasattr(model, "predict_proba") else None

        st.success(f"Prédiction : **{class_names[y_pred]}**")
        if proba is not None:
            st.write(pd.DataFrame({"classe": class_names, "probabilité": proba}).set_index("classe"))

# ---------------------------------------------------------------------------
# Prédiction par lot (CSV)
# ---------------------------------------------------------------------------
with tab2:
    st.write(
        "Le fichier doit contenir les colonnes suivantes, dans cet ordre : "
        "`Marque, Année, Transmission, Quartier, Prix`."
    )
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        try:
            df_enc = df_in.copy()
            df_enc["Marque"] = encoders[0].transform(df_enc["Marque"])
            df_enc["Transmission"] = encoders[1].transform(df_enc["Transmission"])
            df_enc["Quartier"] = encoders[2].transform(df_enc["Quartier"])

            x_batch = df_enc[["Marque", "Année", "Transmission", "Quartier", "Prix"]].values
            x_batch = scaler.transform(x_batch)
            preds = model.predict(x_batch)

            df_out = df_in.copy()
            df_out["Etat_predit"] = [class_names[p] for p in preds]
            st.dataframe(df_out)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name="predictions_car_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur pendant le traitement du fichier : {e}")

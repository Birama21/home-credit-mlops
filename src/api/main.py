from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "lightgbm_final.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

THRESHOLD = 0.51


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


# ============================================================
# Chargement du modèle
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Modèle introuvable : {MODEL_PATH}"
    ) from exc


# ============================================================
# Chargement des données
# ============================================================

try:
    df_features = pd.read_parquet(DATA_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Dataset introuvable : {DATA_PATH}"
    ) from exc

df_features.columns = clean_feature_names(df_features.columns)

if "SK_ID_CURR" not in df_features.columns:
    raise RuntimeError("SK_ID_CURR introuvable.")

df_features = df_features.set_index("SK_ID_CURR", drop=False)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Home Credit Scoring API",
    description="API de prédiction du risque de défaut",
    version="1.0.0",
)


# ============================================================
# Schéma d'entrée
# ============================================================

class ClientRequest(BaseModel):

    sk_id_curr: int = Field(
        ...,
        example=100002,
    )


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Home Credit Scoring API",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "data_loaded": True,
        "n_clients_available": int(df_features.shape[0]),
        "n_model_features": len(model.feature_name_),
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(request: ClientRequest):

    sk_id_curr = request.sk_id_curr

    if sk_id_curr not in df_features.index:
        raise HTTPException(
            status_code=404,
            detail=f"Client {sk_id_curr} introuvable.",
        )

    client = df_features.loc[[sk_id_curr]].copy()

    client = client.drop(
        columns=[
            c
            for c in ["TARGET", "SK_ID_CURR"]
            if c in client.columns
        ]
    )

    expected_features = list(model.feature_name_)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in client.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Features manquantes",
                "missing_features_count": len(missing_features),
                "missing_features_sample": missing_features[:20],
            },
        )

    client = client[expected_features]

    # Même conversion que pendant l'entraînement
    for col in client.select_dtypes(include=["object"]).columns:
        client[col] = pd.to_numeric(
            client[col],
            errors="coerce",
        )

    client = client.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    probability = float(
        model.predict_proba(client)[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    decision = (
        "client_risque"
        if prediction == 1
        else "client_non_risque"
    )

    response = {
        "sk_id_curr": sk_id_curr,
        "probability_default": round(probability, 6),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "decision": decision,
    }

    if (
        "TARGET" in df_features.columns
        and pd.notna(df_features.loc[sk_id_curr, "TARGET"])
    ):
        response["true_target"] = int(
            df_features.loc[sk_id_curr, "TARGET"]
        )

    return response
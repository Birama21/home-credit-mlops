from pathlib import Path
import os
import re

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "lightgbm_final.pkl"
DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        BASE_DIR / "data" / "demo" / "production_demo_clients.csv",
    )
)

THRESHOLD = 0.51
DEFAULT_BATCH_SIZE = 200


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    """Nettoie les noms de colonnes comme pendant l'entraînement."""
    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


def load_production_data():
    """Charge les données de production simulées à la demande."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fichier de production introuvable : {DATA_PATH}",
        ) from exc

    df.columns = clean_feature_names(df.columns)

    if "SK_ID_CURR" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="La colonne SK_ID_CURR est absente du fichier de production.",
        )

    return df


def prepare_features(df):
    """Prépare les features avant prédiction."""
    features = df.drop(
        columns=[
            col
            for col in ["SK_ID_CURR", "TARGET"]
            if col in df.columns
        ]
    )

    expected_features = list(model.feature_name_)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
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

    features = features[expected_features]

    for col in features.select_dtypes(include=["object"]).columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def build_decision(probability):
    """Transforme une probabilité en prédiction métier."""
    prediction = int(probability >= THRESHOLD)

    decision = (
        "client_risque"
        if prediction == 1
        else "client_non_risque"
    )

    return prediction, decision


# ============================================================
# Chargement du modèle au démarrage
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Modèle introuvable : {MODEL_PATH}"
    ) from exc


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
    """Identifiant client à scorer."""

    sk_id_curr: int = Field(
        ...,
        example=128180,
    )


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    """Route d'accueil."""

    return {
        "message": "Home Credit Scoring API",
        "status": "running",
    }


@app.get("/health")
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "data_path": str(DATA_PATH),
        "n_model_features": len(model.feature_name_),
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(request: ClientRequest):
    """Prédit le risque de défaut pour un client donné."""

    df = load_production_data()

    sk_id_curr = request.sk_id_curr

    client = df[df["SK_ID_CURR"] == sk_id_curr].copy()

    if client.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Client {sk_id_curr} introuvable dans les données de production.",
        )

    features = prepare_features(client)

    probability = float(model.predict_proba(features)[0, 1])

    prediction, decision = build_decision(probability)

    response = {
        "sk_id_curr": int(sk_id_curr),
        "probability_default": round(probability, 6),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "decision": decision,
    }

    if "TARGET" in client.columns and pd.notna(client["TARGET"].iloc[0]):
        response["true_target"] = int(client["TARGET"].iloc[0])

    return response


@app.post("/predict_batch")
def predict_batch(
    n_clients: int = Query(
        DEFAULT_BATCH_SIZE,
        ge=1,
        le=1000,
        description="Nombre de clients à scorer depuis le début du fichier CSV.",
    )
):
    """Prédit le risque de défaut pour les n premiers clients du fichier CSV."""

    df = load_production_data()

    batch = df.head(n_clients).copy()

    if batch.empty:
        raise HTTPException(
            status_code=400,
            detail="Le fichier de production est vide.",
        )

    features = prepare_features(batch)

    probabilities = model.predict_proba(features)[:, 1]

    results = []

    for index, probability in enumerate(probabilities):
        prediction, decision = build_decision(float(probability))

        row = batch.iloc[index]

        result = {
            "sk_id_curr": int(row["SK_ID_CURR"]),
            "probability_default": round(float(probability), 6),
            "threshold": THRESHOLD,
            "prediction": prediction,
            "decision": decision,
        }

        if "TARGET" in batch.columns and pd.notna(row["TARGET"]):
            result["true_target"] = int(row["TARGET"])

        results.append(result)

    return {
        "n_predictions": len(results),
        "threshold": THRESHOLD,
        "predictions": results,
    }
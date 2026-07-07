from pathlib import Path
import os
import re
import json

import joblib
import numpy as np
import pandas as pd
import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from src.services.prediction_logger import log_prediction


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
    """
    Charge les données depuis le CSV.

    Ce fallback est indispensable pour Hugging Face, car le Space
    ne dispose pas de notre base PostgreSQL locale.
    """
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


def get_client_from_csv(sk_id_curr: int) -> pd.DataFrame:
    """Récupère un client depuis le CSV de secours."""
    df = load_production_data()

    client = df[df["SK_ID_CURR"] == sk_id_curr].copy()

    if client.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Client {sk_id_curr} introuvable.",
        )

    return client


def get_client_data(sk_id_curr: int) -> pd.DataFrame:
    """
    Récupère un client.

    Priorité :
    1. PostgreSQL en local.
    2. CSV en fallback pour Hugging Face.
    """
    try:
        from src.services.client_service import get_client

        return get_client(sk_id_curr)

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except Exception:
        return get_client_from_csv(sk_id_curr)


def get_batch_from_database(n_clients: int) -> pd.DataFrame:
    """Récupère les n premiers clients depuis PostgreSQL."""
    from sqlalchemy import select

    from src.database.database import SessionLocal
    from src.database.models import ClientDemo

    rows = []

    with SessionLocal() as session:
        query = (
            select(ClientDemo)
            .order_by(ClientDemo.sk_id_curr.asc())
            .limit(n_clients)
        )

        clients = session.execute(query).scalars().all()

        for client in clients:
            client_dict = json.loads(client.client_data)
            client_dict["SK_ID_CURR"] = client.sk_id_curr
            rows.append(client_dict)

    return pd.DataFrame(rows)


def get_batch_data(n_clients: int) -> pd.DataFrame:
    """
    Récupère un batch de clients.

    Priorité :
    1. PostgreSQL en local.
    2. CSV en fallback pour Hugging Face.
    """
    try:
        batch = get_batch_from_database(n_clients)

        if not batch.empty:
            return batch

    except Exception:
        pass

    df = load_production_data()
    return df.head(n_clients).copy()


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


class ClientRequest(BaseModel):
    """Identifiant client à scorer."""

    sk_id_curr: int = Field(
        ...,
        example=100009,
    )


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

    sk_id_curr = request.sk_id_curr
    start_time = time.perf_counter()

    try:
        client = get_client_data(sk_id_curr)

        features = prepare_features(client)

        probability = float(model.predict_proba(features)[0, 1])

        prediction, decision = build_decision(probability)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Sauvegarde la prédiction réussie dans PostgreSQL.
        log_prediction(
            endpoint="/predict",
            sk_id_curr=sk_id_curr,
            probability=probability,
            prediction=prediction,
            decision=decision,
            latency_ms=latency_ms,
            status="success",
        )

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

    except HTTPException as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Sauvegarde l'erreur dans PostgreSQL.
        log_prediction(
            endpoint="/predict",
            sk_id_curr=sk_id_curr,
            probability=None,
            prediction=None,
            decision=None,
            latency_ms=latency_ms,
            status="error",
            error_message=str(exc.detail),
        )

        raise exc


@app.post("/predict_batch")
def predict_batch(
    n_clients: int = Query(
        DEFAULT_BATCH_SIZE,
        ge=1,
        le=1000,
        description="Nombre de clients à scorer.",
    )
):
    """Prédit le risque de défaut pour les n premiers clients."""

    start_time = time.perf_counter()

    try:
        batch = get_batch_data(n_clients)

        if batch.empty:
            raise HTTPException(
                status_code=400,
                detail="Aucun client disponible.",
            )

        features = prepare_features(batch)

        probabilities = model.predict_proba(features)[:, 1]

        results = []

        for index, probability in enumerate(probabilities):
            row = batch.iloc[index]

            prediction, decision = build_decision(float(probability))

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Sauvegarde chaque prédiction du batch dans PostgreSQL.
            log_prediction(
                endpoint="/predict_batch",
                sk_id_curr=int(row["SK_ID_CURR"]),
                probability=float(probability),
                prediction=prediction,
                decision=decision,
                latency_ms=latency_ms,
                status="success",
            )

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

    except HTTPException as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        log_prediction(
            endpoint="/predict_batch",
            sk_id_curr=None,
            probability=None,
            prediction=None,
            decision=None,
            latency_ms=latency_ms,
            status="error",
            error_message=str(exc.detail),
        )

        raise exc
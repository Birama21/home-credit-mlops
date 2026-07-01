# ============================================================
# Tests unitaires de l'API FastAPI
# ============================================================

import pandas as pd
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app


client = TestClient(app)


# ============================================================
# Données simulées pour les tests
# ============================================================

def fake_load_production_data():
    """Crée un mini DataFrame de test avec les features attendues par le modèle."""

    expected_features = list(main.model.feature_name_)

    fake_client = {
        feature: 0
        for feature in expected_features
    }

    fake_client["SK_ID_CURR"] = 309296
    fake_client["TARGET"] = 0

    fake_unknown = {
        feature: 0
        for feature in expected_features
    }

    fake_unknown["SK_ID_CURR"] = 123456
    fake_unknown["TARGET"] = 1

    return pd.DataFrame([fake_client, fake_unknown])


# ============================================================
# Test /health
# ============================================================

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["threshold"] == 0.51


# ============================================================
# Test /predict
# ============================================================

def test_predict_existing_client(monkeypatch):

    monkeypatch.setattr(
        main,
        "load_production_data",
        fake_load_production_data,
    )

    response = client.post(
        "/predict",
        json={"sk_id_curr": 309296},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sk_id_curr"] == 309296
    assert "probability_default" in data
    assert "prediction" in data
    assert "decision" in data
    assert "true_target" in data

    assert 0 <= data["probability_default"] <= 1
    assert data["prediction"] in [0, 1]


# ============================================================
# Client inexistant
# ============================================================

def test_predict_unknown_client(monkeypatch):

    monkeypatch.setattr(
        main,
        "load_production_data",
        fake_load_production_data,
    )

    response = client.post(
        "/predict",
        json={"sk_id_curr": 999999999},
    )

    assert response.status_code == 404


# ============================================================
# Mauvais type
# ============================================================

def test_predict_invalid_type():

    response = client.post(
        "/predict",
        json={"sk_id_curr": "abc"},
    )

    assert response.status_code == 422


# ============================================================
# Champ manquant
# ============================================================

def test_predict_missing_field():

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


# ============================================================
# Test /predict_batch
# ============================================================

def test_predict_batch(monkeypatch):

    monkeypatch.setattr(
        main,
        "load_production_data",
        fake_load_production_data,
    )

    response = client.post(
        "/predict_batch?n_clients=2",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["n_predictions"] == 2
    assert len(data["predictions"]) == 2

    first_prediction = data["predictions"][0]

    assert "sk_id_curr" in first_prediction
    assert "probability_default" in first_prediction
    assert "prediction" in first_prediction
    assert "decision" in first_prediction
    assert "true_target" in first_prediction
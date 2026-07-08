"""
Service de détection simple du data drift.

Le drift est détecté en comparant :
- les statistiques de référence calculées sur le train ;
- les statistiques des clients réellement scorés par l'API.

Les clients scorés sont récupérés via prediction_logs.
Chaque client est compté une seule fois, même s'il a été scoré plusieurs fois.
"""

from pathlib import Path
import json

import pandas as pd
from sqlalchemy import select

from src.database.database import SessionLocal
from src.database.models import PredictionLog


BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_STATS_PATH = BASE_DIR / "data" / "reference" / "reference_stats.json"
PRODUCTION_DATA_PATH = BASE_DIR / "data" / "production" / "production_test_clients.csv"

DRIFT_THRESHOLD = 0.10

DRIFT_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "EXT_SOURCE_2",
]


def get_scored_client_ids() -> list[int]:
    """
    Récupère les identifiants uniques des clients scorés avec succès.

    Un client peut être scoré plusieurs fois via l'API.
    Pour le data drift, on le compte une seule fois afin de ne pas biaiser
    les moyennes.
    """

    if SessionLocal is None:
        return []

    with SessionLocal() as session:
        query = (
            select(PredictionLog.sk_id_curr)
            .where(PredictionLog.status == "success")
            .where(PredictionLog.sk_id_curr.is_not(None))
            .distinct()
        )

        return [int(row[0]) for row in session.execute(query).all()]


def detect_data_drift() -> dict:
    """
    Compare les moyennes de référence avec les moyennes des clients scorés.
    """

    if not REFERENCE_STATS_PATH.exists():
        return {
            "drift_monitoring_enabled": False,
            "message": f"Fichier de référence introuvable : {REFERENCE_STATS_PATH}",
        }

    if not PRODUCTION_DATA_PATH.exists():
        return {
            "drift_monitoring_enabled": False,
            "message": f"Fichier de production introuvable : {PRODUCTION_DATA_PATH}",
        }

    scored_client_ids = get_scored_client_ids()

    if not scored_client_ids:
        return {
            "drift_monitoring_enabled": True,
            "drift_detected": False,
            "message": "Aucun client scoré disponible pour calculer le data drift.",
        }

    with open(REFERENCE_STATS_PATH, "r", encoding="utf-8") as file:
        reference_stats = json.load(file)

    production_df = pd.read_csv(PRODUCTION_DATA_PATH)

    scored_clients_df = production_df[
        production_df["SK_ID_CURR"].isin(scored_client_ids)
    ].copy()

    if scored_clients_df.empty:
        return {
            "drift_monitoring_enabled": True,
            "drift_detected": False,
            "message": "Les clients scorés ne sont pas présents dans les données de production simulée.",
            "n_scored_clients": len(scored_client_ids),
        }

    variables = {}
    variables_with_drift = []

    for feature in DRIFT_FEATURES:
        if feature not in reference_stats or feature not in scored_clients_df.columns:
            continue

        reference_mean = float(reference_stats[feature]["mean"])
        production_mean = float(scored_clients_df[feature].mean())
        production_count = int(scored_clients_df[feature].count())

        if reference_mean == 0:
            relative_change = 0
        else:
            relative_change = abs(production_mean - reference_mean) / abs(reference_mean)

        drift_detected = relative_change > DRIFT_THRESHOLD

        if drift_detected:
            variables_with_drift.append(feature)

        variables[feature] = {
            "reference_mean": round(reference_mean, 6),
            "production_mean": round(production_mean, 6),
            "production_count": production_count,
            "relative_change_percent": round(relative_change * 100, 2),
            "drift_detected": drift_detected,
        }

    global_drift_detected = len(variables_with_drift) > 0

    recommendation = (
        "Data drift détecté. Il est recommandé de surveiller les performances du modèle "
        "et d'envisager un réentraînement si elles se dégradent."
        if global_drift_detected
        else "Aucun data drift significatif détecté. Aucune action immédiate requise."
    )

    return {
        "drift_monitoring_enabled": True,
        "threshold_percent": DRIFT_THRESHOLD * 100,
        "n_scored_clients": len(scored_client_ids),
        "n_clients_used_for_drift": len(scored_clients_df),
        "drift_detected": global_drift_detected,
        "variables_with_drift": variables_with_drift,
        "variables": variables,
        "recommendation": recommendation,
    }
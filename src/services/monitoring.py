"""
Service de monitoring des prédictions.

Ce module lit la table prediction_logs et calcule
des statistiques simples pour le suivi de l'API.
"""

from sqlalchemy import func

from src.database.database import SessionLocal
from src.database.models import PredictionLog


def get_prediction_stats() -> dict:
    """Retourne les statistiques principales de monitoring."""

    if SessionLocal is None:
        return {
            "database_enabled": False,
            "message": "PostgreSQL non configuré.",
        }

    session = SessionLocal()

    try:
        total_predictions = session.query(PredictionLog).count()

        total_success = (
            session.query(PredictionLog)
            .filter(PredictionLog.status == "success")
            .count()
        )

        total_errors = (
            session.query(PredictionLog)
            .filter(PredictionLog.status == "error")
            .count()
        )

        error_rate = (
            total_errors / total_predictions
            if total_predictions > 0
            else 0
        )

        avg_latency_ms = (
            session.query(func.avg(PredictionLog.latency_ms))
            .filter(PredictionLog.status == "success")
            .scalar()
        )

        avg_probability = (
            session.query(func.avg(PredictionLog.probability))
            .filter(PredictionLog.status == "success")
            .scalar()
        )

        risky_clients = (
            session.query(PredictionLog)
            .filter(PredictionLog.prediction == 1)
            .count()
        )

        non_risky_clients = (
            session.query(PredictionLog)
            .filter(PredictionLog.prediction == 0)
            .count()
        )

        return {
            "database_enabled": True,
            "total_predictions": total_predictions,
            "total_success": total_success,
            "total_errors": total_errors,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency_ms or 0, 2),
            "avg_probability_default": round(avg_probability or 0, 6),
            "risky_clients": risky_clients,
            "non_risky_clients": non_risky_clients,
        }

    finally:
        session.close()
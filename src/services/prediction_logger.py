"""
Service d'enregistrement des prédictions dans PostgreSQL.

Ce module permet de sauvegarder chaque prédiction effectuée par l'API
dans la table prediction_logs. Ces logs serviront ensuite au monitoring.
"""

from src.database.database import SessionLocal
from src.database.models import PredictionLog


def log_prediction(
    endpoint: str,
    sk_id_curr: int | None,
    probability: float | None,
    prediction: int | None,
    decision: str | None,
    latency_ms: float | None,
    status: str,
    error_message: str | None = None,
) -> None:
    """
    Enregistre une prédiction ou une erreur dans PostgreSQL.
    """

    session = SessionLocal()

    try:
        log = PredictionLog(
            endpoint=endpoint,
            sk_id_curr=sk_id_curr,
            probability=probability,
            prediction=prediction,
            decision=decision,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

        session.add(log)
        session.commit()

    except Exception:
        # On ne bloque jamais l'API si le logging échoue.
        session.rollback()

    finally:
        session.close()
"""
Service d'enregistrement des prédictions dans PostgreSQL.

Si PostgreSQL n'est pas configuré, par exemple sur GitHub Actions
ou Hugging Face, le logging est simplement désactivé.
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
    Enregistre une seule prédiction ou une erreur dans PostgreSQL.

    Cette fonction est utilisée principalement par l'endpoint /predict.
    """

    if SessionLocal is None:
        return

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
        session.rollback()

    finally:
        session.close()


def log_predictions_batch(logs: list[dict]) -> None:
    """
    Enregistre plusieurs prédictions en une seule transaction PostgreSQL.

    Cette fonction est utilisée par /predict_batch afin d'éviter
    d'exécuter une requête SQL séparée pour chaque client.
    """

    if SessionLocal is None or not logs:
        return

    session = SessionLocal()

    try:
        prediction_logs = [
            PredictionLog(**log_data)
            for log_data in logs
        ]

        session.add_all(prediction_logs)
        session.commit()

    except Exception:
        session.rollback()

    finally:
        session.close()
"""
Modèles SQLAlchemy utilisés par l'application.

Ce fichier définit les tables PostgreSQL nécessaires à l'étape 3 :
- clients_demo : données clients utilisées pour les prédictions.
- prediction_logs : historique des appels à l'API et des prédictions.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.database import Base


class ClientDemo(Base):
    """
    Table contenant les clients disponibles pour les prédictions.

    Chaque ligne correspond à un client du dataset de test.
    Les features seront stockées sous forme JSON dans une étape suivante,
    ce qui évite de créer manuellement plusieurs centaines de colonnes SQL.
    """

    __tablename__ = "clients_demo"

    sk_id_curr: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    client_data: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


class PredictionLog(Base):
    """
    Table contenant l'historique des prédictions effectuées par l'API.

    Cette table servira au monitoring :
    suivi des appels, erreurs, latence et sorties du modèle.
    """

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    endpoint: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    sk_id_curr: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    prediction: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    decision: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
"""
Configuration de la connexion à PostgreSQL.

PostgreSQL est utilisé en local si les variables d'environnement
sont présentes. Sur GitHub Actions ou Hugging Face, la connexion
peut être désactivée afin de garder le fallback CSV fonctionnel.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from src.database.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    is_postgres_configured,
)


class Base(DeclarativeBase):
    """Classe de base des modèles SQLAlchemy."""
    pass


DATABASE_URL = None
engine = None
SessionLocal = None

if is_postgres_configured():
    DATABASE_URL = (
        f"postgresql+psycopg://"
        f"{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    engine = create_engine(DATABASE_URL)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
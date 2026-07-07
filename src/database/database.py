"""
Configuration de la connexion à la base de données.

Ce module charge les variables d'environnement définies
dans le fichier .env afin d'éviter de stocker des
informations sensibles (mot de passe, nom de la base...)
directement dans le code.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from src.database.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

# Construction de l'URL de connexion PostgreSQL.
# Le driver "psycopg" est utilisé par SQLAlchemy
# pour communiquer avec PostgreSQL.
DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Création de l'engine SQLAlchemy.
# L'engine représente la connexion principale
# entre l'application Python et PostgreSQL.
engine = create_engine(DATABASE_URL)

# Création d'une fabrique de sessions.
# Chaque requête vers la base utilisera une nouvelle
# session afin de garantir une bonne gestion des transactions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# Classe de base dont hériteront tous les modèles SQLAlchemy.
# Les futures tables (clients_demo, prediction_logs, ...)
# seront définies en héritant de cette classe.
class Base(DeclarativeBase):
    pass
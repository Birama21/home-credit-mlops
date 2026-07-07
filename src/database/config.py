"""
Chargement des variables d'environnement PostgreSQL.

En local, les variables viennent du fichier .env.
Sur GitHub Actions ou Hugging Face, elles peuvent être absentes :
dans ce cas, PostgreSQL est simplement désactivé.
"""

from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


def is_postgres_configured() -> bool:
    """Vérifie si toutes les variables PostgreSQL sont présentes."""
    return all(
        [
            POSTGRES_HOST,
            POSTGRES_PORT,
            POSTGRES_DB,
            POSTGRES_USER,
            POSTGRES_PASSWORD,
        ]
    )
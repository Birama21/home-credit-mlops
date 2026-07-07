"""
Création automatique des tables PostgreSQL.

Ce script lit les modèles SQLAlchemy définis dans models.py
et crée les tables correspondantes dans PostgreSQL si elles
n'existent pas encore.
"""

from src.database.database import Base, engine

# Import des modèles.
# Cet import est indispensable afin que SQLAlchemy connaisse
# les tables à créer.
from src.database import models  # noqa: F401


def create_tables() -> None:
    """
    Crée toutes les tables définies dans les modèles SQLAlchemy.
    """

    Base.metadata.create_all(bind=engine)

    print("✅ Les tables PostgreSQL ont été créées avec succès.")


if __name__ == "__main__":
    create_tables()
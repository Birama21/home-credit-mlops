"""
Service de récupération des clients depuis PostgreSQL.

Ce module centralise toutes les opérations de lecture de la table
clients_demo. Le reste de l'application (API, modèle...) n'aura jamais
besoin d'écrire directement des requêtes SQL.

Chaque client est stocké dans PostgreSQL sous forme de JSON.
Ce service reconvertit automatiquement ce JSON en DataFrame pandas,
afin que le modèle de Machine Learning puisse l'utiliser.
"""

# Permet de convertir le JSON stocké dans PostgreSQL
# en dictionnaire Python.
import json

# Manipulation des DataFrames.
import pandas as pd

# Construction des requêtes SQLAlchemy.
from sqlalchemy import select

# Création des sessions PostgreSQL.
from src.database.database import SessionLocal

# Modèle SQLAlchemy représentant la table clients_demo.
from src.database.models import ClientDemo


def get_client(sk_id_curr: int) -> pd.DataFrame:
    """
    Récupère un client depuis PostgreSQL.

    Parameters
    ----------
    sk_id_curr : int
        Identifiant du client.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant une seule ligne correspondant
        au client demandé.

    Raises
    ------
    ValueError
        Si le client n'existe pas.
    """

    # Ouvre une connexion à PostgreSQL.
    session = SessionLocal()

    try:
        # Recherche du client grâce à son identifiant.
        query = (
            select(ClientDemo)
            .where(ClientDemo.sk_id_curr == sk_id_curr)
        )

        client = session.execute(query).scalar_one_or_none()

        # Si aucun client n'a été trouvé.
        if client is None:
            raise ValueError(f"Client {sk_id_curr} introuvable.")

        # Conversion du texte JSON en dictionnaire Python.
        client_dict = json.loads(client.client_data)

        # On ajoute l'identifiant du client dans le dictionnaire.
        client_dict["SK_ID_CURR"] = client.sk_id_curr

        # Conversion en DataFrame pandas.
        df = pd.DataFrame([client_dict])

        return df

    finally:
        # Ferme toujours la connexion.
        session.close()
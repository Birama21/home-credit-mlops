"""
Import des clients de production dans PostgreSQL.

Ce script lit le fichier CSV complet de test et insère les clients
dans la table clients_demo. Il est destiné à être lancé en local,
pas sur Hugging Face.
"""

import json
import re
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from src.database.database import SessionLocal
from src.database.models import ClientDemo


BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "production" / "production_test_clients.csv"
BATCH_SIZE = 1000


def clean_feature_names(columns):
    """Nettoie les noms de colonnes comme dans l'API."""

    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


def load_clients_csv() -> pd.DataFrame:
    """Charge le CSV complet et prépare les noms de colonnes."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = clean_feature_names(df.columns)

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("La colonne SK_ID_CURR est absente du fichier.")

    return df


def import_clients() -> None:
    """Importe les clients dans PostgreSQL."""

    df = load_clients_csv()

    total_rows = len(df)
    print(f"📄 {total_rows} clients trouvés dans le CSV.")

    with SessionLocal() as session:
        for start in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[start : start + BATCH_SIZE]

            records = []

            for _, row in batch.iterrows():
                sk_id_curr = int(row["SK_ID_CURR"])

                client_data = row.drop(labels=["SK_ID_CURR"]).where(
                    pd.notna(row.drop(labels=["SK_ID_CURR"])),
                    None,
                )

                records.append(
                    {
                        "sk_id_curr": sk_id_curr,
                        "client_data": json.dumps(client_data.to_dict()),
                    }
                )

            stmt = insert(ClientDemo).values(records)

            stmt = stmt.on_conflict_do_update(
                index_elements=["sk_id_curr"],
                set_={"client_data": stmt.excluded.client_data},
            )

            session.execute(stmt)
            session.commit()

            print(f"✅ Clients importés : {min(start + BATCH_SIZE, total_rows)} / {total_rows}")

    print("🎉 Import terminé avec succès.")


if __name__ == "__main__":
    import_clients()
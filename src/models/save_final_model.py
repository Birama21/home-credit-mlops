# ============================================================
# Sauvegarde du modèle final LightGBM pour déploiement API
#
# Objectifs :
# - réentraîner le modèle final validé
# - sauvegarder le modèle avec joblib
# - sauvegarder les métadonnées du modèle
# - sauvegarder le jeu de test comme données de production simulées
# ============================================================

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_final.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

PRODUCTION_DIR = BASE_DIR / "data" / "production"
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTION_PARQUET_PATH = PRODUCTION_DIR / "production_test_clients.parquet"
PRODUCTION_CSV_PATH = PRODUCTION_DIR / "production_test_clients.csv"

# ============================================================
# Données de référence pour le monitoring du data drift
# ============================================================

REFERENCE_DIR = BASE_DIR / "data" / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PARQUET_PATH = (
    REFERENCE_DIR / "reference_train_clients.parquet"
)

REFERENCE_CSV_PATH = (
    REFERENCE_DIR / "reference_train_clients.csv"
)


REFERENCE_STATS_PATH = REFERENCE_DIR / "reference_stats.json"

DRIFT_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "EXT_SOURCE_2",
]

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.51

FN_COST = 10
FP_COST = 1


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    """Nettoie les noms de colonnes pour LightGBM."""
    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


def compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    """Calcule le coût métier."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    business_cost = (fn_cost * fn) + (fp_cost * fp)

    return business_cost, tn, fp, fn, tp


# ============================================================
# Chargement et préparation des données
# ============================================================

print("Chargement du dataset...")

df = pd.read_parquet(DATA_PATH)

train_df = df[df["TARGET"].notna()].copy()

y = train_df["TARGET"].astype(int)

X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

X.columns = clean_feature_names(X.columns)

print("X shape :", X.shape)
print("y shape :", y.shape)


# ============================================================
# Split train / test final
# ============================================================

X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ============================================================
# Sauvegarde du jeu d'entraînement comme référence
# pour le monitoring du data drift
# ============================================================

reference_train_clients = X_train_full.copy()

reference_train_clients.insert(
    0,
    "SK_ID_CURR",
    train_df.loc[X_train_full.index, "SK_ID_CURR"].values,
)

reference_train_clients["TARGET"] = y_train_full.values

reference_train_clients.to_parquet(
    REFERENCE_PARQUET_PATH,
    index=False,
)

reference_train_clients.to_csv(
    REFERENCE_CSV_PATH,
    index=False,
)

print("\nJeu de référence sauvegardé.")
print("Parquet :", REFERENCE_PARQUET_PATH)
print("CSV     :", REFERENCE_CSV_PATH)
print("Shape   :", reference_train_clients.shape)

# ============================================================
# Sauvegarde des statistiques de référence pour le data drift
# ============================================================

reference_stats = {}

for feature in DRIFT_FEATURES:
    if feature in reference_train_clients.columns:
        reference_stats[feature] = {
            "mean": float(reference_train_clients[feature].mean()),
            "count": int(reference_train_clients[feature].count()),
        }

with open(REFERENCE_STATS_PATH, "w", encoding="utf-8") as file:
    json.dump(reference_stats, file, indent=4)

print("\nStatistiques de référence sauvegardées.")
print("JSON :", REFERENCE_STATS_PATH)


# ============================================================
# Sauvegarde du jeu de test comme données de production simulées
# ============================================================

production_test_clients = X_test_final.copy()
production_test_clients.insert(
    0,
    "SK_ID_CURR",
    train_df.loc[X_test_final.index, "SK_ID_CURR"].values,
)
production_test_clients["TARGET"] = y_test_final.values

production_test_clients.to_parquet(
    PRODUCTION_PARQUET_PATH,
    index=False,
)

production_test_clients.to_csv(
    PRODUCTION_CSV_PATH,
    index=False,
)

print("\nDonnées de production simulées sauvegardées.")
print("Parquet :", PRODUCTION_PARQUET_PATH)
print("CSV     :", PRODUCTION_CSV_PATH)
print("Shape   :", production_test_clients.shape)

print("\nExemples de SK_ID_CURR disponibles :")
print(production_test_clients["SK_ID_CURR"].head(10).tolist())


# ============================================================
# Modèle final issu du GridSearchCV
# ============================================================

params = {
    "learning_rate": 0.05,
    "n_estimators": 200,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

print("\nEntraînement du modèle final...")

model = LGBMClassifier(**params)

model.fit(X_train_full, y_train_full)


# ============================================================
# Vérification rapide sur test final
# ============================================================

y_proba = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

auc = roc_auc_score(y_test_final, y_proba)
accuracy = accuracy_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred, zero_division=0)
recall = recall_score(y_test_final, y_pred, zero_division=0)
f1 = f1_score(y_test_final, y_pred, zero_division=0)

business_cost, tn, fp, fn, tp = compute_business_cost(
    y_test_final,
    y_pred,
    fn_cost=FN_COST,
    fp_cost=FP_COST,
)

print("\nPerformance du modèle sauvegardé")
print(f"AUC           : {auc:.4f}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")
print(f"Business cost : {business_cost}")
print(f"Threshold     : {FINAL_THRESHOLD}")


# ============================================================
# Sauvegarde du modèle et des métadonnées
# ============================================================

joblib.dump(model, MODEL_PATH)

metadata = {
    "model_name": "LightGBM final scoring model",
    "model_path": str(MODEL_PATH),
    "threshold": FINAL_THRESHOLD,
    "production_data_path": str(PRODUCTION_PARQUET_PATH),
    "features": list(X.columns),
    "params": params,
    "metrics": {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "business_cost": int(business_cost),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nModèle sauvegardé dans :", MODEL_PATH)
print("Métadonnées sauvegardées dans :", METADATA_PATH)
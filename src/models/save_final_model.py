# ============================================================
# Sauvegarde du modèle final LightGBM pour déploiement API
#
# Objectif :
# - réentraîner le modèle final validé
# - sauvegarder le modèle avec joblib
# - sauvegarder le seuil métier et les métadonnées
# ============================================================

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, accuracy_score, confusion_matrix


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_final.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

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

# Conversion des colonnes object restantes en numérique
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remplacement des valeurs infinies par NaN
X = X.replace([np.inf, -np.inf], np.nan)

# LightGBM gère les NaN
X.columns = clean_feature_names(X.columns)

print("X shape :", X.shape)
print("y shape :", y.shape)


# ============================================================
# Split test final pour vérifier la performance sauvegardée
# ============================================================

X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


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
# Sauvegarde du modèle
# ============================================================

joblib.dump(model, MODEL_PATH)

metadata = {
    "model_name": "LightGBM final scoring model",
    "model_path": str(MODEL_PATH),
    "threshold": FINAL_THRESHOLD,
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
# ============================================================
# Optimisation du seuil métier pour LightGBM
#
# Objectif :
# - entraîner le meilleur modèle retenu : LightGBM balanced
# - prédire les probabilités sur le test final
# - tester plusieurs seuils entre 0.10 et 0.90
# - choisir le seuil qui minimise le coût métier
# - tracer la courbe Business Cost vs Threshold
# - logger les résultats dans MLflow
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.lightgbm

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Configuration générale
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "Home Credit - Model Experiments"
RUN_NAME = "lightgbm_threshold_optimization_v1"

RANDOM_STATE = 42

FN_COST = 10
FP_COST = 1

OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PLOT_PATH = OUTPUT_DIR / "lightgbm_business_cost_vs_threshold.png"


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    """Nettoie les noms de colonnes pour éviter les erreurs LightGBM."""
    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


def compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    """Calcule le coût métier à partir de la matrice de confusion."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    business_cost = (fn_cost * fn) + (fp_cost * fp)

    return business_cost, tn, fp, fn, tp


def compute_metrics_for_threshold(y_true, y_proba, threshold):
    """Calcule les métriques pour un seuil donné."""
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    business_cost, tn, fp, fn, tp = compute_business_cost(
        y_true,
        y_pred,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )

    return {
        "threshold": threshold,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "business_cost": business_cost,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# Chargement des données
# ============================================================

print("Chargement du dataset...")

df = pd.read_parquet(DATA_PATH)

# On conserve uniquement les lignes avec TARGET
train_df = df[df["TARGET"].notna()].copy()

# Variable cible
y = train_df["TARGET"].astype(int)

# Variables explicatives
X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])

# Conversion des colonnes object restantes en numérique
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remplacement des valeurs infinies par NaN
X = X.replace([np.inf, -np.inf], np.nan)

# LightGBM gère les NaN, donc on les conserve
X.columns = clean_feature_names(X.columns)

print("X shape :", X.shape)
print("y shape :", y.shape)
print("Taux moyen de NaN :", round(X.isna().mean().mean(), 4))


# ============================================================
# Split train final / test final
# ============================================================

X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nSplit final")
print("X_train_full :", X_train_full.shape)
print("X_test_final :", X_test_final.shape)

print("\nDistribution Train")
print(y_train_full.value_counts(normalize=True))

print("\nDistribution Test")
print(y_test_final.value_counts(normalize=True))


# ============================================================
# Modèle LightGBM retenu
# ============================================================

params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


# ============================================================
# MLflow + optimisation du seuil
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    # Logging des paramètres
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("optimization", "threshold")
    mlflow.log_param("fn_cost", FN_COST)
    mlflow.log_param("fp_cost", FP_COST)
    mlflow.log_param("threshold_min", 0.10)
    mlflow.log_param("threshold_max", 0.90)
    mlflow.log_param("threshold_step", 0.01)

    for key, value in params.items():
        mlflow.log_param(key, value)

    # ========================================================
    # Entraînement du modèle retenu
    # ========================================================

    print("\nEntraînement LightGBM...")

    model = LGBMClassifier(**params)

    model.fit(X_train_full, y_train_full)

    # Probabilités sur le test final
    y_test_proba = model.predict_proba(X_test_final)[:, 1]

    # AUC indépendante du seuil
    test_auc = roc_auc_score(y_test_final, y_test_proba)

    print(f"Test AUC : {test_auc:.4f}")

    # ========================================================
    # Test de plusieurs seuils
    # ========================================================

    thresholds = np.arange(0.10, 0.91, 0.01)

    threshold_results = []

    for threshold in thresholds:
        metrics = compute_metrics_for_threshold(
            y_true=y_test_final,
            y_proba=y_test_proba,
            threshold=threshold,
        )

        threshold_results.append(metrics)

    results_df = pd.DataFrame(threshold_results)

    # Seuil optimal = coût métier minimal
    best_row = results_df.loc[
        results_df["business_cost"].idxmin()
    ]

    best_threshold = best_row["threshold"]
    best_business_cost = best_row["business_cost"]

    print("\nMeilleur seuil métier")
    print(f"Best threshold     : {best_threshold:.2f}")
    print(f"Business cost      : {best_business_cost:.0f}")
    print(f"Recall             : {best_row['recall']:.4f}")
    print(f"Precision          : {best_row['precision']:.4f}")
    print(f"F1-score           : {best_row['f1']:.4f}")
    print(f"Accuracy           : {best_row['accuracy']:.4f}")
    print(f"AUC                : {best_row['auc']:.4f}")
    print(f"False Negatives    : {best_row['fn']:.0f}")
    print(f"False Positives    : {best_row['fp']:.0f}")

    # ========================================================
    # Logging MLflow des meilleurs résultats
    # ========================================================

    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_business_cost", best_business_cost)
    mlflow.log_metric("best_accuracy", best_row["accuracy"])
    mlflow.log_metric("best_precision", best_row["precision"])
    mlflow.log_metric("best_recall", best_row["recall"])
    mlflow.log_metric("best_f1", best_row["f1"])
    mlflow.log_metric("best_tn", best_row["tn"])
    mlflow.log_metric("best_fp", best_row["fp"])
    mlflow.log_metric("best_fn", best_row["fn"])
    mlflow.log_metric("best_tp", best_row["tp"])

    # ========================================================
    # Sauvegarde du tableau des seuils
    # ========================================================

    threshold_results_path = OUTPUT_DIR / "lightgbm_threshold_results.csv"

    results_df.to_csv(threshold_results_path, index=False)

    mlflow.log_artifact(str(threshold_results_path))

    # ========================================================
    # Courbe coût métier vs seuil
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["threshold"],
        results_df["business_cost"],
        marker="o",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}",
    )

    plt.title("Business Cost vs Decision Threshold - LightGBM")
    plt.xlabel("Decision threshold")
    plt.ylabel("Business cost")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(THRESHOLD_PLOT_PATH)
    plt.close()

    mlflow.log_artifact(str(THRESHOLD_PLOT_PATH))

    # ========================================================
    # Sauvegarde du modèle
    # ========================================================

    mlflow.lightgbm.log_model(
        model,
        name="model",
    )

print("\nOptimisation du seuil terminée.")
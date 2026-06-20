# ============================================================
# Optimisation LightGBM avec GridSearchCV + seuil métier
#
# Objectifs :
# - optimiser quelques hyperparamètres LightGBM
# - évaluer le meilleur modèle sur un test final
# - optimiser le seuil de décision selon le coût métier
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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)


# ============================================================
# Configuration générale
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "Home Credit - Model Experiments"
RUN_NAME = "lightgbm_gridsearch_threshold_v1"

RANDOM_STATE = 42

FN_COST = 10
FP_COST = 1

OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PLOT_PATH = OUTPUT_DIR / "lightgbm_gridsearch_business_cost_vs_threshold.png"
GRID_RESULTS_PATH = OUTPUT_DIR / "lightgbm_gridsearch_results.csv"
THRESHOLD_RESULTS_PATH = OUTPUT_DIR / "lightgbm_gridsearch_threshold_results.csv"


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
    """Calcule le coût métier : FN coûte plus cher que FP."""
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

train_df = df[df["TARGET"].notna()].copy()

y = train_df["TARGET"].astype(int)

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
# Modèle de base + grille raisonnable
# ============================================================

base_model = LGBMClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [31, 63],
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# On optimise d'abord l'AUC avec GridSearchCV.
# Le seuil métier sera optimisé ensuite.
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    verbose=2,
    return_train_score=True,
)


# ============================================================
# MLflow + GridSearchCV + optimisation du seuil
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("optimization", "GridSearchCV + threshold")
    mlflow.log_param("grid_scoring", "roc_auc")
    mlflow.log_param("cv_splits", 3)
    mlflow.log_param("fn_cost", FN_COST)
    mlflow.log_param("fp_cost", FP_COST)

    print("\nLancement GridSearchCV...")

    grid_search.fit(X_train_full, y_train_full)

    print("\nGridSearch terminé.")

    best_params = grid_search.best_params_
    best_cv_auc = grid_search.best_score_

    print("Best params :", best_params)
    print(f"Best CV AUC : {best_cv_auc:.4f}")

    # Logging des meilleurs paramètres
    for key, value in best_params.items():
        mlflow.log_param(f"best_{key}", value)

    mlflow.log_metric("best_cv_auc", best_cv_auc)

    # Sauvegarde des résultats complets du GridSearch
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv(GRID_RESULTS_PATH, index=False)
    mlflow.log_artifact(str(GRID_RESULTS_PATH))

    # ========================================================
    # Meilleur modèle sur test final
    # ========================================================

    best_model = grid_search.best_estimator_

    y_test_proba = best_model.predict_proba(X_test_final)[:, 1]

    test_auc = roc_auc_score(y_test_final, y_test_proba)

    print(f"\nTest AUC avec meilleur modèle : {test_auc:.4f}")

    # ========================================================
    # Optimisation du seuil métier
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

    best_row = results_df.loc[
        results_df["business_cost"].idxmin()
    ]

    best_threshold = best_row["threshold"]
    best_business_cost = best_row["business_cost"]

    print("\nMeilleur seuil métier après GridSearch")
    print(f"Best threshold     : {best_threshold:.2f}")
    print(f"Business cost      : {best_business_cost:.0f}")
    print(f"Recall             : {best_row['recall']:.4f}")
    print(f"Precision          : {best_row['precision']:.4f}")
    print(f"F1-score           : {best_row['f1']:.4f}")
    print(f"Accuracy           : {best_row['accuracy']:.4f}")
    print(f"AUC                : {best_row['auc']:.4f}")
    print(f"False Negatives    : {best_row['fn']:.0f}")
    print(f"False Positives    : {best_row['fp']:.0f}")

    # Logging MLflow
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

    # Sauvegarde du tableau des seuils
    results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)
    mlflow.log_artifact(str(THRESHOLD_RESULTS_PATH))

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

    plt.title("Business Cost vs Decision Threshold - LightGBM GridSearch")
    plt.xlabel("Decision threshold")
    plt.ylabel("Business cost")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(THRESHOLD_PLOT_PATH)
    plt.close()

    mlflow.log_artifact(str(THRESHOLD_PLOT_PATH))

    # Sauvegarde du modèle final
    mlflow.lightgbm.log_model(
        best_model,
        name="model",
    )

print("\nGridSearch + optimisation du seuil terminés.")
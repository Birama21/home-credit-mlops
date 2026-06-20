# ============================================================
# XGBoost avec train/test final + validation croisée + MLflow
# Gestion du déséquilibre avec scale_pos_weight
# Score métier : FN coûte 10 fois plus cher que FP
# ============================================================

from pathlib import Path
import re
import gc

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
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
RUN_NAME = "xgboost_balanced_cv_v1"

RANDOM_STATE = 42
N_SPLITS = 3

FN_COST = 10
FP_COST = 1
MISSING_VALUE = -999


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    """Nettoie les noms de colonnes pour éviter les erreurs XGBoost."""
    cleaned_columns = []

    for col in columns:
        clean_col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
        clean_col = clean_col.strip("_")
        cleaned_columns.append(clean_col)

    return cleaned_columns


def compute_business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    """Calcule un coût métier basé sur FP et FN."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    business_cost = (fn_cost * fn) + (fp_cost * fp)

    return business_cost, tn, fp, fn, tp


def compute_metrics(y_true, y_pred, y_proba):
    """Calcule les métriques classiques et métier."""
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

# On conserve uniquement les lignes issues de application_train
# car elles possèdent la variable cible TARGET.
train_df = df[df["TARGET"].notna()].copy()

# Variable cible
y = train_df["TARGET"].astype(int)

# Variables explicatives
X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])

# Conversion des colonnes object restantes en numérique
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remplacement des valeurs infinies
X = X.replace([np.inf, -np.inf], np.nan)

# XGBoost accepte parfois les NaN, mais pour garder un pipeline
# robuste et explicite, on remplace les NaN par une valeur sentinelle.
X = X.fillna(MISSING_VALUE)

# Nettoyage des noms de colonnes
X.columns = clean_feature_names(X.columns)

print("X shape :", X.shape)
print("y shape :", y.shape)
print("Taux moyen de NaN après traitement :", round(X.isna().mean().mean(), 4))


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
print("y_train_full :", y_train_full.shape)
print("y_test_final :", y_test_final.shape)

print("\nDistribution Train")
print(y_train_full.value_counts(normalize=True))

print("\nDistribution Test")
print(y_test_final.value_counts(normalize=True))


# ============================================================
# Gestion du déséquilibre pour XGBoost
# ============================================================

# XGBoost n'utilise pas class_weight="balanced".
# On utilise scale_pos_weight = nombre classe 0 / nombre classe 1.
n_negative = (y_train_full == 0).sum()
n_positive = (y_train_full == 1).sum()

scale_pos_weight = n_negative / n_positive

print("\nScale pos weight :", round(scale_pos_weight, 4))


# ============================================================
# Paramètres du modèle XGBoost
# ============================================================

params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "tree_method": "hist",
}


# ============================================================
# MLflow + validation croisée
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_metrics = []

with mlflow.start_run(run_name=RUN_NAME):

    # Paramètres généraux
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("run_type", "cv_plus_final_test")
    mlflow.log_param("validation", "StratifiedKFold")
    mlflow.log_param("n_splits", N_SPLITS)
    mlflow.log_param("test_size_final", 0.20)
    mlflow.log_param("fn_cost", FN_COST)
    mlflow.log_param("fp_cost", FP_COST)
    mlflow.log_param("missing_value", MISSING_VALUE)

    # Paramètres du modèle
    for key, value in params.items():
        mlflow.log_param(key, value)

    # ========================================================
    # Validation croisée sur train_full uniquement
    # ========================================================

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X_train_full, y_train_full),
        start=1,
    ):
        print(f"\nFold {fold}/{N_SPLITS}")

        X_train = X_train_full.iloc[train_idx]
        X_valid = X_train_full.iloc[valid_idx]

        y_train = y_train_full.iloc[train_idx]
        y_valid = y_train_full.iloc[valid_idx]

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_valid)[:, 1]

        # Seuil par défaut à 0.5 pour cette première version
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_valid, y_pred, y_proba)
        metrics["fold"] = fold

        fold_metrics.append(metrics)

        # Logging MLflow par fold
        for metric_name, metric_value in metrics.items():
            if metric_name != "fold":
                mlflow.log_metric(f"fold_{fold}_{metric_name}", metric_value)

        print(f"AUC           : {metrics['auc']:.4f}")
        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"Precision     : {metrics['precision']:.4f}")
        print(f"Recall        : {metrics['recall']:.4f}")
        print(f"F1-score      : {metrics['f1']:.4f}")
        print(f"Business cost : {metrics['business_cost']}")

        del model, X_train, X_valid, y_train, y_valid
        gc.collect()

    # ========================================================
    # Résultats moyens de validation croisée
    # ========================================================

    metrics_df = pd.DataFrame(fold_metrics)

    mean_auc = metrics_df["auc"].mean()
    std_auc = metrics_df["auc"].std()
    mean_accuracy = metrics_df["accuracy"].mean()
    mean_precision = metrics_df["precision"].mean()
    mean_recall = metrics_df["recall"].mean()
    mean_f1 = metrics_df["f1"].mean()
    mean_business_cost = metrics_df["business_cost"].mean()

    mlflow.log_metric("cv_mean_auc", mean_auc)
    mlflow.log_metric("cv_std_auc", std_auc)
    mlflow.log_metric("cv_mean_accuracy", mean_accuracy)
    mlflow.log_metric("cv_mean_precision", mean_precision)
    mlflow.log_metric("cv_mean_recall", mean_recall)
    mlflow.log_metric("cv_mean_f1", mean_f1)
    mlflow.log_metric("cv_mean_business_cost", mean_business_cost)

    print("\nRésultats moyens CV")
    print(f"Mean AUC           : {mean_auc:.4f}")
    print(f"Std AUC            : {std_auc:.4f}")
    print(f"Mean Accuracy      : {mean_accuracy:.4f}")
    print(f"Mean Precision     : {mean_precision:.4f}")
    print(f"Mean Recall        : {mean_recall:.4f}")
    print(f"Mean F1-score      : {mean_f1:.4f}")
    print(f"Mean Business cost : {mean_business_cost:.2f}")

    # ========================================================
    # Entraînement final sur tout le train_full
    # ========================================================

    print("\nEntraînement final sur train_full...")

    final_model = XGBClassifier(**params)

    final_model.fit(X_train_full, y_train_full)

    # ========================================================
    # Évaluation finale sur test_final
    # ========================================================

    y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    test_metrics = compute_metrics(
        y_test_final,
        y_test_pred,
        y_test_proba,
    )

    # Logging des métriques finales
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)

    print("\nRésultats sur test final")
    print(f"Test AUC           : {test_metrics['auc']:.4f}")
    print(f"Test Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision     : {test_metrics['precision']:.4f}")
    print(f"Test Recall        : {test_metrics['recall']:.4f}")
    print(f"Test F1-score      : {test_metrics['f1']:.4f}")
    print(f"Test Business cost : {test_metrics['business_cost']}")

    # Sauvegarde du modèle final dans MLflow
    mlflow.xgboost.log_model(
        final_model,
        name="model",
    )

print("\nRun MLflow terminé.")
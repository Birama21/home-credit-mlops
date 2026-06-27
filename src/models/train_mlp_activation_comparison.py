# ============================================================
# Comparaison MLP - fonctions d'activation
#
# Objectif :
# - tester un réseau de neurones MLP
# - comparer plusieurs fonctions d'activation
# - éviter un entraînement trop lourd
# - logger les résultats dans MLflow
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "Home Credit - Model Experiments"
RUN_NAME = "mlp_activation_comparison_v1"

RANDOM_STATE = 42

FN_COST = 10
FP_COST = 1

ACTIVATIONS = ["relu", "tanh", "logistic"]


# ============================================================
# Fonctions utilitaires
# ============================================================

def clean_feature_names(columns):
    """Nettoie les noms de colonnes."""
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

# On garde uniquement les lignes avec TARGET
train_df = df[df["TARGET"].notna()].copy()

y = train_df["TARGET"].astype(int)

X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])

# Conversion des colonnes object restantes en numérique
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remplacement des valeurs infinies par NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Nettoyage des noms de colonnes
X.columns = clean_feature_names(X.columns)

print("X shape :", X.shape)
print("y shape :", y.shape)
print("Taux moyen de NaN :", round(X.isna().mean().mean(), 4))


# ============================================================
# Split train / test final
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nSplit final")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

print("\nDistribution Train")
print(y_train.value_counts(normalize=True))

print("\nDistribution Test")
print(y_test.value_counts(normalize=True))


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

results = []

with mlflow.start_run(run_name=RUN_NAME):

    mlflow.log_param("model_family", "MLPClassifier")
    mlflow.log_param("activations_tested", ", ".join(ACTIVATIONS))
    mlflow.log_param("hidden_layer_sizes", "(32,)")
    mlflow.log_param("max_iter", 50)
    mlflow.log_param("early_stopping", True)
    mlflow.log_param("fn_cost", FN_COST)
    mlflow.log_param("fp_cost", FP_COST)

    # ========================================================
    # Boucle sur les fonctions d'activation
    # ========================================================

    for activation in ACTIVATIONS:

        print(f"\nEntraînement MLP avec activation = {activation}")

        # Pipeline obligatoire pour MLP :
        # 1. Imputer les NaN
        # 2. Standardiser les variables
        # 3. Entraîner le réseau de neurones
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation=activation,
                        max_iter=50,
                        early_stopping=True,
                        validation_fraction=0.10,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Seuil par défaut 0.5 pour comparaison simple
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_proba)

        metrics["activation"] = activation

        results.append(metrics)

        # Logging MLflow pour chaque activation
        mlflow.log_metric(f"{activation}_auc", metrics["auc"])
        mlflow.log_metric(f"{activation}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{activation}_precision", metrics["precision"])
        mlflow.log_metric(f"{activation}_recall", metrics["recall"])
        mlflow.log_metric(f"{activation}_f1", metrics["f1"])
        mlflow.log_metric(f"{activation}_business_cost", metrics["business_cost"])

        print(f"AUC           : {metrics['auc']:.4f}")
        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"Precision     : {metrics['precision']:.4f}")
        print(f"Recall        : {metrics['recall']:.4f}")
        print(f"F1-score      : {metrics['f1']:.4f}")
        print(f"Business cost : {metrics['business_cost']}")

    # ========================================================
    # Synthèse des résultats
    # ========================================================

    results_df = pd.DataFrame(results)

    # Meilleure activation selon le coût métier
    best_row = results_df.loc[
        results_df["business_cost"].idxmin()
    ]

    best_activation = best_row["activation"]

    print("\nMeilleure fonction d'activation")
    print(f"Activation retenue : {best_activation}")
    print(f"AUC                : {best_row['auc']:.4f}")
    print(f"Recall             : {best_row['recall']:.4f}")
    print(f"F1-score           : {best_row['f1']:.4f}")
    print(f"Business cost      : {best_row['business_cost']}")

    mlflow.log_param("best_activation", best_activation)
    mlflow.log_metric("best_auc", best_row["auc"])
    mlflow.log_metric("best_recall", best_row["recall"])
    mlflow.log_metric("best_f1", best_row["f1"])
    mlflow.log_metric("best_business_cost", best_row["business_cost"])

    # Sauvegarde du tableau comparatif
    output_path = BASE_DIR / "reports" / "figures" / "mlp_activation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    mlflow.log_artifact(str(output_path))

print("\nComparaison des fonctions d'activation terminée.")
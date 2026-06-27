# ============================================================
# Explicabilité du modèle final LightGBM avec SHAP
#
# Objectifs :
# - entraîner le modèle final LightGBM avec les meilleurs paramètres
# - utiliser le seuil métier final 0.51
# - analyser l'importance globale des variables
# - analyser localement une prédiction client
# - sauvegarder les graphiques dans reports/figures
# - logger les graphiques dans MLflow
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.lightgbm

import shap
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")


# ============================================================
# Configuration générale
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "home_credit_features.parquet"

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "Home Credit - Model Experiments"
RUN_NAME = "lightgbm_shap_explainability_v1"

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.51

FN_COST = 10
FP_COST = 1

OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "lightgbm_feature_importance.png"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "lightgbm_shap_summary.png"
SHAP_WATERFALL_PATH = OUTPUT_DIR / "lightgbm_shap_waterfall_client.png"


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


# ============================================================
# Modèle final LightGBM
# Hyperparamètres issus du GridSearchCV
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

print("\nEntraînement du modèle final LightGBM...")

model = LGBMClassifier(**params)

model.fit(X_train_full, y_train_full)

y_test_proba = model.predict_proba(X_test_final)[:, 1]
y_test_pred = (y_test_proba >= FINAL_THRESHOLD).astype(int)

auc = roc_auc_score(y_test_final, y_test_proba)

business_cost, tn, fp, fn, tp = compute_business_cost(
    y_test_final,
    y_test_pred,
    fn_cost=FN_COST,
    fp_cost=FP_COST,
)

print("\nPerformance modèle final")
print(f"AUC             : {auc:.4f}")
print(f"Threshold       : {FINAL_THRESHOLD}")
print(f"Business cost   : {business_cost}")
print(f"TN              : {tn}")
print(f"FP              : {fp}")
print(f"FN              : {fn}")
print(f"TP              : {tp}")


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    # Paramètres
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("explainability", "SHAP")
    mlflow.log_param("threshold", FINAL_THRESHOLD)
    mlflow.log_param("fn_cost", FN_COST)
    mlflow.log_param("fp_cost", FP_COST)

    for key, value in params.items():
        mlflow.log_param(key, value)

    # Métriques
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("business_cost", business_cost)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("fn", fn)
    mlflow.log_metric("tp", tp)

    # ========================================================
    # 1. Importance globale LightGBM
    # ========================================================

    print("\nCréation du graphique d'importance globale LightGBM...")

    feature_importances = pd.DataFrame(
        {
            "feature": X_train_full.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    top_features = feature_importances.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )
    plt.title("Top 20 Feature Importances - LightGBM")
    plt.xlabel("Importance")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()

    mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))

    feature_importance_csv_path = OUTPUT_DIR / "lightgbm_feature_importance.csv"
    feature_importances.to_csv(feature_importance_csv_path, index=False)
    mlflow.log_artifact(str(feature_importance_csv_path))

    # ========================================================
    # 2. SHAP Global Summary Plot
    # ========================================================

    print("\nCalcul des valeurs SHAP globales...")

    # Pour éviter un calcul trop lourd, on prend un échantillon du test final
    shap_sample_size = min(2000, len(X_test_final))

    X_shap_sample = X_test_final.sample(
        n=shap_sample_size,
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_shap_sample)

    # Selon la version de SHAP, shap_values peut être une liste
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]
    else:
        shap_values_class_1 = shap_values

    plt.figure()
    shap.summary_plot(
        shap_values_class_1,
        X_shap_sample,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(SHAP_SUMMARY_PATH))

    # ========================================================
    # 3. Explicabilité locale d'un client
    # ========================================================

    print("\nCréation d'une explication locale SHAP...")

    # On choisit un client prédit comme risqué si possible
    risky_indices = np.where(y_test_pred == 1)[0]

    if len(risky_indices) > 0:
        client_position = risky_indices[0]
    else:
        client_position = 0

    X_client = X_test_final.iloc[[client_position]]
    y_client_true = y_test_final.iloc[client_position]
    y_client_proba = y_test_proba[client_position]
    y_client_pred = y_test_pred[client_position]

    print("\nClient expliqué")
    print(f"Position client      : {client_position}")
    print(f"Probabilité défaut   : {y_client_proba:.4f}")
    print(f"Seuil métier         : {FINAL_THRESHOLD}")
    print(f"Prédiction           : {y_client_pred}")
    print(f"Vraie classe         : {y_client_true}")

    shap_values_client = explainer.shap_values(X_client)

    if isinstance(shap_values_client, list):
        shap_values_client_class_1 = shap_values_client[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_client_class_1 = shap_values_client
        expected_value = explainer.expected_value

    explanation = shap.Explanation(
        values=shap_values_client_class_1[0],
        base_values=expected_value,
        data=X_client.iloc[0],
        feature_names=X_client.columns,
    )

    plt.figure()
    shap.plots.waterfall(
        explanation,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_WATERFALL_PATH, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(SHAP_WATERFALL_PATH))

    # Logging du client expliqué
    mlflow.log_metric("explained_client_probability", y_client_proba)
    mlflow.log_metric("explained_client_prediction", int(y_client_pred))
    mlflow.log_metric("explained_client_true_class", int(y_client_true))

    # ========================================================
    # Sauvegarde du modèle
    # ========================================================

    mlflow.lightgbm.log_model(
        model,
        name="model",
    )

print("\nExplicabilité LightGBM terminée.")
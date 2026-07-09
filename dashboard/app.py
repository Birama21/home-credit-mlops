import streamlit as st

st.set_page_config(
    page_title="Home Credit MLOps Dashboard",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Home Credit MLOps Dashboard")

st.markdown(
    """
Bienvenue sur le dashboard de monitoring du projet **Home Credit MLOps**.

Ce tableau de bord permet de visualiser les informations principales de l'API de scoring :

- suivi des prédictions ;
- monitoring opérationnel ;
- détection du data drift ;
- test manuel d'une prédiction client.

---

### Architecture

```text
FastAPI + LightGBM
        ↓
PostgreSQL
        ↓
Monitoring / Drift / Profiling
        ↓
Streamlit Dashboard
```

---

### Pages disponibles

- 📊 **Monitoring** : statistiques globales des prédictions.
- 📈 **Data Drift** : suivi des dérives des données.
- 🤖 **Scoring** : test manuel d'un client.

Ce dashboard est totalement indépendant de l'API FastAPI et communique avec celle-ci via des appels HTTP.
"""
)

st.info(
    "⚠️ Lance d'abord ton API FastAPI avant d'utiliser le dashboard."
)
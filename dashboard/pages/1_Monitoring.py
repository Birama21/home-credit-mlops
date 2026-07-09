import streamlit as st
import pandas as pd

from utils import monitoring_stats

st.set_page_config(
    page_title="Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Monitoring des prédictions")

st.markdown(
    """
Cette page affiche les principales statistiques opérationnelles de l'API de scoring.
Les données proviennent de l'endpoint FastAPI `/monitoring/stats`.
"""
)

try:
    stats = monitoring_stats()

    if not stats.get("database_enabled", False):
        st.warning("La base PostgreSQL n'est pas disponible ou le monitoring est désactivé.")
        st.json(stats)
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            label="Total prédictions",
            value=stats.get("total_predictions", 0),
        )

        col2.metric(
            label="Succès",
            value=stats.get("total_success", 0),
        )

        col3.metric(
            label="Erreurs",
            value=stats.get("total_errors", 0),
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            label="Taux d'erreur",
            value=f"{stats.get('error_rate', 0) * 100:.2f} %",
        )

        col5.metric(
            label="Dernière latence",
            value=f"{stats.get('last_latency_ms', 0):.2f} ms",
        )

        col6.metric(
            label="Probabilité moyenne",
            value=f"{stats.get('avg_probability_default', 0):.3f}",
        )

        st.divider()

        risk_data = {
            "Catégorie": ["Clients risqués", "Clients non risqués"],
            "Nombre": [
                stats.get("risky_clients", 0),
                stats.get("non_risky_clients", 0),
            ],
        }

        risk_df = pd.DataFrame(risk_data)

        st.subheader("Répartition des décisions")

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.bar_chart(
                risk_df.set_index("Catégorie")
            )

        with col_table:
            st.dataframe(
                risk_df,
                use_container_width=True,
            )

        st.divider()

        with st.expander("Voir la réponse brute de l'API"):
            st.json(stats)

except Exception as error:
    st.error("Impossible de récupérer les données de monitoring.")
    st.exception(error)
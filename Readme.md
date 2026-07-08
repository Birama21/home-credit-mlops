# Home Credit MLOps

Projet MLOps de scoring du risque de défaut de paiement basé sur le dataset **Home Credit Default Risk**.

Ce projet met en œuvre un pipeline complet de Machine Learning allant de la préparation des données jusqu'au déploiement d'une API de prédiction conteneurisée, avec monitoring, logging des prédictions et analyse des performances.

---

# Objectifs du projet

Ce projet a pour objectifs de :

- préparer les données pour un modèle de scoring ;
- entraîner un modèle de Machine Learning ;
- suivre les expérimentations avec MLflow ;
- déployer le modèle sous forme d'API REST avec FastAPI ;
- conteneuriser l'application avec Docker ;
- enregistrer les prédictions dans PostgreSQL ;
- surveiller les performances du modèle ;
- analyser et optimiser les temps d'inférence.

---

# Architecture du projet

```
home-credit-mlops/
│
├── data/
│   ├── raw/
│   ├── reference/
│   └── processed/
│
├── notebooks/
│
├── reports/
│   └── performance_analysis.md
│
├── src/
│   ├── api/
│   ├── database/
│   ├── models/
│   ├── services/
│   └── monitoring/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Technologies utilisées

- Python 3
- FastAPI
- PostgreSQL
- SQLAlchemy
- Scikit-Learn
- Pandas
- MLflow
- Docker
- Docker Compose
- Hugging Face Spaces

---

# Fonctionnalités

## API REST

Deux endpoints principaux sont disponibles.

### Prédiction individuelle

```
POST /predict
```

Permet de calculer le risque de défaut d'un client à partir de son identifiant.

Exemple de réponse :

```json
{
  "sk_id_curr": 100009,
  "probability_default": 0.070834,
  "prediction": 0,
  "decision": "client_non_risque"
}
```

---

### Prédiction par batch

```
POST /predict_batch
```

Permet de scorer plusieurs clients en une seule requête.

Le nombre de clients est paramétrable.

Exemple :

```
POST /predict_batch?n_clients=200
```

---

# Monitoring

Chaque prédiction est enregistrée dans PostgreSQL afin de conserver un historique d'inférence.

Les informations suivantes sont sauvegardées :

- identifiant client ;
- probabilité prédite ;
- décision ;
- endpoint appelé ;
- temps de réponse ;
- statut (succès ou erreur).

---

# Analyse des performances

Un système de profiling a été ajouté afin de mesurer les performances de l'API.

Les temps suivants sont mesurés :

- récupération des données ;
- préparation des features ;
- inférence du modèle ;
- enregistrement PostgreSQL ;
- temps total de traitement.

Les résultats complets sont disponibles dans :

```
reports/performance_analysis.md
```

---

# Optimisations réalisées

Le profiling a permis d'identifier le principal goulot d'étranglement de l'application.

## Endpoint `/predict`

Après analyse, aucune optimisation supplémentaire n'a été nécessaire.

Le temps de réponse se stabilise autour de **100 ms** après le premier appel.

---

## Endpoint `/predict_batch`

L'analyse a montré que le principal coût provenait de l'enregistrement individuel des prédictions dans PostgreSQL.

L'implémentation a été optimisée en regroupant les insertions dans une transaction unique (`session.add_all()` puis `session.commit()`).

Cette optimisation a permis de :

- réduire le temps d'écriture PostgreSQL d'environ **1,3 seconde** à **moins de 100 ms** ;
- réduire le temps total de traitement d'un batch de **200 clients** d'environ **1,5 seconde** à **220 ms**.

---

# Lancement du projet

## Cloner le dépôt

```bash
git clone <repository_url>
cd home-credit-mlops
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer les services Docker

```bash
docker compose up --build
```

L'API sera accessible à l'adresse :

```
http://localhost:8000
```

Documentation Swagger :

```
http://localhost:8000/docs
```

---

# Déploiement

Le projet est conçu pour être exécuté :

- localement avec Docker ;
- sur GitHub Actions ;
- sur Hugging Face Spaces.

Lorsque PostgreSQL n'est pas disponible (par exemple sur Hugging Face), le système de logging est automatiquement désactivé afin de garantir le fonctionnement de l'API.

---

# Auteur

Projet réalisé dans le cadre de la formation **AI Engineer**.

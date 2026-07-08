# Analyse des performances du modèle

## Objectif

L'objectif de cette étape est d'analyser les performances de l'API de scoring dans un environnement de production simulé afin de :

- identifier les goulots d'étranglement ;
- mesurer le temps d'inférence du modèle ;
- analyser les différentes étapes du pipeline de prédiction ;
- proposer et valider des optimisations basées sur les mesures observées.

---

# Méthodologie

Un système de profiling a été intégré directement dans les endpoints :

- `/predict`
- `/predict_batch`

Chaque étape importante est chronométrée à l'aide de `time.perf_counter()`.

Les temps suivants sont mesurés :

- récupération des données depuis PostgreSQL ;
- préparation des features ;
- inférence du modèle ;
- enregistrement des prédictions dans PostgreSQL ;
- temps total de traitement de la requête.

Cette approche permet d'identifier précisément les parties les plus coûteuses de l'application avant toute optimisation.

---

# Résultats initiaux

Les premières mesures réalisées après l'ajout du profiling servent de référence afin d'identifier les principaux goulots d'étranglement.

## Endpoint `/predict`

| Étape | Temps |
|--------|-------:|
| Chargement du client | 1117.64 ms |
| Préparation des features | 146.89 ms |
| Prédiction du modèle | 22.22 ms |
| Enregistrement PostgreSQL | 26.57 ms |
| Temps total | 1313.37 ms |

---

## Endpoint `/predict_batch`

Test réalisé sur un batch de **200 clients**.

| Étape | Temps |
|--------|-------:|
| Chargement du batch | 1027.05 ms |
| Préparation des features | 26.95 ms |
| Prédiction du modèle | 19.71 ms |
| Écriture PostgreSQL | 2119.15 ms |
| Temps total | 3192.92 ms |

---

# Analyse des performances

Les premières mesures montrent immédiatement que le modèle de Machine Learning n'est pas le principal facteur limitant.

L'inférence du modèle est particulièrement rapide :

- environ **22 ms** pour une prédiction individuelle ;
- environ **20 ms** pour un batch de 200 clients.

À l'inverse, les opérations d'entrée/sortie représentent la majeure partie du temps de traitement.

Les deux coûts principaux sont :

- la lecture des données depuis PostgreSQL ;
- l'enregistrement des prédictions dans PostgreSQL.

Pour le endpoint `/predict_batch`, l'écriture des logs représente plus de **2 secondes**, soit la majorité du temps total de traitement.

---

# Goulot d'étranglement identifié

Le principal goulot d'étranglement est l'enregistrement des prédictions dans PostgreSQL.

Dans l'implémentation initiale, chaque prédiction du batch était enregistrée individuellement via un appel à `log_prediction()`.

Pour un batch de 200 clients, cela entraînait :

- 200 insertions SQL ;
- 200 transactions ;
- 200 validations (`commit()`).

Cette stratégie est simple mais peu adaptée à un traitement par lots.

---

# Optimisation du endpoint `/predict`

Afin de vérifier si le premier appel influençait les performances, plusieurs exécutions successives du endpoint ont été réalisées.

Les résultats obtenus sont les suivants :

| Appel | get_client_data | prepare_features | model_prediction | database_logging | Temps total |
|-------|----------------:|-----------------:|-----------------:|-----------------:|------------:|
| Premier appel | 138.10 ms | 93.44 ms | 9.44 ms | 15.66 ms | 256.68 ms |
| Deuxième appel | 18.09 ms | 73.56 ms | 4.43 ms | 7.22 ms | 103.33 ms |
| Troisième appel | 21.62 ms | 80.12 ms | 3.45 ms | 5.97 ms | 111.21 ms |

## Interprétation

Le premier appel est plus lent que les suivants.

Ce comportement correspond à un **cold start**.

Lors de la première requête, plusieurs composants sont initialisés :

- première connexion PostgreSQL ;
- initialisation de SQLAlchemy ;
- chargement des ressources Python nécessaires.

Une fois cette phase terminée, les performances deviennent stables.

Les appels suivants présentent un temps de réponse d'environ **100 ms**.

## Conclusion

Aucun goulot d'étranglement significatif n'a été identifié sur le endpoint `/predict`.

Le temps d'inférence du modèle reste inférieur à **5 ms**, ce qui confirme que le modèle de Machine Learning n'est pas le facteur limitant.

Le temps total est principalement lié aux opérations d'accès aux données et reste inférieur à **120 ms** après l'initialisation de l'application.

Aucune optimisation supplémentaire n'a donc été jugée nécessaire sur ce endpoint.

---

# Optimisation du endpoint `/predict_batch`

Trois exécutions successives du endpoint `/predict_batch` ont été réalisées avec un batch de **200 clients**.

Les résultats montrent que le premier appel est légèrement plus lent en raison de l'initialisation des connexions et des caches.

En revanche, un problème important apparaît : le temps consacré à l'enregistrement des prédictions dans PostgreSQL reste largement supérieur aux autres étapes.

Les temps observés sont les suivants :

- Chargement des données : environ **120 à 180 ms** ;
- Préparation des features : **5 à 7 ms** ;
- Inférence du modèle : **4 à 8 ms** ;
- Enregistrement PostgreSQL : **1,2 à 1,4 seconde**.

Cette analyse confirme que le modèle de Machine Learning n'est pas responsable des performances globales.

Le véritable goulot d'étranglement provient des écritures répétées dans PostgreSQL.

---

# Optimisation réalisée

Afin de réduire le coût des écritures en base de données, une nouvelle fonction `log_predictions_batch()` a été développée.

Au lieu d'effectuer une insertion SQL pour chaque client, toutes les prédictions sont désormais regroupées dans une seule transaction PostgreSQL grâce à :

- `session.add_all()`
- un unique `session.commit()`

Cette approche réduit considérablement le nombre d'allers-retours entre l'application et PostgreSQL tout en conservant exactement le même comportement fonctionnel.

---

# Résultats après optimisation

Les mesures obtenues après cette optimisation montrent une amélioration importante des performances.

| Étape | Avant optimisation | Après optimisation |
|--------|-------------------:|-------------------:|
| Chargement des données | ~130 ms | ~120 ms |
| Préparation des features | 5 à 11 ms | 4 à 7 ms |
| Prédiction du modèle | 4 à 11 ms | 4 à 8 ms |
| Logging PostgreSQL | **1200 à 1400 ms** | **90 à 114 ms** |
| Temps total | **1360 à 1580 ms** | **220 à 310 ms** |

L'enregistrement des prédictions est désormais près de **10 fois plus rapide**.

Le temps total d'exécution du batch est réduit d'environ **1,5 seconde** à moins de **300 ms**.

---

# Conclusion

Cette étape a permis d'analyser précisément les performances de l'API de scoring grâce à l'intégration d'un système de profiling.

Les mesures ont montré que le modèle de Machine Learning n'était pas le principal facteur limitant. Les temps d'inférence restent très faibles, de l'ordre de quelques millisecondes, aussi bien pour une prédiction individuelle que pour un batch de plusieurs centaines de clients.

Le principal goulot d'étranglement provenait des opérations d'entrée/sortie avec PostgreSQL, plus précisément de l'enregistrement individuel de chaque prédiction dans le endpoint `/predict_batch`.

La mise en place d'une insertion groupée (`session.add_all()` suivi d'un unique `commit()`) a permis de réduire très fortement le coût des écritures en base de données.

Les performances globales de l'API ont ainsi été significativement améliorées sans modifier son comportement fonctionnel.

Cette démarche illustre l'intérêt d'une approche basée sur des mesures objectives avant toute optimisation et répond pleinement aux objectifs fixés pour cette étape du projet.
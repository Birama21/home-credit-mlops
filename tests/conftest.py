# ============================================================
# Configuration Pytest
#
# Objectif :
# - ajouter la racine du projet au PYTHONPATH
# - permettre aux tests d'importer les modules du dossier src
# ============================================================

import sys
from pathlib import Path


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ajout de la racine au chemin Python
sys.path.insert(0, str(PROJECT_ROOT))
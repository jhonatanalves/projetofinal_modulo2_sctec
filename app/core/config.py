"""Configurações centrais da aplicação.

Todos os caminhos são construídos a partir da localização deste arquivo,
então o projeto roda em qualquer máquina sem caminhos absolutos.
"""
import os
from pathlib import Path

# app/core/config.py -> parents[0]=core, [1]=app, [2]=raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.joblib"))
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

API_TITLE = "API de Previsão de Churn"
API_VERSION = "1.0.0"
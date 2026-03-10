"""
utils/io.py — Salvar e carregar artefatos do modelo.

Centraliza toda I/O em um lugar só.
"""

import os
import joblib
from config import OUTPUT_DIR


def salvar_modelo(modelo, feature_names: list, threshold: float,
                  metricas: dict, caminho: str = None):
    """Serializa modelo + metadados em um único arquivo .pkl."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    caminho = caminho or f"{OUTPUT_DIR}/modelo_cardiovascular.pkl"

    payload = {
        "modelo":        modelo,
        "feature_names": feature_names,
        "threshold":     threshold,
        "metricas":      metricas,
    }
    joblib.dump(payload, caminho)
    print(f"✓ Modelo salvo: {caminho}")
    return caminho


def carregar_modelo(caminho: str) -> dict:
    """Carrega modelo e metadados. Retorna dicionário completo."""
    payload = joblib.load(caminho)
    print(f"✓ Modelo carregado: {caminho}")
    print(f"  ROC-AUC  : {payload['metricas'].get('roc_auc', 'N/A'):.4f}")
    print(f"  Threshold: {payload['threshold']:.4f}")
    return payload

"""
evaluation/metrics.py — Cálculo de todas as métricas clínicas.

Responsabilidade exclusiva: receber y_true e y_prob (ou y_pred)
e devolver dicionários de métricas. Nenhuma lógica de modelo aqui.

Referências
-----------
- Steyerberg et al. (2010, Epidemiology) — validação de modelos clínicos
- Collins et al. (2015, BMJ) — checklist TRIPOD
- Harrell FE (2015) — Regression Modeling Strategies
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import N_CV_FOLDS, RANDOM_STATE


# ==============================================================================
# MÉTRICAS COMPLETAS
# ==============================================================================

def calcular_metricas(y_true, y_prob, y_pred, threshold: float) -> dict:
    """
    Retorna dicionário com todas as métricas relevantes para publicação clínica.

    Inclui métricas de discriminação, calibração e performance binária.
    Segue o checklist TRIPOD (Collins et al., 2015, BMJ).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc   = (tp + tn) / n

    return {
        # Discriminação
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "pr_auc":        average_precision_score(y_true, y_prob),

        # Calibração
        "brier_score":   brier_score_loss(y_true, y_prob),

        # Performance binária
        "sensibilidade": sens,
        "especificidade": spec,
        "ppv":           ppv,
        "npv":           npv,
        "acuracia":      acc,
        "youden_index":  sens + spec - 1,

        # Matriz de confusão
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),

        # Meta
        "threshold": threshold,
        "n_amostras": n,
    }


def imprimir_metricas(metricas: dict):
    """Exibe métricas formatadas para console."""
    m = metricas
    print("\n" + "=" * 60)
    print("MÉTRICAS DE AVALIAÇÃO DO MODELO")
    print(f"(threshold = {m['threshold']:.4f} | n = {m['n_amostras']})")
    print("=" * 60)

    print(f"\n{'--- DISCRIMINAÇÃO ---'}")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}   (>0.80 = bom | >0.90 = excelente)")
    print(f"  PR-AUC   : {m['pr_auc']:.4f}   (útil em datasets desbalanceados)")

    print(f"\n{'--- CALIBRAÇÃO ---'}")
    print(f"  Brier Score : {m['brier_score']:.4f}   (<0.15 = bom | 0.25 = aleatório)")

    print(f"\n{'--- PERFORMANCE BINÁRIA ---'}")
    print(f"  Sensibilidade : {m['sensibilidade']:.4f}   (recall, TPR)")
    print(f"  Especificidade: {m['especificidade']:.4f}   (TNR)")
    print(f"  VPP           : {m['ppv']:.4f}   (precisão)")
    print(f"  VPN           : {m['npv']:.4f}")
    print(f"  Acurácia      : {m['acuracia']:.4f}")
    print(f"  Youden Index  : {m['youden_index']:.4f}")

    print(f"\n{'--- MATRIZ DE CONFUSÃO ---'}")
    print(f"  VP: {m['tp']:5d}   FP: {m['fp']:5d}")
    print(f"  FN: {m['fn']:5d}   VN: {m['tn']:5d}")
    print(f"  Falsos Negativos (não detectados): {m['fn']}  ← crítico em triagem CV")


def calcular_ic_cv(modelo, X, y, n_folds: int = None) -> dict:
    """
    Validação cruzada estratificada com intervalo de confiança (IC 95%).

    O IC 95% é calculado como mean ± 1.96 × std, assumindo normalidade
    assintótica das distribuições de ROC-AUC em grandes amostras.

    Referência: Hanley & McNeil (1982, Radiology)
    """
    n_folds = n_folds or N_CV_FOLDS
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    resultado = {
        "media":  scores.mean(),
        "std":    scores.std(),
        "ic95_low":  scores.mean() - 1.96 * scores.std(),
        "ic95_high": scores.mean() + 1.96 * scores.std(),
        "scores": scores.tolist(),
    }

    print(f"\n{'--- VALIDAÇÃO CRUZADA ({n_folds}-fold estratificado) ---'}")
    print(f"  ROC-AUC: {resultado['media']:.4f} ± {resultado['std']:.4f}")
    print(f"  IC 95%:  [{resultado['ic95_low']:.4f},  {resultado['ic95_high']:.4f}]")

    return resultado


def exportar_metricas(metricas: dict, cv_resultado: dict, caminho: str):
    """Salva métricas em CSV para registro e reprodutibilidade."""
    linha = {**metricas, **{f"cv_{k}": v for k, v in cv_resultado.items()
                            if k != "scores"}}
    pd.DataFrame([linha]).to_csv(caminho, index=False)
    print(f"✓ Métricas exportadas: {caminho}")

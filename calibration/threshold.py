"""
calibration/threshold.py — Otimização e análise de threshold clínico.

Responsabilidade exclusiva: encontrar e comparar o melhor threshold
de decisão para converter probabilidades em classificações binárias.

Referências
-----------
- Youden WJ (1950). Index for rating diagnostic tests. Cancer 3(1):32-35.
- Perkins NJ & Schisterman EF (2006). The inconsistency of "optimal"
  cutpoints. Am J Epidemiol 163(7):670-675.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from config import YOUDEN_FN_WEIGHT


# ==============================================================================
# OTIMIZAÇÃO
# ==============================================================================

def otimizar_youden(y_true, y_prob, fn_weight: float = None) -> float:
    """
    Calcula o threshold ótimo pelo Índice de Youden ponderado.

    J(w) = w × Sensibilidade + Especificidade - 1
         = w × TPR - FPR

    Com w > 1, penalizamos mais os Falsos Negativos — adequado para
    triagem cardiovascular, onde não detectar risco é pior que
    alarme falso.

    Parâmetros
    ----------
    fn_weight : float
        Peso para sensibilidade. Default = YOUDEN_FN_WEIGHT (config.py).
        w=1.0 → Youden clássico (sem preferência)
        w=2.0 → falso negativo custa 2× mais que falso positivo
    """
    w = fn_weight if fn_weight is not None else YOUDEN_FN_WEIGHT

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_w = w * tpr - fpr
    idx      = np.argmax(youden_w)

    threshold = float(thresholds[idx])
    j_valor   = float(youden_w[idx])

    print(f"\n{'='*50}")
    print(f"OTIMIZAÇÃO DE THRESHOLD (Youden ponderado, w={w})")
    print(f"{'='*50}")
    print(f"  Threshold padrão  : 0.500")
    print(f"  Threshold Youden  : {threshold:.4f}")
    print(f"  Índice J (w={w:.1f}) : {j_valor:.4f}")
    print(f"  Sensibilidade     : {tpr[idx]:.4f}")
    print(f"  Especificidade    : {1 - fpr[idx]:.4f}")

    return threshold


# ==============================================================================
# ANÁLISE COMPARATIVA DE THRESHOLDS
# ==============================================================================

def comparar_thresholds(y_true, y_prob) -> dict:
    """
    Calcula métricas para quatro estratégias de threshold diferentes.

    Útil para escolher o threshold com base no contexto clínico.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Youden clássico (w=1)
    j1   = tpr - fpr
    t_j1 = float(thresholds[np.argmax(j1)])

    # Youden ponderado (w=2, prioriza sensibilidade)
    j2   = 2 * tpr - fpr
    t_j2 = float(thresholds[np.argmax(j2)])

    # Maximiza F1 (equilíbrio precisão-recall)
    t_f1 = _threshold_max_f1(y_true, y_prob)

    # Padrão (referência bibliográfica)
    t_pad = 0.50

    tabela = {
        "Padrão (0.50)":            t_pad,
        "Youden clássico (w=1)":    t_j1,
        "Youden ponderado (w=2)":   t_j2,
        "Máximo F1":                t_f1,
    }

    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE ESTRATÉGIAS DE THRESHOLD")
    print("=" * 70)
    print(f"{'Estratégia':<28} {'Threshold':>10} {'Sensib.':>9} "
          f"{'Espec.':>9} {'VPP':>8} {'FN':>6}")
    print("-" * 70)

    metricas = {}
    for nome, t in tabela.items():
        m = _calcular_metricas_threshold(y_true, y_prob, t)
        print(f"{nome:<28} {t:>10.4f} {m['sensibilidade']:>9.4f} "
              f"{m['especificidade']:>9.4f} {m['ppv']:>8.4f} {m['fn']:>6}")
        metricas[nome] = {"threshold": t, **m}

    return metricas


def plotar_curva_threshold(y_true, y_prob, threshold_escolhido: float,
                           salvar_em: str = None):
    """
    Plota Sensibilidade e Especificidade em função do threshold.
    Marca visualmente o ponto escolhido pelo Youden.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    especificidade = 1 - fpr

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, tpr,           "b-", lw=2, label="Sensibilidade (TPR)")
    ax.plot(thresholds, especificidade,"r-", lw=2, label="Especificidade (TNR)")
    ax.plot(thresholds, tpr - fpr,     "g--",lw=1, label="Índice de Youden (J)")

    ax.axvline(x=threshold_escolhido, color="black", linestyle="--", lw=1.5,
               label=f"Threshold escolhido = {threshold_escolhido:.3f}")
    ax.axvline(x=0.50, color="gray", linestyle=":", lw=1, label="Threshold padrão = 0.50")

    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Valor da Métrica")
    ax.set_title("Sensibilidade vs Especificidade por Threshold\n"
                 "Referência: Youden (1950), Cancer")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"✓ Gráfico de threshold salvo: {salvar_em}")
    plt.close()


# ==============================================================================
# HELPERS INTERNOS
# ==============================================================================

def _calcular_metricas_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensibilidade": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "especificidade": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "fn": int(fn),
        "fp": int(fp),
    }


def _threshold_max_f1(y_true, y_prob) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])

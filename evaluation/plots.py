"""
evaluation/plots.py — Todas as visualizações de validação do modelo.

Responsabilidade exclusiva: gerar figuras a partir de y_true / y_prob / y_pred.
Nenhuma lógica de modelo ou métricas aqui.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve


def painel_validacao(y_true, y_prob, y_pred,
                     feature_names: list,
                     importancias: np.ndarray = None,
                     salvar_em: str = None):
    """
    Painel completo de validação com 6 subplots.
    Segue padrões visuais de publicação científica (600 dpi para journals).
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Validação do Modelo Preditivo de Risco Cardiovascular\n"
        "Referência: TRIPOD Statement (Collins et al., 2015, BMJ)",
        fontsize=13, fontweight="bold", y=0.99
    )

    _plot_roc(fig.add_subplot(2, 3, 1), y_true, y_prob)
    _plot_pr(fig.add_subplot(2, 3, 2), y_true, y_prob)
    _plot_calibration(fig.add_subplot(2, 3, 3), y_true, y_prob)
    _plot_importancias(fig.add_subplot(2, 3, 4), feature_names, importancias)
    _plot_distribuicao(fig.add_subplot(2, 3, 5), y_true, y_prob)
    _plot_confusion(fig.add_subplot(2, 3, 6), y_true, y_pred)

    plt.tight_layout()

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"✓ Painel de validação salvo: {salvar_em}")
    plt.close()


# ==============================================================================
# SUBPLOTS INDIVIDUAIS (exportados para uso avulso)
# ==============================================================================

def _plot_roc(ax, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, "b-", lw=2, label=f"Ensemble (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
    ax.fill_between(fpr, tpr, alpha=0.08, color="blue")
    ax.set_xlabel("Taxa de Falso Positivo (1 − Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (Sensibilidade)")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_pr(ax, y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    ax.plot(recall, precision, "g-", lw=2, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall (Sensibilidade)")
    ax.set_ylabel("Precisão (VPP)")
    ax.set_title("Curva Precision-Recall\n(melhor para datasets desbalanceados)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_calibration(ax, y_true, y_prob):
    frac, mean = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    ax.plot(mean, frac, "rs-", lw=2, label=f"Modelo (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Calibração Perfeita")
    ax.fill_between(mean, frac, mean, alpha=0.15, color="red", label="Desvio")
    ax.set_xlabel("Probabilidade Predita Média")
    ax.set_ylabel("Fração de Positivos Observados")
    ax.set_title("Reliability Diagram\n(calibração das probabilidades)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_importancias(ax, feature_names, importancias):
    if importancias is None:
        ax.text(0.5, 0.5, "Importâncias\nnão disponíveis",
                ha="center", va="center", transform=ax.transAxes)
        return
    top_n = 15
    idx = np.argsort(importancias)[-top_n:]
    ax.barh(range(len(idx)), importancias[idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(True, alpha=0.3, axis="x")


def _plot_distribuicao(ax, y_true, y_prob):
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, color="blue",
            label="Sem Risco (real)", density=True)
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, color="red",
            label="Com Risco (real)", density=True)
    ax.set_xlabel("Probabilidade de Risco Predita")
    ax.set_ylabel("Densidade")
    ax.set_title("Separação das Distribuições\n(maior sobreposição = pior discriminação)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_confusion(ax, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
                xticklabels=["Sem Risco", "Com Risco"],
                yticklabels=["Sem Risco", "Com Risco"])
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão (normalizada)")

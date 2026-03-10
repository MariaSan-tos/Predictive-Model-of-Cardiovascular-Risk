"""
calibration/calibrator.py — Calibração probabilística do modelo.

Responsabilidade exclusiva: garantir que P(risco=1) predita
reflita a frequência real de eventos (calibração).

Referências
-----------
- Platt (1999) — Probabilistic outputs for SVMs
- Niculescu-Mizil & Caruana (2005, ICML) — Calibração de modelos
- Guo et al. (2017, ICML) — Temperature Scaling
- Brier (1950) — Brier Score como medida de calibração
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from config import CALIBRATION_METHOD, N_CV_FOLDS, RANDOM_STATE


# ==============================================================================
# DIAGNÓSTICO — ver se calibração é necessária
# ==============================================================================

def diagnosticar_calibracao(modelo_bruto, X_val, y_val, nome_modelo: str = "Modelo"):
    """
    Plota Reliability Diagram e calcula métricas de calibração
    ANTES de qualquer correção.

    Use isso para confirmar que o modelo está mal calibrado antes
    de aplicar Platt Scaling ou Isotonic Regression.

    Interpretação
    -------------
    - Curva ABAIXO da diagonal → modelo confiante demais (overconfident)
      Comum em Random Forest (comprime probabilidades para 0.3–0.7)
    - Curva ACIMA da diagonal → modelo conservador (underconfident)
    - ECE próximo de 0 → bem calibrado
    - Brier Score < 0.15 → bom para risco CV
    """
    y_prob = modelo_bruto.predict_proba(X_val)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=10)
    brier = brier_score_loss(y_val, y_prob)
    ece   = _calcular_ece(y_val, y_prob)

    print(f"\n{'='*50}")
    print(f"DIAGNÓSTICO DE CALIBRAÇÃO — {nome_modelo}")
    print(f"{'='*50}")
    print(f"  Brier Score : {brier:.4f}  (0 = perfeito | 0.25 = aleatório)")
    print(f"  ECE         : {ece:.4f}  (0 = calibração perfeita)")

    if ece > 0.05:
        print("  ⚠️  ECE > 0.05 → calibração recomendada")
    else:
        print("  ✓  ECE ≤ 0.05 → modelo já bem calibrado")

    return {"brier": brier, "ece": ece, "frac_pos": frac_pos, "mean_pred": mean_pred}


# ==============================================================================
# CALIBRAÇÃO
# ==============================================================================

def calibrar(modelo_base, X_train, y_train,
             method: str = None) -> CalibratedClassifierCV:
    """
    Aplica calibração probabilística ao modelo base.

    Parâmetros
    ----------
    method : "sigmoid" (Platt Scaling) ou "isotonic" (Regressão Isotônica)
        - "sigmoid"  → recomendado para datasets < 5000 amostras
        - "isotonic" → mais flexível, risco de overfitting em datasets pequenos

    O cv=N_CV_FOLDS garante que os dados de treino não vazam para
    o ajuste da calibração (evita overfit na calibração).
    """
    method = method or CALIBRATION_METHOD

    modelo_calibrado = CalibratedClassifierCV(
        estimator=modelo_base,
        method=method,
        cv=N_CV_FOLDS
    )
    modelo_calibrado.fit(X_train, y_train)

    print(f"\n✓ Calibração aplicada: {method.upper()} "
          f"({'Platt Scaling' if method == 'sigmoid' else 'Isotonic Regression'})")
    return modelo_calibrado


# ==============================================================================
# COMPARAÇÃO DE MÉTODOS
# ==============================================================================

def comparar_metodos(modelo_base, X_train, y_train, X_val, y_val) -> dict:
    """
    Treina e compara 3 variantes de calibração lado a lado.

    Retorna o nome do melhor método (menor ECE) e todos os modelos calibrados.
    Use os resultados para escolher qual versão salvar.
    """
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE MÉTODOS DE CALIBRAÇÃO")
    print("=" * 60)
    print(f"{'Método':<30} {'Brier Score':>12} {'ECE':>10}")
    print("-" * 54)

    resultados = {}

    # Sem calibração (baseline)
    y_prob_raw = modelo_base.predict_proba(X_val)[:, 1]
    brier_raw  = brier_score_loss(y_val, y_prob_raw)
    ece_raw    = _calcular_ece(y_val, y_prob_raw)
    print(f"{'Sem calibração (baseline)':<30} {brier_raw:>12.4f} {ece_raw:>10.4f}")
    resultados["sem_calibracao"] = {
        "modelo": modelo_base, "brier": brier_raw, "ece": ece_raw
    }

    # Platt Scaling
    modelo_sigmoid = calibrar(modelo_base, X_train, y_train, method="sigmoid")
    y_prob_sig     = modelo_sigmoid.predict_proba(X_val)[:, 1]
    brier_sig      = brier_score_loss(y_val, y_prob_sig)
    ece_sig        = _calcular_ece(y_val, y_prob_sig)
    print(f"{'Platt Scaling (sigmoid)':<30} {brier_sig:>12.4f} {ece_sig:>10.4f}")
    resultados["platt_scaling"] = {
        "modelo": modelo_sigmoid, "brier": brier_sig, "ece": ece_sig
    }

    # Isotonic Regression
    modelo_iso  = calibrar(modelo_base, X_train, y_train, method="isotonic")
    y_prob_iso  = modelo_iso.predict_proba(X_val)[:, 1]
    brier_iso   = brier_score_loss(y_val, y_prob_iso)
    ece_iso     = _calcular_ece(y_val, y_prob_iso)
    print(f"{'Isotonic Regression':<30} {brier_iso:>12.4f} {ece_iso:>10.4f}")
    resultados["isotonic"] = {
        "modelo": modelo_iso, "brier": brier_iso, "ece": ece_iso
    }

    # Escolhe o melhor por ECE
    melhor = min(
        ["platt_scaling", "isotonic"],
        key=lambda k: resultados[k]["ece"]
    )
    print(f"\n✓ Melhor método: {melhor} (ECE = {resultados[melhor]['ece']:.4f})")

    resultados["melhor"] = melhor
    return resultados


def plotar_comparacao(modelo_base, modelos_calibrados: dict,
                      X_val, y_val, salvar_em: str = None):
    """
    Plota Reliability Diagrams lado a lado para comparação visual.

    Um ponto próximo da diagonal = probabilidade confiável.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Comparação de Métodos de Calibração\n"
                 "Reliability Diagram — quanto mais próximo da diagonal, melhor",
                 fontsize=12, fontweight="bold")

    configs = [
        ("sem_calibracao", "Sem Calibração (baseline)", "red"),
        ("platt_scaling",  "Platt Scaling (sigmoid)",   "blue"),
        ("isotonic",       "Isotonic Regression",       "green"),
    ]

    for ax, (chave, titulo, cor) in zip(axes, configs):
        modelo = modelos_calibrados[chave]["modelo"]
        y_prob = modelo.predict_proba(X_val)[:, 1]

        frac, mean = calibration_curve(y_val, y_prob, n_bins=10)
        brier = modelos_calibrados[chave]["brier"]
        ece   = modelos_calibrados[chave]["ece"]

        ax.plot(mean, frac, f"s-", color=cor, lw=2,
                label=f"Modelo\nBrier={brier:.4f}\nECE={ece:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfeita")

        ax.fill_between(mean, frac, mean,
                        alpha=0.15, color=cor, label="Desvio")
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Probabilidade Predita Média")
        ax.set_ylabel("Fração de Positivos Reais")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"✓ Gráfico de calibração salvo: {salvar_em}")
    plt.close()


# ==============================================================================
# HELPERS INTERNOS
# ==============================================================================

def _calcular_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Mede o quanto as probabilidades preditas desviam das frequências reais.
    ECE = 0 significa calibração perfeita.

    Referência: Naeini et al. (2015, AAAI)
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n       = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)

    return ece

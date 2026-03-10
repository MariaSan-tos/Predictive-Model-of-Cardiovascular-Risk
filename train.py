"""
train.py — Pipeline de treinamento completo.

Orquestra todos os módulos na ordem correta:
  dados → modelo → calibração → threshold → avaliação → salvar

Uso:
    python train.py --data cardiovascular_risk_dataset.csv
    python train.py --data cardiovascular_risk_dataset.csv --calibration-only
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Módulos do projeto
from data.preprocessing import pipeline_dados
from calibration.calibrator import diagnosticar_calibracao, comparar_metodos, plotar_comparacao
from calibration.threshold  import otimizar_youden, comparar_thresholds, plotar_curva_threshold
from evaluation.metrics     import calcular_metricas, imprimir_metricas, calcular_ic_cv, exportar_metricas
from evaluation.plots       import painel_validacao
from utils.io               import salvar_modelo
from config                 import RANDOM_STATE, TEST_SIZE, OUTPUT_DIR, CALIBRATION_METHOD

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# CONSTRUÇÃO DO MODELO BASE
# ==============================================================================

def construir_modelo_base(X_train, y_train):
    """
    Ensemble de três estimadores com soft voting.

    Justificativa da escolha:
    - Random Forest: robusto a outliers, não-linear (Breiman, 2001)
    - Gradient Boosting: melhor performance em dados tabulares (Friedman, 2001)
    - Logistic Regression: interpretável, âncora linear do ensemble
    - Soft voting: usa probabilidades → aproveita calibração posterior
    - Class weights: corrige desbalanceamento (Japkowicz & Stephen, 2002)
    """
    classes = np.unique(y_train)
    pesos   = compute_class_weight("balanced", classes=classes, y=y_train)
    cw      = dict(zip(classes, pesos))

    print(f"\n✓ Pesos de classe: {cw}")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        max_features="sqrt", class_weight=cw,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=10,
        random_state=RANDOM_STATE
    )
    lr = LogisticRegression(
        C=0.1, class_weight=cw, max_iter=5000, solver='saga', random_state=RANDOM_STATE
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[3, 3, 1]
    )

    print("⏳ Treinando modelo base...")
    ensemble.fit(X_train, y_train)
    print("✓ Modelo base treinado.")
    return ensemble


# ==============================================================================
# EXTRAÇÃO DE IMPORTÂNCIAS
# ==============================================================================

def extrair_importancias(modelo_calibrado, feature_names: list):
    """Tenta extrair importâncias do RF dentro do ensemble calibrado."""
    try:
        rf = modelo_calibrado.calibrated_classifiers_[0].estimator.estimators_[0][1]
        return rf.feature_importances_
    except Exception:
        return None


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run(caminho_csv: str, calibration_only: bool = False):

    # 1. Dados
    print("\n" + "=" * 60)
    print("ETAPA 1 — DADOS")
    print("=" * 60)
    X, y = pipeline_dados(caminho_csv)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✓ Treino: {len(X_train)} | Teste: {len(X_test)}")

    # 2. Modelo base
    if not calibration_only:
        print("\n" + "=" * 60)
        print("ETAPA 2 — MODELO BASE")
        print("=" * 60)
        modelo_base = construir_modelo_base(X_train, y_train)
    else:
        from utils.io import carregar_modelo
        payload    = carregar_modelo(f"{OUTPUT_DIR}/modelo_cardiovascular_base.pkl")
        modelo_base = payload["modelo"]

    # 3. Diagnóstico de calibração (ANTES de calibrar)
    print("\n" + "=" * 60)
    print("ETAPA 3 — DIAGNÓSTICO DE CALIBRAÇÃO")
    print("=" * 60)
    diagnosticar_calibracao(modelo_base, X_test, y_test, "Ensemble base")

    # 4. Comparar métodos de calibração e escolher o melhor
    print("\n" + "=" * 60)
    print("ETAPA 4 — CALIBRAÇÃO")
    print("=" * 60)
    resultados_cal = comparar_metodos(modelo_base, X_train, y_train, X_test, y_test)
    plotar_comparacao(
        modelo_base, resultados_cal, X_test, y_test,
        salvar_em=f"{OUTPUT_DIR}/calibracao_comparacao.png"
    )

    melhor_metodo  = resultados_cal["melhor"]
    modelo_final   = resultados_cal[melhor_metodo]["modelo"]
    print(f"\n✓ Modelo selecionado: {melhor_metodo}")

    # 5. Threshold
    print("\n" + "=" * 60)
    print("ETAPA 5 — THRESHOLD")
    print("=" * 60)
    y_prob    = modelo_final.predict_proba(X_test)[:, 1]
    threshold = otimizar_youden(y_test, y_prob)

    comparar_thresholds(y_test, y_prob)
    plotar_curva_threshold(
        y_test, y_prob, threshold,
        salvar_em=f"{OUTPUT_DIR}/threshold_analise.png"
    )

    # 6. Avaliação final
    print("\n" + "=" * 60)
    print("ETAPA 6 — AVALIAÇÃO FINAL")
    print("=" * 60)
    y_pred   = (y_prob >= threshold).astype(int)
    metricas = calcular_metricas(y_test, y_prob, y_pred, threshold)
    imprimir_metricas(metricas)

    cv_resultado = calcular_ic_cv(modelo_final, X_test, y_test)
    exportar_metricas(metricas, cv_resultado, f"{OUTPUT_DIR}/metricas.csv")

    # 7. Visualizações de validação
    print("\n" + "=" * 60)
    print("ETAPA 7 — GRÁFICOS DE VALIDAÇÃO")
    print("=" * 60)
    importancias = extrair_importancias(modelo_final, feature_names)
    painel_validacao(
        y_test, y_prob, y_pred,
        feature_names=feature_names,
        importancias=importancias,
        salvar_em=f"{OUTPUT_DIR}/validacao_modelo.png"
    )

    # 8. Salvar
    print("\n" + "=" * 60)
    print("ETAPA 8 — SALVANDO MODELO")
    print("=" * 60)
    salvar_modelo(modelo_final, feature_names, threshold, metricas)

    print("\n" + "=" * 60)
    print("✅ PIPELINE CONCLUÍDO")
    print(f"   ROC-AUC : {metricas['roc_auc']:.4f}")
    print(f"   Sensib. : {metricas['sensibilidade']:.4f}")
    print(f"   Espec.  : {metricas['especificidade']:.4f}")
    print(f"   Threshold: {threshold:.4f}")
    print("=" * 60)

    return modelo_final, feature_names, threshold


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo de risco cardiovascular")
    parser.add_argument("--data", default=None,
                        help="Caminho para o CSV (omitir = baixar do Kaggle automaticamente)")
    parser.add_argument("--calibration-only", action="store_true",
                        help="Recalibra modelo já treinado sem re-treinar")
    args = parser.parse_args()
    run(args.data, calibration_only=args.calibration_only)

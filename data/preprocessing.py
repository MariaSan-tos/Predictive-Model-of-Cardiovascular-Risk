"""
data/preprocessing.py — Limpeza, encoding e engenharia de features.

Toda transformação de dados fica aqui.
Nenhum outro módulo deve importar pandas diretamente para transformar dados.
"""

import os
import pandas as pd
import numpy as np
from config import (
    TARGET_COL, ID_COL, FEATURE_COLS, CATEGORICAL_MAPPINGS
)


# ==============================================================================
# CARREGAMENTO
# ==============================================================================

def carregar_dados(caminho: str = None) -> pd.DataFrame:
    """
    Carrega o dataset. Se caminho=None, baixa automaticamente do Kaggle.
    """
    if caminho is None:
        import kagglehub
        print("⏳ Baixando dataset do Kaggle...")
        path = kagglehub.dataset_download(
            "bertnardomariouskono/cardiovascular-disease-risk-prediction-dataset"
        )
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        caminho = os.path.join(path, files[0])
        print(f"✓ Dataset baixado: {caminho}")

    df = pd.read_csv(caminho)
    df.columns = df.columns.str.strip()
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    
    print("=" * 60)
    print("INSPEÇÃO DO DATASET")
    print("=" * 60)
    print(f"  Shape          : {df.shape[0]} pacientes × {df.shape[1]} variáveis")

    ausentes = df.isnull().sum()
    ausentes = ausentes[ausentes > 0]
    if ausentes.empty:
        print("  Valores ausentes: nenhum ✓")
    else:
        print(f"  Valores ausentes:\n{ausentes}")

    counts = df[TARGET_COL].value_counts()
    print(f"  Alvo (0/1)     : {counts.to_dict()}  —  "
          f"{counts[1]/len(df)*100:.1f}% positivos")

    return df


# ==============================================================================
# ENGENHARIA DE FEATURES (embasamento científico)
# ==============================================================================

def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva features com embasamento clínico.

    Referências
    -----------
    - NonHDL-C: ACC/AHA Guidelines (Stone et al., 2013, JACC)
    - LDL/HDL ratio (Castelli Risk Index II): Castelli et al., 1983, Am J Med
    - Total/HDL ratio (Castelli Risk Index I): idem
    - Pulse Pressure: Franklin et al., 1999, Circulation
    - Mean Arterial Pressure: padrão hemodinâmico clínico
    - Metabolic Risk Score: ATP III Criteria (Expert Panel, JAMA 2001)
    - Lifestyle Burden: composto baseado em fatores de risco modificáveis
    - Age × SBP interaction: não-linearidade documentada no Framingham Study
    """
    df = df.copy()

    # --- Lipídicos ---
    df["NonHDL_Cholesterol"] = df["Cholesterol_Total"] - df["Cholesterol_HDL"]
    df["LDL_HDL_Ratio"]      = df["Cholesterol_LDL"]   / (df["Cholesterol_HDL"] + 1e-6)
    df["Total_HDL_Ratio"]    = df["Cholesterol_Total"]  / (df["Cholesterol_HDL"] + 1e-6)

    # --- Hemodinâmicos ---
    df["Pulse_Pressure"]        = df["Systolic_BP"] - df["Diastolic_BP"]
    df["Mean_Arterial_Pressure"]= df["Diastolic_BP"] + df["Pulse_Pressure"] / 3

    # Estágio de hipertensão (JNC 8 / ESC 2018)
    df["Hypertension_Stage"] = 0
    df.loc[df["Systolic_BP"] >= 130, "Hypertension_Stage"] = 1
    df.loc[df["Systolic_BP"] >= 140, "Hypertension_Stage"] = 2
    df.loc[df["Systolic_BP"] >= 160, "Hypertension_Stage"] = 3

    # --- Compostos ---
    smoke_current = (df["Smoking_Status"] == 2).astype(int)

    df["Metabolic_Risk_Score"] = (
        (df["BMI"] >= 30).astype(int) +
        (df["Systolic_BP"] >= 130).astype(int) +
        (df["Fasting_Blood_Sugar"] >= 100).astype(int) +
        (df["Cholesterol_HDL"] < 40).astype(int)
    )

    # Lifestyle Burden: fumo ativo tem peso 3 (maior risco relativo)
    phys_low = (df["Physical_Activity_Level"] == 0).astype(int)
    df["Lifestyle_Burden"] = (
        smoke_current * 3 +
        phys_low      * 2 +
        (df["Sleep_Hours"] < 6).astype(int) +
        (df["Stress_Level"] > 7).astype(int)
    )

    # --- Interações ---
    df["Age_SBP_Interaction"]          = df["Age"] * df["Systolic_BP"] / 1000
    df["Smoking_Cholesterol_Interaction"] = smoke_current * df["Cholesterol_LDL"]

    n_novas = df.shape[1] - (len(FEATURE_COLS) + 1)  # +1 = target
    print(f"\n✓ Engenharia de features: {n_novas} variáveis derivadas criadas")
    return df


# ==============================================================================
# PRÉ-PROCESSAMENTO FINAL
# ==============================================================================

def preprocessar(df: pd.DataFrame):
    """
    Encoding de categóricas e separação X / y.

    Retorna
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = df.copy()

    if ID_COL in df.columns:
        df.drop(ID_COL, axis=1, inplace=True)

    # Encoding com mapeamento explícito (rastreável, sem LabelEncoder automático)
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(mapping)

    # Recalcula features compostas que dependem dos valores encoded
    df["Lifestyle_Burden"] = (
        (df["Smoking_Status"] == 2).astype(int) * 3 +
        (df["Physical_Activity_Level"] == 0).astype(int) * 2 +
        (df["Sleep_Hours"] < 6).astype(int) +
        (df["Stress_Level"] > 7).astype(int)
    )
    df["Smoking_Cholesterol_Interaction"] = (
        (df["Smoking_Status"] == 2).astype(int) * df["Cholesterol_LDL"]
    )

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    print(f"✓ Pré-processamento: {X.shape[1]} features  |  {y.sum()} positivos de {len(y)}")
    return X, y


# PIPELINE COMPLETO (conveniência)
def pipeline_dados(caminho: str = None):   # ← só adiciona = None aqui
    """Executa carregamento → features → encoding em uma chamada."""
    df = carregar_dados(caminho)
    df = engenharia_features(df)
    X, y = preprocessar(df)
    return X, y


def encode_paciente(paciente: dict, feature_names: list) -> pd.DataFrame:
    """
    Prepara dicionário de um paciente individual para predição.
    Aplica as mesmas transformações do pipeline de treino.
    """
    df = pd.DataFrame([paciente])
    df = engenharia_features(df)

    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df["Lifestyle_Burden"] = (
        (df["Smoking_Status"] == 2).astype(int) * 3 +
        (df["Physical_Activity_Level"] == 0).astype(int) * 2 +
        (df["Sleep_Hours"] < 6).astype(int) +
        (df["Stress_Level"] > 7).astype(int)
    )
    df["Smoking_Cholesterol_Interaction"] = (
        (df["Smoking_Status"] == 2).astype(int) * df["Cholesterol_LDL"]
    )

    return df[feature_names]

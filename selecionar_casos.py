"""
selecionar_casos.py — Seleciona pacientes reais do dataset para uso no artigo.

Critério: busca casos com perfil clínico extremo (claramente alto risco
e claramente baixo risco) para ilustrar o comportamento do modelo.

Uso:
    python selecionar_casos.py --model outputs/modelo_cardiovascular.pkl
                               --data caminho_do_csv (opcional, baixa do Kaggle)
"""

import argparse
import os
import pandas as pd
import numpy as np

from data.preprocessing import carregar_dados, engenharia_features, encode_paciente
from utils.io import carregar_modelo


# ==============================================================================
# SELEÇÃO DE CASOS
# ==============================================================================

def selecionar_casos(df_original: pd.DataFrame, payload: dict):
    """
    Seleciona automaticamente:
    - 2 casos de ALTO RISCO real (Heart_Disease_Risk=1) com perfil mais extremo
    - 2 casos de BAIXO RISCO real (Heart_Disease_Risk=0) com perfil mais protetor
    """
    modelo        = payload["modelo"]
    feature_names = payload["feature_names"]
    threshold     = payload["threshold"]

    # Prepara dataset completo para predição
    df = df_original.copy()
    if 'Patient_ID' in df.columns:
        df = df.drop('Patient_ID', axis=1)

    df_features = engenharia_features(df.drop('Heart_Disease_Risk', axis=1)
                                      .assign(Heart_Disease_Risk=df['Heart_Disease_Risk']))

    from data.preprocessing import preprocessar
    X, y = preprocessar(df_features)
    X = X[feature_names]

    # Probabilidades para todos os pacientes
    probs = modelo.predict_proba(X)[:, 1]
    df_original = df_original.copy()
    df_original['prob_predita'] = probs
    df_original['pred_label']   = (probs >= threshold).astype(int)

    # --- Casos de ALTO RISCO: positivos reais com maior probabilidade predita ---
    alto_risco = (
        df_original[df_original['Heart_Disease_Risk'] == 1]
        .nlargest(5, 'prob_predita')
        .head(2)
    )

    # --- Casos de BAIXO RISCO: negativos reais com menor probabilidade predita ---
    baixo_risco = (
        df_original[df_original['Heart_Disease_Risk'] == 0]
        .nsmallest(5, 'prob_predita')
        .head(2)
    )

    casos = pd.concat([alto_risco, baixo_risco]).reset_index(drop=True)
    casos['grupo'] = ['Alto Risco', 'Alto Risco', 'Baixo Risco', 'Baixo Risco']

    return casos


# ==============================================================================
# PREDIÇÃO E RELATÓRIO POR CASO
# ==============================================================================

def relatorio_caso(paciente_row: pd.Series, payload: dict, grupo: str):
    """Gera relatório completo para um paciente individual."""
    from predict import calcular_risco

    # Converte para dicionário com os campos originais
    campos_originais = [
        'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI',
        'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total',
        'Cholesterol_LDL', 'Cholesterol_HDL', 'Fasting_Blood_Sugar',
        'Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level',
        'Family_History', 'Stress_Level', 'Sleep_Hours'
    ]

    # Decode categóricas de volta para string (para encode_paciente)
    paciente = paciente_row[campos_originais].to_dict()

    # Reverse mapping para apresentação
    gender_map   = {1: 'Male', 0: 'Female'}
    smoking_map  = {0: 'Never', 1: 'Former', 2: 'Current'}
    activity_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
    fh_map       = {0: 'No', 1: 'Yes'}

    paciente_str = paciente.copy()
    if isinstance(paciente['Gender'], (int, float)):
        paciente_str['Gender']                 = gender_map.get(int(paciente['Gender']), paciente['Gender'])
        paciente_str['Smoking_Status']         = smoking_map.get(int(paciente['Smoking_Status']), paciente['Smoking_Status'])
        paciente_str['Physical_Activity_Level']= activity_map.get(int(paciente['Physical_Activity_Level']), paciente['Physical_Activity_Level'])
        paciente_str['Family_History']         = fh_map.get(int(paciente['Family_History']), paciente['Family_History'])

    print("\n" + "=" * 60)
    print(f"CASO CLÍNICO — {grupo}")
    print(f"Diagnóstico Real: {'Doença Cardiovascular' if paciente_row['Heart_Disease_Risk'] == 1 else 'Sem Doença Cardiovascular'}")
    print("=" * 60)
    print(f"  Idade          : {int(paciente_str['Age'])} anos")
    print(f"  Sexo           : {paciente_str['Gender']}")
    print(f"  IMC            : {paciente_str['BMI']:.1f} kg/m²")
    print(f"  PA Sistólica   : {int(paciente_str['Systolic_BP'])} mmHg")
    print(f"  PA Diastólica  : {int(paciente_str['Diastolic_BP'])} mmHg")
    print(f"  Col. Total     : {int(paciente_str['Cholesterol_Total'])} mg/dL")
    print(f"  Col. LDL       : {int(paciente_str['Cholesterol_LDL'])} mg/dL")
    print(f"  Col. HDL       : {int(paciente_str['Cholesterol_HDL'])} mg/dL")
    print(f"  Glicemia Jejum : {int(paciente_str['Fasting_Blood_Sugar'])} mg/dL")
    print(f"  Tabagismo      : {paciente_str['Smoking_Status']}")
    print(f"  Atividade Fís. : {paciente_str['Physical_Activity_Level']}")
    print(f"  Hist. Familiar : {paciente_str['Family_History']}")
    print(f"  Estresse       : {int(paciente_str['Stress_Level'])}/10")
    print(f"  Sono           : {paciente_str['Sleep_Hours']:.0f}h/noite")
    print(f"  Álcool         : {int(paciente_str['Alcohol_Consumption'])} doses/semana")

    resultado = calcular_risco(paciente_str, payload)
    return paciente_str, resultado


# ==============================================================================
# EXPORTAR TABELA PARA ARTIGO
# ==============================================================================

def exportar_tabela(casos_lista: list, caminho: str = "outputs/casos_artigo.csv"):
    """Exporta tabela formatada dos casos para uso no artigo."""
    rows = []
    for paciente, resultado, grupo, real in casos_lista:
        rows.append({
            'Grupo':           grupo,
            'Diagnóstico Real': 'Positivo' if real == 1 else 'Negativo',
            'Idade':           int(paciente['Age']),
            'Sexo':            paciente['Gender'],
            'IMC':             paciente['BMI'],
            'PA Sistólica':    int(paciente['Systolic_BP']),
            'LDL':             int(paciente['Cholesterol_LDL']),
            'HDL':             int(paciente['Cholesterol_HDL']),
            'Glicemia':        int(paciente['Fasting_Blood_Sugar']),
            'Tabagismo':       paciente['Smoking_Status'],
            'Prob. Predita':   resultado['porcentagem'],
            'Status Predito':  resultado['status'],
            'Categoria':       resultado['categoria'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(caminho, index=False)
    print(f"\n✓ Tabela exportada: {caminho}")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Caminho para o modelo .pkl")
    parser.add_argument("--data", default=None,
                        help="Caminho para o CSV (omitir = baixar do Kaggle)")
    args = parser.parse_args()

    # Carrega modelo e dados
    payload    = carregar_modelo(args.model)
    df_original = carregar_dados(args.data)

    # Seleciona casos
    print("\n⏳ Selecionando casos representativos do dataset...")
    casos_df = selecionar_casos(df_original, payload)

    print(f"\n✓ {len(casos_df)} casos selecionados")
    print(casos_df[['grupo', 'Heart_Disease_Risk', 'Age', 'Gender',
                     'Smoking_Status', 'BMI', 'Systolic_BP', 'prob_predita']].to_string())

    # Gera relatório individual por caso
    casos_lista = []
    for _, row in casos_df.iterrows():
        paciente, resultado = relatorio_caso(row, payload, row['grupo'])
        casos_lista.append((paciente, resultado, row['grupo'], row['Heart_Disease_Risk']))

    # Exporta tabela para o artigo
    tabela = exportar_tabela(casos_lista)
    print("\n" + "=" * 60)
    print("TABELA PARA O ARTIGO")
    print("=" * 60)
    print(tabela.to_string(index=False))

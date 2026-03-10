"""
predict.py — Calculadora clínica de risco cardiovascular individual.

Uso:
    python predict.py --model outputs/modelo_cardiovascular.pkl
    python predict.py --model outputs/modelo_cardiovascular.pkl --paciente paciente.json
"""

import argparse
import json
from data.preprocessing import encode_paciente
from utils.io import carregar_modelo
from config import RISK_TIERS


# ==============================================================================
# PREDIÇÃO INDIVIDUAL
# ==============================================================================

def calcular_risco(paciente: dict, payload: dict) -> dict:
    """
    Predição de risco para um paciente individual.

    Estratificação por limiares ACC/AHA 2019 (Grundy et al., JACC 2019).
    """
    modelo        = payload["modelo"]
    feature_names = payload["feature_names"]
    threshold     = payload["threshold"]

    X = encode_paciente(paciente, feature_names)
    probabilidade = float(modelo.predict_proba(X)[0, 1])
    status        = "ALTO RISCO" if probabilidade >= threshold else "BAIXO RISCO"

    # Estratificação clínica
    for limite, categoria, recomendacao in RISK_TIERS:
        if probabilidade < limite:
            break

    resultado = {
        "status":        status,
        "probabilidade": probabilidade,
        "porcentagem":   f"{probabilidade * 100:.1f}%",
        "categoria":     categoria,
        "recomendacao":  recomendacao,
        "threshold":     threshold,
    }

    _imprimir_resultado(resultado)
    return resultado


def _imprimir_resultado(r: dict):
    print("\n" + "=" * 55)
    print("  AVALIAÇÃO DE RISCO CARDIOVASCULAR")
    print("=" * 55)
    print(f"  Status         : {r['status']}")
    print(f"  Probabilidade  : {r['porcentagem']}")
    print(f"  Categoria      : {r['categoria']}")
    print(f"  Recomendação   : {r['recomendacao']}")
    print(f"  Threshold usado: {r['threshold']:.4f} (Youden ponderado)")
    print("=" * 55)
    print("  ⚠️  Este modelo é suporte à decisão clínica.")
    print("     Não substitui avaliação médica profissional.")
    print("=" * 55)


# ==============================================================================
# PACIENTE DE EXEMPLO
# ==============================================================================

PACIENTE_EXEMPLO = {
    "Age": 65, "Gender": "Male", "Height_cm": 175, "Weight_kg": 95,
    "BMI": 31.0, "Systolic_BP": 150, "Diastolic_BP": 95,
    "Cholesterol_Total": 240, "Cholesterol_LDL": 160, "Cholesterol_HDL": 35,
    "Fasting_Blood_Sugar": 130, "Smoking_Status": "Current",
    "Alcohol_Consumption": 5, "Physical_Activity_Level": "Low",
    "Family_History": "Yes", "Stress_Level": 8, "Sleep_Hours": 5,
}


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculadora de risco cardiovascular")
    parser.add_argument("--model", required=True,
                        help="Caminho para o arquivo .pkl do modelo")
    parser.add_argument("--paciente", default=None,
                        help="JSON com dados do paciente (opcional, usa exemplo se omitido)")
    args = parser.parse_args()

    payload = carregar_modelo(args.model)

    if args.paciente:
        with open(args.paciente) as f:
            paciente = json.load(f)
    else:
        print("\n(Nenhum paciente fornecido — usando exemplo de alto risco)")
        paciente = PACIENTE_EXEMPLO

    calcular_risco(paciente, payload)

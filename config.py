"""
config.py — Constantes e configurações globais do projeto.

Centralizar aqui evita "magic numbers" espalhados pelo código.
"""

# Reprodutibilidade
RANDOM_STATE = 42

# Colunas do dataset
TARGET_COL = "Heart_Disease_Risk"
ID_COL     = "Patient_ID"

FEATURE_COLS = [
    "Age", "Gender", "Height_cm", "Weight_kg", "BMI",
    "Systolic_BP", "Diastolic_BP",
    "Cholesterol_Total", "Cholesterol_LDL", "Cholesterol_HDL",
    "Fasting_Blood_Sugar", "Smoking_Status", "Alcohol_Consumption",
    "Physical_Activity_Level", "Family_History", "Stress_Level", "Sleep_Hours",
]

# Mapeamentos categóricos (explícitos para rastreabilidade)
CATEGORICAL_MAPPINGS = {
    "Gender":                  {"Male": 1, "Female": 0},
    "Smoking_Status":          {"Never": 0, "Former": 1, "Current": 2},
    "Physical_Activity_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Family_History":          {"No": 0, "Yes": 1},
}

# Divisão treino/teste
TEST_SIZE = 0.20

# Modelo
N_CV_FOLDS = 5

# Calibração
# "sigmoid" = Platt Scaling | "isotonic" = Regressão Isotônica
CALIBRATION_METHOD = "sigmoid"

# Threshold
# Peso para o Youden ponderado: w > 1 penaliza mais falsos negativos
# Recomendado para triagem CV: 1.5 – 2.0
YOUDEN_FN_WEIGHT = 2.0

# Estratificação de risco (ACC/AHA 2019)
RISK_TIERS = [
    (0.075, "🟢 Baixo Risco (<7.5%)",
     "Manter estilo de vida saudável. Reavaliação em 5 anos."),
    (0.200, "🟡 Risco Intermediário (7.5–20%)",
     "Avaliar fatores potenciadores. Consulta médica anual."),
    (0.400, "🟠 Alto Risco (20–40%)",
     "Intervenção nos fatores modificáveis. Avaliação cardiológica."),
    (1.001, "🔴 Muito Alto Risco (>40%)",
     "Referência urgente para cardiologia. Terapia intensiva de fatores de risco."),
]

# Paths de saída
OUTPUT_DIR   = "outputs"
MODEL_FILE   = "outputs/modelo_cardiovascular.pkl"
PLOTS_FILE   = "outputs/validacao_modelo.png"
METRICS_FILE = "outputs/metricas.csv"

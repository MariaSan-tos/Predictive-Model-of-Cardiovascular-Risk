# 🫀 Predictive Model of Cardiovascular Risk

Modelo preditivo de risco cardiovascular baseado em ensemble de Machine Learning com calibração probabilística e threshold clínico otimizado pelo Índice de Youden.

> Desenvolvido com embasamento científico nas diretrizes **ACC/AHA 2019**, **ESC 2021**, **Framingham Heart Study** e validado seguindo o checklist **TRIPOD** (Collins et al., 2015, BMJ).

---

## 📊 Resultados do Modelo

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.79 |
| Sensibilidade | ~0.75 (com Youden ponderado) |
| Especificidade | ~0.72 |
| Brier Score | < 0.18 |
| Threshold | Otimizado (Youden, w=2.0) |

---

## 🗂️ Estrutura do Projeto

```
cardio_risk/
│
├── config.py                    # Constantes globais (thresholds, paths, tiers de risco)
│
├── data/
│   └── preprocessing.py         # Limpeza, encoding e engenharia de features
│
├── calibration/
│   ├── calibrator.py            # Diagnóstico + Platt Scaling + Isotonic Regression
│   └── threshold.py             # Índice de Youden clássico e ponderado
│
├── evaluation/
│   ├── metrics.py               # ROC-AUC, Brier Score, VPP, VPN, IC 95%
│   └── plots.py                 # Painel de validação, Reliability Diagram, ROC
│
├── utils/
│   └── io.py                    # Salvar e carregar modelo (.pkl)
│
├── train.py                     # Pipeline de treinamento completo (8 etapas)
├── predict.py                   # Calculadora clínica individual
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

### 1. Pré-requisitos

- Python **3.9 ou superior**
- pip

Verifique sua versão:
```bash
python --version
```

### 2. Clone o repositório

```bash
git clone https://github.com/MariaSan-tos/Predictive-Model-of-Cardiovascular-Risk.git
cd Predictive-Model-of-Cardiovascular-Risk
```

### 3. Crie um ambiente virtual (recomendado)

```bash
# Criar
python -m venv venv

# Ativar — Windows
venv\Scripts\activate

# Ativar — macOS / Linux
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar

### Treinar o modelo do zero

O arquivo CSV é baixado automaticamente, mas pode encontrar e baixar manualmente em: 
https://www.kaggle.com/datasets/bertnardomariouskono/cardiovascular-disease-risk-prediction-dataset


```bash
# Sem baixar nada manualmente
python train.py

# Com arquivo baixado manualmente.
python train.py --data cardiovascular_risk_dataset.csv
```

O pipeline executa automaticamente as 8 etapas:

```
ETAPA 1 — DADOS              → carregamento + engenharia de features
ETAPA 2 — MODELO BASE        → ensemble (Random Forest + Gradient Boosting + LR)
ETAPA 3 — DIAGNÓSTICO        → verifica se calibração é necessária (ECE, Brier)
ETAPA 4 — CALIBRAÇÃO         → compara Platt Scaling vs Isotonic, escolhe o melhor
ETAPA 5 — THRESHOLD          → Índice de Youden ponderado (w=2.0)
ETAPA 6 — AVALIAÇÃO FINAL    → métricas completas + IC 95% via validação cruzada
ETAPA 7 — GRÁFICOS           → painel de validação salvo em outputs/
ETAPA 8 — SALVAR             → modelo salvo em outputs/modelo_cardiovascular.pkl
```

Ao final você verá no terminal:

```
✅ PIPELINE CONCLUÍDO
   ROC-AUC : 0.8100
   Sensib. : 0.7532
   Espec.  : 0.7245
   Threshold: 0.3841
```

---

### Recalibrar sem re-treinar

Se quiser apenas ajustar a calibração de um modelo já treinado:

```bash
python train.py --data cardiovascular_risk_dataset.csv --calibration-only
```

---

### Predição individual (calculadora clínica)

É possível utilizar a Predição de 3 formas:

**Usando o paciente de exemplo embutido:**
```bash
python predict.py --model outputs/modelo_cardiovascular.pkl
```

**Usando 4 casos reais, disponíveis no próprio Dataset:**
```bash
python selecionar_casos.py --model outputs/modelo_cardiovascular.pkl
```

**Usando um paciente em JSON criado por você:**
Crie um arquivo `paciente.json`:
```json
{
    "Age": 58,
    "Gender": "Female",
    "Height_cm": 162,
    "Weight_kg": 78,
    "BMI": 29.7,
    "Systolic_BP": 138,
    "Diastolic_BP": 88,
    "Cholesterol_Total": 215,
    "Cholesterol_LDL": 140,
    "Cholesterol_HDL": 42,
    "Fasting_Blood_Sugar": 105,
    "Smoking_Status": "Former",
    "Alcohol_Consumption": 2,
    "Physical_Activity_Level": "Moderate",
    "Family_History": "Yes",
    "Stress_Level": 6,
    "Sleep_Hours": 7
}
```

```bash
python predict.py --model outputs/modelo_cardiovascular.pkl --paciente paciente.json
```

**Saída esperada para o Paciente exemplo (Paciente de alto risco):**
```
=======================================================
  AVALIAÇÃO DE RISCO CARDIOVASCULAR
=======================================================
  Status         : ALTO RISCO
  Probabilidade  : 100.0%
  Categoria      : 🔴 Muito Alto Risco (>40%)
  Recomendação   : Referência urgente para cardiologia. Terapia intensiva de fatores de risco.
  Threshold usado: 0.3047 (Youden ponderado)
=======================================================
  ⚠️  Este modelo é suporte à decisão clínica.
     Não substitui avaliação médica profissional.
=======================================================
```

---

## 📁 Arquivos Gerados

Após o treinamento, a pasta `outputs/` conterá:

| Arquivo | Descrição |
|---|---|
| `modelo_cardiovascular.pkl` | Modelo serializado + metadados |
| `validacao_modelo.png` | Painel completo de validação (6 gráficos) |
| `calibracao_comparacao.png` | Reliability Diagrams: sem calibração vs Platt vs Isotonic |
| `threshold_analise.png` | Sensibilidade vs Especificidade por threshold |
| `metricas.csv` | Todas as métricas exportadas para registro |

---

## 🧬 Features do Modelo

### Variáveis originais do dataset
`Age`, `Gender`, `Height_cm`, `Weight_kg`, `BMI`, `Systolic_BP`, `Diastolic_BP`, `Cholesterol_Total`, `Cholesterol_LDL`, `Cholesterol_HDL`, `Fasting_Blood_Sugar`, `Smoking_Status`, `Alcohol_Consumption`, `Physical_Activity_Level`, `Family_History`, `Stress_Level`, `Sleep_Hours`

### Features derivadas (engenharia científica)

| Feature | Base Científica |
|---|---|
| `NonHDL_Cholesterol` | ACC/AHA Guidelines (Stone et al., 2013) |
| `LDL_HDL_Ratio` | Castelli Risk Index II (1983) |
| `Total_HDL_Ratio` | Castelli Risk Index I (1983) |
| `Pulse_Pressure` | Franklin et al., 1999, Circulation |
| `Mean_Arterial_Pressure` | Padrão hemodinâmico clínico |
| `Hypertension_Stage` | JNC 8 / ESC 2018 Guidelines |
| `Metabolic_Risk_Score` | ATP III Criteria (JAMA, 2001) |
| `Lifestyle_Burden` | Composto de fatores modificáveis |
| `Age_SBP_Interaction` | Framingham Heart Study |
| `Smoking_Cholesterol_Interaction` | Sinergismo documentado na literatura |

---

## 🔬 Decisões Técnicas

**Por que ensemble e não um único modelo?**
Random Forest captura não-linearidades mas tende a comprimir probabilidades para 0.3–0.7. Gradient Boosting tem melhor discriminação em dados tabulares. Logistic Regression fornece âncora linear interpretável. O soft voting combina os três usando probabilidades calibradas.

**Por que calibrar as probabilidades?**
Modelos de ensemble sem calibração produzem probabilidades não confiáveis — quando o modelo diz "80% de risco", a probabilidade real pode ser 60% ou 90%. Em contexto clínico, probabilidades precisas são essenciais para decisão. A calibração é verificada via ECE (Expected Calibration Error) e Brier Score antes e depois.

**Por que threshold ≠ 0.50?**
O threshold padrão assume custo igual para falso positivo e falso negativo. Em triagem cardiovascular, não detectar um paciente em risco é clinicamente muito mais grave que um alarme desnecessário. O Índice de Youden ponderado (w=2.0) penaliza 2× mais os falsos negativos, aumentando a sensibilidade do modelo.

---

## 📚 Referências

- Wilson PWF et al. (1998). Prediction of Coronary Heart Disease. *Circulation*
- Grundy SM et al. (2019). ACC/AHA Guideline on the Management of Blood Cholesterol. *JACC*
- Mach F et al. (2020). 2019 ESC/EAS Guidelines for the management of dyslipidaemias. *EHJ*
- Collins GS et al. (2015). Transparent reporting of a multivariable prediction model (TRIPOD). *BMJ*
- Niculescu-Mizil A & Caruana R (2005). Predicting good probabilities with supervised learning. *ICML*
- Youden WJ (1950). Index for rating diagnostic tests. *Cancer* 3(1):32-35
- Breiman L (2001). Random forests. *Machine Learning* 45:5-32
- Friedman JH (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*

---

## ⚠️ Aviso

Este modelo foi desenvolvido para fins acadêmicos e de pesquisa com dados do Kaggle. **Não deve ser utilizado para diagnóstico clínico real** sem validação prospectiva em população-alvo e aprovação por comitê de ética. Sempre consulte um profissional de saúde qualificado.

---

## 📄 Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.

# 🧬 Clinical Trial Analyser

An end-to-end data science project simulating a Phase II clinical trial analysis pipeline — built to demonstrate healthcare analytics skills for life sciences consulting roles.

---

## 📌 Project Overview

This project analyses a simulated clinical trial dataset to:
- Explore patient demographics and treatment patterns (EDA)
- Predict treatment response using ML models (XGBoost + Logistic Regression)
- Explain predictions using SHAP (SHapley Additive exPlanations)
- Surface KPIs and insights via an interactive Streamlit dashboard

---

## 🗂️ Project Structure

```
clinical-trial-analyser/
│
├── data/
│   ├── generate_data.py        ← Synthetic dataset generator
│   └── trial_data.csv          ← Generated dataset (500 patients)
│
├── models/
│   ├── train_model.py          ← Model training (XGBoost + Logistic Regression)
│   ├── xgb_model.pkl           ← Saved XGBoost model
│   ├── lr_model.pkl            ← Saved Logistic Regression model
│   ├── results.pkl             ← Evaluation metrics
│   └── shap_data.pkl           ← SHAP values for explainability
│
├── app/
│   └── dashboard.py            ← Streamlit dashboard (5 pages)
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
cd data
python generate_data.py
```

### 3. Train the models
```bash
cd models
python train_model.py
```

### 4. Launch the dashboard
```bash
cd app
streamlit run dashboard.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPI cards, dataset preview, trial summary |
| 🔬 EDA | Response rates, age distribution, dosage trends, correlation heatmap |
| 🤖 Model Performance | Accuracy, AUC, confusion matrix, model comparison |
| 🧠 SHAP Explainability | Feature importance, SHAP value heatmap |
| 🔮 Predict Patient | Enter patient profile → get response probability |

---

## 🤖 Models Used

| Model | Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | 58% | 0.653 |
| Logistic Regression | 65% | 0.707 |

Logistic Regression outperforms XGBoost on this clean, smaller tabular dataset — consistent with literature on linear models for structured clinical data.

---

## 🔬 Key Findings

- **Treatment group** is the strongest predictor of response
- **Higher dosage** correlates with improved response rates
- **Younger patients** with fewer comorbidities respond better
- **Adverse events** slightly reduce response probability

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Scikit-Learn, XGBoost, SHAP
- **Visualisation** — Matplotlib, Seaborn
- **Dashboard** — Streamlit
- **Version Control** — Git/GitHub

---

## 📝 Resume Bullet

> *Built an end-to-end clinical trial analytics pipeline on simulated Phase II patient data; applied XGBoost and Logistic Regression (AUC: 0.71) with SHAP explainability to predict treatment response, and deployed a 5-page Streamlit dashboard surfacing efficacy KPIs for pharma decision support.*

---

*Built as a portfolio project targeting life sciences analytics roles.*

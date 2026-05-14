import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../data/trial_data.csv')
MODELS = os.path.join(BASE, '../models')

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clinical Trial Analyser", page_icon="🧬", layout="wide")

# ── load assets ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_models():
    with open(os.path.join(MODELS, 'xgb_model.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODELS, 'lr_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(MODELS, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(MODELS, 'shap_data.pkl'), 'rb') as f:
        shap_data = pickle.load(f)
    return xgb_model, lr_model, results, shap_data

df = load_data()
xgb_model, lr_model, results, shap_data = load_models()

# ── sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
st.sidebar.title("Clinical Trial\nAnalyser")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔬 EDA", "🤖 Model Performance", "🧠 SHAP Explainability", "🔮 Predict Patient"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🧬 Clinical Trial Analyser")
    st.markdown("**Simulated Phase II Trial — Treatment Outcome Prediction Dashboard**")
    st.markdown("---")

    total     = len(df)
    responders = df['outcome'].sum()
    response_rate = round(responders / total * 100, 1)
    drug_patients = len(df[df['treatment_group'] == 'Drug'])
    avg_age   = round(df['age'].mean(), 1)
    ae_rate   = round(df['adverse_events'].mean() * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",    total)
    c2.metric("Responders",        responders)
    c3.metric("Response Rate",     f"{response_rate}%")
    c4.metric("Drug Arm Patients", drug_patients)
    c5.metric("Adverse Event Rate",f"{ae_rate}%")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Trial Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Attribute | Value |
|---|---|
| Trial Phase | Phase II (Simulated) |
| Total Enrolled | 500 patients |
| Treatment Arms | Drug vs Placebo |
| Primary Endpoint | Treatment Response (Binary) |
| Duration Range | 4 – 24 weeks |
        """)
    with col2:
        st.markdown("""
| Attribute | Value |
|---|---|
| Age Range | 25 – 74 years |
| Dosage Levels | 50, 100, 150, 200 mg |
| Adverse Event Rate | 25% |
| Best Model (AUC) | Logistic Regression — 0.71 |
| XGBoost AUC | 0.65 |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Response Rate by Treatment Group")
        grp = df.groupby('treatment_group')['outcome'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#2196F3' if t == 'Drug' else '#90CAF9' for t in grp['treatment_group']]
        ax.bar(grp['treatment_group'], grp['outcome'] * 100, color=colors)
        ax.set_ylabel("Response Rate (%)")
        ax.set_ylim(0, 80)
        for i, v in enumerate(grp['outcome'] * 100):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(5, 3))
        df[df['outcome']==1]['age'].hist(ax=ax, alpha=0.7, bins=20, color='#4CAF50', label='Responded')
        df[df['outcome']==0]['age'].hist(ax=ax, alpha=0.7, bins=20, color='#F44336', label='No Response')
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Response Rate by Dosage")
        drug_df = df[df['treatment_group'] == 'Drug']
        dose_grp = drug_df.groupby('dosage_mg')['outcome'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(dose_grp['dosage_mg'], dose_grp['outcome']*100, marker='o', color='#9C27B0', linewidth=2)
        ax.set_xlabel("Dosage (mg)")
        ax.set_ylabel("Response Rate (%)")
        ax.set_ylim(0, 80)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Adverse Events vs Outcome")
        ae_grp = df.groupby(['adverse_events','outcome']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3))
        ae_grp.plot(kind='bar', ax=ax, color=['#F44336','#4CAF50'], edgecolor='white')
        ax.set_xticklabels(['No AE','Adverse Event'], rotation=0)
        ax.set_ylabel("Patient Count")
        ax.legend(['No Response','Responded'])
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.subheader("Correlation Heatmap")
    num_cols = ['age','dosage_mg','duration_weeks','adverse_events','comorbidities','outcome']
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost Accuracy",  f"{results['xgb_accuracy']*100:.1f}%")
    col2.metric("XGBoost AUC",       f"{results['xgb_auc']:.3f}")
    col3.metric("Logistic Reg. Acc.",f"{results['lr_accuracy']*100:.1f}%")
    col4.metric("Logistic Reg. AUC", f"{results['lr_auc']:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Comparison")
        fig, ax = plt.subplots(figsize=(5, 3))
        models  = ['XGBoost','Logistic Reg.']
        acc     = [results['xgb_accuracy'], results['lr_accuracy']]
        auc     = [results['xgb_auc'],      results['lr_auc']]
        x = np.arange(len(models))
        ax.bar(x - 0.2, acc, 0.35, label='Accuracy', color='#2196F3')
        ax.bar(x + 0.2, auc, 0.35, label='AUC-ROC',  color='#4CAF50')
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylim(0, 1); ax.legend()
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("XGBoost Confusion Matrix")
        cm = np.array(results['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted 0','Predicted 1'],
                    yticklabels=['Actual 0','Actual 1'])
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📌 Interpretation")
    st.info("""
- **Logistic Regression** outperforms XGBoost on this dataset — expected for clean, smaller tabular data with linear relationships.
- **AUC of 0.71** means the model correctly ranks a responder above a non-responder 71% of the time — solid for a Phase II simulation.
- Key drivers of response: **treatment group, dosage, age, and comorbidities**.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Feature Importance")
    st.markdown("SHAP (SHapley Additive exPlanations) explains *which features* drive each prediction.")
    st.markdown("---")

    sv      = shap_data['shap_values']
    X_test  = shap_data['X_test']
    features = results['features']
    feature_labels = ['Age','Gender','Treatment Group','Dosage (mg)','Duration (weeks)','Adverse Events','Comorbidities']

    # Mean absolute SHAP
    mean_shap = np.abs(sv).mean(axis=0)
    shap_df   = pd.DataFrame({'Feature': feature_labels, 'Mean |SHAP|': mean_shap}).sort_values('Mean |SHAP|', ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(shap_df['Feature'], shap_df['Mean |SHAP|'], color='#7B1FA2')
        ax.set_xlabel("Mean |SHAP value|")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("What This Means")
        st.markdown("""
| Feature | Impact |
|---|---|
| **Treatment Group** | Strongest driver — Drug arm significantly boosts response probability |
| **Dosage** | Higher dosage → higher response likelihood |
| **Age** | Younger patients respond better |
| **Comorbidities** | More conditions → lower response rate |
| **Adverse Events** | Presence slightly reduces response probability |
| **Duration** | Longer treatment mildly improves outcomes |
| **Gender** | Minimal impact in this trial |
        """)

    st.markdown("---")
    st.subheader("SHAP Values — First 50 Test Patients")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap_sample = sv[:50]
    im = ax.imshow(shap_sample.T, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel("Patient Index")
    plt.colorbar(im, ax=ax, label='SHAP value')
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Patient":
    st.title("🔮 Predict Treatment Outcome")
    st.markdown("Enter a patient profile to predict their likelihood of responding to treatment.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age             = st.slider("Age", 25, 74, 45)
        gender          = st.selectbox("Gender", ["Male", "Female"])
        treatment_group = st.selectbox("Treatment Group", ["Drug", "Placebo"])
        dosage          = st.selectbox("Dosage (mg)", [0, 50, 100, 150, 200]) if treatment_group == "Drug" else 0
    with col2:
        duration        = st.slider("Duration (weeks)", 4, 24, 12)
        adverse_events  = st.selectbox("Adverse Events", [0, 1], format_func=lambda x: "Yes" if x else "No")
        comorbidities   = st.selectbox("Comorbidities", [0, 1, 2], format_func=lambda x: f"{x} condition(s)")

    if st.button("🔬 Predict Outcome", use_container_width=True):
        gender_enc    = 1 if gender == "Male" else 0
        treatment_enc = 1 if treatment_group == "Placebo" else 0
        input_data = pd.DataFrame([[age, gender_enc, treatment_enc, dosage, duration, adverse_events, comorbidities]],
                                   columns=results['features'])

        xgb_prob = xgb_model.predict_proba(input_data)[0][1]
        lr_prob  = lr_model.predict_proba(input_data)[0][1]
        avg_prob = (xgb_prob + lr_prob) / 2

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost Probability",     f"{xgb_prob*100:.1f}%")
        c2.metric("Logistic Reg. Probability",f"{lr_prob*100:.1f}%")
        c3.metric("Ensemble Probability",     f"{avg_prob*100:.1f}%")

        st.markdown("---")
        if avg_prob >= 0.5:
            st.success(f"✅ **Likely Responder** — Ensemble probability: {avg_prob*100:.1f}%")
        else:
            st.error(f"❌ **Unlikely to Respond** — Ensemble probability: {avg_prob*100:.1f}%")

        # Risk gauge
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(0, 100, color='#EEEEEE', height=0.4)
        color = '#4CAF50' if avg_prob >= 0.5 else '#F44336'
        ax.barh(0, avg_prob*100, color=color, height=0.4)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_yticks([]); ax.set_xlabel("Response Probability (%)")
        ax.set_title("Response Probability Gauge")
        ax.spines[['top','right','left']].set_visible(False)
        st.pyplot(fig)
        plt.close()

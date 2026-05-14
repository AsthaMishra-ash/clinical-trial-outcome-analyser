import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import shap
import pickle
import os

def train_and_save():
    # Load data
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, '../data/trial_data.csv'))

    # Encode categorical
    le = LabelEncoder()
    df['gender_enc'] = le.fit_transform(df['gender'])
    df['treatment_enc'] = le.fit_transform(df['treatment_group'])

    features = ['age', 'gender_enc', 'treatment_enc', 'dosage_mg', 'duration_weeks', 'adverse_events', 'comorbidities']
    X = df[features]
    y = df['outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Evaluate
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lr_preds = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    results = {
        'xgb_accuracy': round(accuracy_score(y_test, xgb_preds), 4),
        'xgb_auc': round(roc_auc_score(y_test, xgb_proba), 4),
        'lr_accuracy': round(accuracy_score(y_test, lr_preds), 4),
        'lr_auc': round(roc_auc_score(y_test, lr_proba), 4),
        'confusion_matrix': confusion_matrix(y_test, xgb_preds).tolist(),
        'features': features
    }

    # SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # Save everything
    model_dir = os.path.join(base)
    with open(os.path.join(model_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(model_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(model_dir, 'shap_data.pkl'), 'wb') as f:
        pickle.dump({'shap_values': shap_values, 'X_test': X_test}, f)

    print("✅ Models trained and saved.")
    print(f"XGBoost — Accuracy: {results['xgb_accuracy']}, AUC: {results['xgb_auc']}")
    print(f"Logistic Regression — Accuracy: {results['lr_accuracy']}, AUC: {results['lr_auc']}")
    return results

if __name__ == '__main__':
    train_and_save()

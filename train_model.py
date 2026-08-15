import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('viridis')

DATA_DIR = os.getcwd()
DEPLOYMENT_DIR = os.path.join(DATA_DIR, "preprocessing_objects")
MODEL_DIR = os.path.join(DATA_DIR, "models")
REPORTS_DIR = os.path.join(DATA_DIR, "Model_Reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"{'='*70}\nPHASE 3: CATBOOST MODEL TRAINING & COMPREHENSIVE EVALUATION\n{'='*70}")

print("\n[1/6] Loading preprocessed datasets...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Loaded X_test : {X_test.shape}, y_test : {y_test.shape}")

metadata_path = os.path.join(DEPLOYMENT_DIR, "preprocessing_metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        prep_metadata = json.load(f)
    categorical_cols = prep_metadata.get('categorical_features', [])
    cat_feature_indices = prep_metadata.get('cat_feature_indices', [])
    target_mapping = prep_metadata.get('target_mapping', {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3})
else:
    categorical_cols = []
    cat_feature_indices = []
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}

class_names = [k for k, v in sorted(target_mapping.items(), key=lambda item: item[1])]

print("\n[2/6] Preparing internal validation split (80/20)...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)
print(f"Internal Train shape: {X_tr.shape}, Internal Validation shape: {X_val.shape}")

print("\n[3/6] Fitting CatBoostClassifier with MultiClass loss...")
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.08,
    depth=6,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)

print(f"\nTraining completed! Best iteration: {model.get_best_iteration()}")

print("\n[4/6] Evaluating CatBoost Model on Test Set (40,000 samples)...")
y_pred = model.predict(X_test).ravel()
y_prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
prec_weighted = precision_score(y_test, y_pred, average='weighted')
rec_macro = recall_score(y_test, y_pred, average='macro')
rec_weighted = recall_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
log_loss_val = log_loss(y_test, y_prob)
roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

print("\n" + "="*50)
print("             CATBOOST TEST METRICS SUMMARY")
print("="*50)
print(f" Accuracy                 : {acc * 100:.2f}%")
print(f" Macro Precision          : {prec_macro * 100:.2f}%")
print(f" Macro Recall             : {rec_macro * 100:.2f}%")
print(f" Macro F1-Score           : {f1_macro * 100:.2f}%")
print(f" Weighted F1-Score        : {f1_weighted * 100:.2f}%")
print(f" Multi-Class Log Loss     : {log_loss_val:.4f}")
print(f" ROC-AUC (OVR Macro)      : {roc_auc_ovr:.4f}")
print("="*50)

print("\nPer-Class Detailed Classification Report:")
clf_report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n[5/6] Generating evaluation visual dashboards in 'Model_Reports/'...")

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax[0])
ax[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight='bold')
ax[0].set_xlabel("Predicted Label", fontsize=12)
ax[0].set_ylabel("True Label", fontsize=12)

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=ax[1])
ax[1].set_title("Normalized Confusion Matrix (%)", fontsize=14, fontweight='bold')
ax[1].set_xlabel("Predicted Label", fontsize=12)
ax[1].set_ylabel("True Label", fontsize=12)

plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "01_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

feature_importances = model.get_feature_importance()
feature_names = X_train.columns

df_fi = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
df_fi = df_fi.sort_values(by='importance', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 8))
sns.barplot(data=df_fi.head(20), x='importance', y='feature', palette='mako')
plt.title("Top 20 CatBoost Feature Importances (PredictionValuesChange)", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score (%)", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.tight_layout()
fi_path = os.path.join(REPORTS_DIR, "02_feature_importance.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"Saved: {fi_path}")

plt.figure(figsize=(10, 7))
y_test_bin = pd.get_dummies(y_test).values
for i, col in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {col} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("CatBoost One-vs-Rest ROC Curves by Risk Category", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, "03_roc_curves.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved: {roc_path}")

df_per_class = pd.DataFrame({
    'Risk Category': class_names,
    'Precision': [clf_report_dict[c]['precision'] for c in class_names],
    'Recall': [clf_report_dict[c]['recall'] for c in class_names],
    'F1-Score': [clf_report_dict[c]['f1-score'] for c in class_names]
})

df_melted = df_per_class.melt(id_vars='Risk Category', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Risk Category', y='Score', hue='Metric', palette='Set2')
plt.ylim([0, 1.05])
plt.title("CatBoost Performance Metrics Across Risk Categories", fontsize=14, fontweight='bold')
plt.xlabel("Risk Category", fontsize=12)
plt.ylabel("Score", fontsize=12)
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height / 2.),
                           ha='center', va='center', fontsize=9, color='white', fontweight='bold')
plt.tight_layout()
metrics_path = os.path.join(REPORTS_DIR, "04_metrics_per_class.png")
plt.savefig(metrics_path, dpi=300)
plt.close()
print(f"Saved: {metrics_path}")

print("\n[6/6] Exporting CatBoost model artifacts & metadata...")

cbm_model_path = os.path.join(MODEL_DIR, "catboost_risk_model.cbm")
model.save_model(cbm_model_path)
print(f"Saved CatBoost model (cbm): '{cbm_model_path}'")

try:
    import joblib
    joblib_model_path = os.path.join(MODEL_DIR, "catboost_risk_model.joblib")
    joblib.dump(model, joblib_model_path)
    print(f"Saved CatBoost model (joblib): '{joblib_model_path}'")
except Exception as e:
    print(f"Note: Joblib model export notice: {e}")

model_metadata = {
    "model_type": "CatBoostClassifier",
    "target_classes": class_names,
    "target_mapping": target_mapping,
    "best_iteration": model.get_best_iteration(),
    "test_metrics": {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "multi_class_log_loss": float(log_loss_val),
        "roc_auc_ovr_macro": float(roc_auc_ovr)
    },
    "per_class_metrics": {
        c: {
            "precision": float(clf_report_dict[c]['precision']),
            "recall": float(clf_report_dict[c]['recall']),
            "f1_score": float(clf_report_dict[c]['f1-score']),
            "support": int(clf_report_dict[c]['support'])
        } for c in class_names
    },
    "top_10_features": df_fi.head(10).to_dict(orient='records')
}

meta_json_path = os.path.join(MODEL_DIR, "model_metadata.json")
with open(meta_json_path, 'w') as f:
    json.dump(model_metadata, f, indent=4)
print(f"Saved model metadata JSON: '{meta_json_path}'")

print(f"\n{'='*70}\nPHASE 3 COMPLETE: CATBOOST MODEL READY FOR DEPLOYMENT!\n{'='*70}")

import os
import sys
import json
import numpy as np
import pandas as pd

DEPLOYMENT_DIR = "preprocessing_objects"
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
print(f"Deployment objects directory ready: '{DEPLOYMENT_DIR}/'")


DATASET_PATH = "project_risk_dataset.csv"
print(f"\n{'='*60}\nSTEP 1: LOADING DATASET\n{'='*60}")
df = pd.read_csv(DATASET_PATH)
print(f"Loaded raw dataset from '{DATASET_PATH}'. Initial Shape: {df.shape}")


print(f"\n{'='*60}\nSTEP 2 & 3: FEATURE SEPARATION & LEAKAGE REMOVAL\n{'='*60}")

unnecessary_cols = ['project_id', 'risk_score']
target_column = 'risk_category'

print(f"Dropping identifier column: 'project_id'")
print(f"Dropping target-leakage column: 'risk_score' (prevents artificial 100% target leakage)")

X_raw = df.drop(columns=unnecessary_cols + [target_column])
y_raw = df[target_column]

print(f"Raw Features (X) Shape: {X_raw.shape}")
print(f"Target (y) Shape       : {y_raw.shape}")


print(f"\n{'='*60}\nSTEP 4: FEATURE TAXONOMY IDENTIFICATION\n{'='*60}")
categorical_cols = list(X_raw.select_dtypes(include=['object', 'category', 'string']).columns)
numerical_cols = list(X_raw.select_dtypes(include=['int64', 'float64']).columns)

print(f"Number of Categorical Features ({len(categorical_cols)}): {categorical_cols}")
print(f"Number of Numerical Features   ({len(numerical_cols)}): {numerical_cols}")


print(f"\n{'='*60}\nSTEP 5: MISSING VALUE AUDIT & HANDLING\n{'='*60}")
missing_count = X_raw.isnull().sum().sum()
print(f"Total Missing Values in Feature Matrix X: {missing_count}")
if missing_count == 0:
    print("No missing values found. Dataset is 100% complete.")
else:
    
    for col in numerical_cols:
        if X_raw[col].isnull().sum() > 0:
            median_val = X_raw[col].median()
            X_raw[col].fillna(median_val, inplace=True)
    for col in categorical_cols:
        if X_raw[col].isnull().sum() > 0:
            mode_val = X_raw[col].mode()[0]
            X_raw[col].fillna(mode_val, inplace=True)


print(f"\n{'='*60}\nSTEP 6: DUPLICATE RECORDS CHECK & REMOVAL\n{'='*60}")
duplicate_count = X_raw.duplicated().sum()
print(f"Duplicate Feature Rows: {duplicate_count}")
if duplicate_count > 0:
    dedup_mask = ~X_raw.duplicated()
    X_raw = X_raw[dedup_mask]
    y_raw = y_raw[dedup_mask]
    print(f"Removed {duplicate_count} duplicates. Updated X shape: {X_raw.shape}")
else:
    print("Zero duplicate records found.")


print(f"\n{'='*60}\nSTEP 7: OUTLIER HANDLING VIA WINSORIZATION (1st & 99th PERCENTILES)\n{'='*60}")


capper_bounds = {}
X_capped = X_raw.copy()

for col in numerical_cols:
    p01 = X_raw[col].quantile(0.01)
    p99 = X_raw[col].quantile(0.99)
    capper_bounds[col] = {'lower_bound': float(p01), 'upper_bound': float(p99)}
    X_capped[col] = np.clip(X_raw[col], p01, p99)

print(f"Successfully applied quantile capping to {len(numerical_cols)} numerical features.")


print(f"\n{'='*60}\nSTEP 8: CATEGORICAL & TARGET ENCODING\n{'='*60}")


target_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
inv_target_mapping = {v: k for k, v in target_mapping.items()}
y_encoded = y_raw.map(target_mapping)

print(f"Target Label Mapping: {target_mapping}")
print("Encoded Target Distribution:")
print(y_encoded.value_counts(normalize=True).rename(index=inv_target_mapping) * 100)


categorical_mappings = {}
X_encoded = X_capped.copy()

for col in categorical_cols:
    unique_vals = sorted(X_capped[col].astype(str).unique())
    col_map = {val: idx for idx, val in enumerate(unique_vals)}
    categorical_mappings[col] = col_map
    X_encoded[col] = X_capped[col].astype(str).map(col_map)

cat_feature_indices = [X_encoded.columns.get_loc(col) for col in categorical_cols]


print(f"\n{'='*60}\nSTEP 9: FEATURE SCALING ANALYSIS (CATBOOST REQUIREMENT)\n{'='*60}")
print("Explanation on Feature Scaling:")
print("- CatBoost is a Gradient Boosted Decision Tree (GBDT) algorithm.")
print("- Decision trees evaluate feature splits using threshold inequalities (e.g. Feature_X > 25.5).")
print("- Monotonic transformations or linear feature scaling (StandardScaler / MinMaxScaler)")
print("  do NOT alter the rank order or split points of tree nodes.")
print("- Therefore, Feature Scaling is UNNECESSARY for CatBoost model training.")
print("- For deployment completeness, fitted scaling parameters (mean/std) are saved in deployment objects.")

scaler_params = {}
for col in numerical_cols:
    q25 = float(X_encoded[col].quantile(0.25))
    q75 = float(X_encoded[col].quantile(0.75))
    iqr = q75 - q25 if (q75 - q25) != 0 else 1.0
    median = float(X_encoded[col].median())
    scaler_params[col] = {'median': median, 'iqr': iqr}


print(f"\n{'='*60}\nSTEP 10: STRATIFIED TRAIN-TEST SPLIT (80% / 20%)\n{'='*60}")

np.random.seed(42)
test_size = 0.20

train_indices = []
test_indices = []

for class_val in sorted(y_encoded.unique()):
    class_idx = np.where(y_encoded.values == class_val)[0]
    np.random.shuffle(class_idx)
    n_test = int(len(class_idx) * test_size)
    test_indices.extend(class_idx[:n_test])
    train_indices.extend(class_idx[n_test:])

np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

X_train = X_encoded.iloc[train_indices].reset_index(drop=True)
X_test = X_encoded.iloc[test_indices].reset_index(drop=True)
y_train = y_encoded.iloc[train_indices].reset_index(drop=True)
y_test = y_encoded.iloc[test_indices].reset_index(drop=True)


print(f"\n{'='*60}\nSTEP 11: PREPROCESSING METRICS & SHAPES\n{'='*60}")
print(f"Training Set Shape (X_train) : {X_train.shape}")
print(f"Testing Set Shape  (X_test)  : {X_test.shape}")
print(f"Training Labels    (y_train) : {y_train.shape}")
print(f"Testing Labels     (y_test)  : {y_test.shape}")
print(f"Number of Numerical Features : {len(numerical_cols)}")
print(f"Number of Categorical Features: {len(categorical_cols)}")
print(f"Total Feature Count           : {X_train.shape[1]}")

print("\nClass Distribution Check (Train vs Test Ratio):")
train_dist = y_train.value_counts(normalize=True) * 100
test_dist = y_test.value_counts(normalize=True) * 100
split_check = pd.DataFrame({
    'Class': [inv_target_mapping[i] for i in sorted(target_mapping.values())],
    'Train Count': y_train.value_counts().sort_index().values,
    'Train Pct (%)': train_dist.sort_index().values.round(2),
    'Test Count': y_test.value_counts().sort_index().values,
    'Test Pct (%)': test_dist.sort_index().values.round(2)
})
print(split_check.to_string(index=False))


print(f"\n{'='*60}\nSTEP 12: SAVING PROCESSED DATASETS\n{'='*60}")
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_frame(name="risk_category").to_csv("y_train.csv", index=False)
y_test.to_frame(name="risk_category").to_csv("y_test.csv", index=False)

print("Saved files successfully:")
print(" - X_train.csv")
print(" - X_test.csv")
print(" - y_train.csv")
print(" - y_test.csv")


print(f"\n{'='*60}\nSTEP 13: SAVING DEPLOYMENT PREPROCESSING OBJECTS\n{'='*60}")

metadata = {
    'target_column': target_column,
    'target_mapping': target_mapping,
    'inverse_target_mapping': {str(k): v for k, v in inv_target_mapping.items()},
    'categorical_columns': categorical_cols,
    'numerical_columns': numerical_cols,
    'cat_feature_indices': cat_feature_indices,
    'dropped_columns': unnecessary_cols,
    'capper_bounds': capper_bounds,
    'categorical_mappings': categorical_mappings,
    'scaler_params': scaler_params,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'catboost_ready': True
}

metadata_path = os.path.join(DEPLOYMENT_DIR, "preprocessing_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Deployment metadata saved to: '{metadata_path}'")


try:
    import joblib
    joblib.dump(target_mapping, os.path.join(DEPLOYMENT_DIR, "target_encoder.joblib"))
    joblib.dump(categorical_mappings, os.path.join(DEPLOYMENT_DIR, "categorical_encoders.joblib"))
    joblib.dump(capper_bounds, os.path.join(DEPLOYMENT_DIR, "outlier_capper.joblib"))
    joblib.dump(scaler_params, os.path.join(DEPLOYMENT_DIR, "scaler.joblib"))
    print("Joblib deployment objects saved successfully in 'preprocessing_objects/'.")
except Exception as e:
    print(f"Note: joblib save notice ({e}); JSON metadata successfully covers all parameters.")


print(f"\n{'='*60}\nDATA PREPROCESSING COMPLETE & CATBOOST TRAINING READY!\n{'='*60}")

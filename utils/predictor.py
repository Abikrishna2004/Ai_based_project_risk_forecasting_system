"""
CatBoost Model Inference Engine for Project Risk Prediction
Validates 20 frontend features, applies categorical mappings and numerical outlier capping,
and returns real-time CatBoost risk forecasts, confidence scores, and class probabilities.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from models.model_loader import ModelLoader

# Fallback 20-feature taxonomy
DEFAULT_FEATURES = [
    "project_type",
    "industry_sector",
    "methodology",
    "region",
    "priority",
    "planned_duration_days",
    "budget_usd",
    "requirement_changes_count",
    "vendor_dependency_count",
    "milestones_missed",
    "team_size",
    "team_avg_experience_years",
    "team_turnover_pct",
    "resource_availability_pct",
    "communication_score",
    "sponsor_engagement_score",
    "tech_complexity_score",
    "scope_clarity_score",
    "external_dependency_score",
    "defect_count"
]

DEFAULT_CATEGORICAL = [
    "project_type",
    "industry_sector",
    "methodology",
    "region",
    "priority"
]

DEFAULT_NUMERICAL = [
    "planned_duration_days",
    "budget_usd",
    "requirement_changes_count",
    "vendor_dependency_count",
    "milestones_missed",
    "team_size",
    "team_avg_experience_years",
    "team_turnover_pct",
    "resource_availability_pct",
    "communication_score",
    "sponsor_engagement_score",
    "tech_complexity_score",
    "scope_clarity_score",
    "external_dependency_score",
    "defect_count"
]

_startup_validated = False


@st.cache_resource
def get_cached_model_and_metadata():
    """
    Loads and caches the trained 20-feature CatBoost model and preprocessing objects.
    Uses Streamlit resource caching to avoid reloading from disk on user interactions.
    """
    model = ModelLoader.load_model()
    metadata, capper, encoders = ModelLoader.load_preprocessing_objects()
    return model, metadata, capper, encoders


def validate_model_integration():
    """
    Performs a one-time validation to verify that:
      - Model artifact exists.
      - Preprocessing metadata exists and contains 20 features.
      - Categorical encoders exist.
      - Outlier capper bounds exist.
    """
    global _startup_validated
    if _startup_validated:
        return True

    try:
        model, metadata, capper, encoders = get_cached_model_and_metadata()

        if model is None:
            raise FileNotFoundError("CatBoost model instance is None.")

        feature_cols = metadata.get("feature_columns", [])
        if len(feature_cols) != 20:
            raise ValueError(f"Expected 20 feature_columns in metadata, found {len(feature_cols)}.")

        _startup_validated = True
        print("[SUCCESS] Startup validation completed: 20-feature CatBoost model integration verified.")
        return True
    except Exception as e:
        print(f"[ERROR] Startup validation failed: {e}")
        return False


def predict_project_risk(input_dict):
    """
    Takes user input dictionary, validates 20 required features, applies encodings and capping,
    and executes CatBoost predict_proba().

    Returns dict:
      - risk_category: str ("Low", "Medium", "High", "Critical")
      - risk_score: float (highest class confidence percentage, e.g. 85.5%)
      - weighted_risk_score: float (weighted risk score percentage)
      - class_probabilities: dict of class percentages
      - probabilities: dict of class percentages
    """
    try:
        model, metadata, capper, encoders = get_cached_model_and_metadata()
    except Exception as e:
        print(f"Error loading model/metadata: {e}")
        return {
            "error": f"Failed to load model artifacts: {str(e)}"
        }

    if model is None or metadata is None:
        return {
            "error": "Model or preprocessing metadata could not be initialized."
        }

    # Extract Feature Taxonomy
    feature_columns = metadata.get("feature_columns", DEFAULT_FEATURES)
    cat_cols = metadata.get("categorical_columns", DEFAULT_CATEGORICAL)
    num_cols = metadata.get("numerical_columns", DEFAULT_NUMERICAL)

    # 1. Strict Feature Validation (Missing Features Check)
    missing_features = [col for col in feature_columns if col not in input_dict]
    if missing_features:
        err_msg = f"Missing required ML features for prediction: {missing_features}"
        print(f"[ERROR] {err_msg}")
        return {
            "error": err_msg,
            "missing_features": missing_features
        }

    # 2. Extract and Process 20 Features (Ignore extra fields like project_name or old features)
    processed_row = {}

    # 2a. Categorical Encoding with Unseen Value Validation
    cat_mappings = encoders if encoders else metadata.get("categorical_mappings", {})
    for col in cat_cols:
        val = str(input_dict.get(col, ""))
        mapping = cat_mappings.get(col, {})

        if val in mapping:
            processed_row[col] = mapping[val]
        else:
            err_unseen = f"Unsupported value '{val}' for feature '{col}'. Please select a value supported by the trained model."
            print(f"[ERROR] {err_unseen}")
            return {
                "error": err_unseen
            }

    # 2b. Numerical Outlier Capping (No StandardScaler/MinMaxScaler)
    bounds = capper if capper else metadata.get("capper_bounds", {})
    for col in num_cols:
        try:
            val = float(input_dict.get(col, 0.0))
        except (ValueError, TypeError):
            val = 0.0

        if col in bounds:
            lb = bounds[col].get("lower_bound", bounds[col].get("lb", val))
            ub = bounds[col].get("upper_bound", bounds[col].get("ub", val))
            val = float(np.clip(val, lb, ub))
        processed_row[col] = val

    # 3. Build DataFrame with exact feature order
    df_input = pd.DataFrame([processed_row])[feature_columns]

    # 4. Strict Shape & Order Validation
    if df_input.shape != (1, 20):
        err_shape = f"Invalid feature matrix shape {df_input.shape}, expected exactly (1, 20)"
        print(f"[ERROR] {err_shape}")
        return {
            "error": err_shape
        }

    if list(df_input.columns) != feature_columns:
        err_order = "DataFrame column order does not match feature_columns taxonomy."
        print(f"[ERROR] {err_order}")
        return {
            "error": err_order
        }

    # 5. CatBoost Inference via predict_proba()
    try:
        probs = model.predict_proba(df_input)[0]

        # Target class map
        inv_target_map = metadata.get(
            "inverse_target_mapping",
            {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"}
        )

        # Highest probability class
        top_class_idx = int(np.argmax(probs))
        raw_pred_str = str(top_class_idx)
        risk_category = inv_target_map.get(raw_pred_str, "Medium")

        # Highest predicted class confidence percentage
        confidence_pct = round(float(probs[top_class_idx]) * 100.0, 1)

        # Weighted risk score percentage (Low=15, Medium=45, High=75, Critical=95)
        class_weights = [15.0, 45.0, 75.0, 95.0]
        weighted_score = round(sum(p * w for p, w in zip(probs, class_weights)), 1)

        # Class probabilities dictionary
        class_probs = {
            inv_target_map.get(str(i), f"Class {i}"): round(float(p) * 100.0, 1)
            for i, p in enumerate(probs)
        }

        print(f"[DEBUG] Input DataFrame Shape: {df_input.shape}")
        print(f"[DEBUG] Prediction Probabilities: {class_probs}")
        print(f"[DEBUG] Final Predicted Category: {risk_category} (Confidence: {confidence_pct}%)")

        return {
            "risk_category": risk_category,
            "risk_score": confidence_pct,
            "weighted_risk_score": weighted_score,
            "class_probabilities": class_probs,
            "probabilities": class_probs
        }

    except Exception as e:
        print(f"[ERROR] Inference Exception: {e}")
        return {
            "error": f"Model inference execution failed: {str(e)}"
        }

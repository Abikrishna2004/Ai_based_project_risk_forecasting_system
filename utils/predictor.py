"""
CatBoost Model Inference Engine for Project Risk Prediction
Validates 20 frontend features, applies categorical mappings and numerical outlier capping,
and returns real-time CatBoost risk forecasts, prediction confidence, and overall risk scores.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from models.model_loader import ModelLoader

# Exact 20-feature taxonomy in training order
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
      - prediction_confidence: float (highest class confidence percentage, e.g. 71.8%)
      - overall_risk_score: float (weighted risk score percentage 0-100%)
      - risk_score: float (alias to prediction_confidence)
      - weighted_risk_score: float (alias to overall_risk_score)
      - class_probabilities: dict of class percentages
    """
    try:
        model, metadata, capper, encoders = get_cached_model_and_metadata()
    except Exception as e:
        print(f"Error loading model/metadata: {e}")
        return {
            "error": "Model prediction could not be completed because the model or preprocessing artifacts could not be loaded."
        }

    if model is None or metadata is None:
        return {
            "error": "Model prediction could not be completed because the model or preprocessing artifacts could not be loaded."
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

    # 2. Extract and Process 20 Features
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

    # 2b. Numerical Outlier Capping
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

        # Dynamic class label mapping from model.classes_ if present
        class_labels = ["Low", "Medium", "High", "Critical"]
        if hasattr(model, "classes_") and model.classes_ is not None and len(model.classes_) == 4:
            raw_classes = list(model.classes_)
            class_labels = [inv_target_map.get(str(c), str(c)) for c in raw_classes]

        # Highest probability class
        top_class_idx = int(np.argmax(probs))
        risk_category = class_labels[top_class_idx] if top_class_idx < len(class_labels) else "Medium"

        # Prediction confidence percentage (highest class probability)
        prediction_confidence = round(float(probs[top_class_idx]) * 100.0, 1)

        # Overall weighted risk score percentage (Low=15.0, Medium=45.0, High=75.0, Critical=95.0)
        class_weights = [15.0, 45.0, 75.0, 95.0]
        overall_risk_score = round(sum(p * w for p, w in zip(probs, class_weights)), 1)

        # Class probabilities dictionary
        class_probs = {
            label: round(float(p) * 100.0, 1)
            for label, p in zip(class_labels, probs)
        }

        print(f"[DEBUG] Input DataFrame Shape: {df_input.shape}")
        print(f"[DEBUG] Class Probabilities: {class_probs}")
        print(f"[DEBUG] Predicted Category: {risk_category} | Confidence: {prediction_confidence}% | Overall Risk Score: {overall_risk_score}%")

        return {
            "risk_category": risk_category,
            "prediction_confidence": prediction_confidence,
            "overall_risk_score": overall_risk_score,
            "risk_score": prediction_confidence,        # Alias for backward compatibility
            "weighted_risk_score": overall_risk_score,  # Alias for backward compatibility
            "class_probabilities": class_probs,
            "probabilities": class_probs
        }

    except Exception as e:
        print(f"[ERROR] Inference Exception: {e}")
        return {
            "error": f"Model prediction could not be completed because model inference failed: {str(e)}"
        }

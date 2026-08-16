"""
CatBoost Model Inference Engine for Project Risk Prediction
Validates 20 frontend features, applies categorical mappings and numerical outlier capping,
and returns real-time CatBoost risk forecasts, prediction confidence, and weighted overall risk scores.
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


def get_risk_category_from_score(score: float) -> str:
    """
    Determines final risk category based on exact overall risk score ranges:
      0 <= score <= 35: Low
      35 < score <= 65: Medium
      65 < score <= 85: High
      85 < score <= 100: Critical
    """
    score = float(score)
    if score <= 35.0:
        return "Low"
    elif score <= 65.0:
        return "Medium"
    elif score <= 85.0:
        return "High"
    else:
        return "Critical"


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
      - model_predicted_category: str ("Low", "Medium", "High", "Critical")
      - risk_category: str ("Low", "Medium", "High", "Critical")
      - overall_risk_score: float (weighted risk score 0.0-100.0)
      - prediction_confidence: float (highest class confidence percentage, e.g. 70.0%)
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

        # Target class map (0: Low, 1: Medium, 2: High, 3: Critical)
        inv_target_map = metadata.get(
            "inverse_target_mapping",
            {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"}
        )

        class_labels = ["Low", "Medium", "High", "Critical"]
        if hasattr(model, "classes_") and model.classes_ is not None and len(model.classes_) == 4:
            raw_classes = list(model.classes_)
            class_labels = [inv_target_map.get(str(c), str(c)) for c in raw_classes]

        # Highest probability class
        top_class_idx = int(np.argmax(probs))
        model_predicted_category = class_labels[top_class_idx] if top_class_idx < len(class_labels) else "Medium"

        # Prediction Confidence (highest class probability percentage)
        prediction_confidence = round(float(probs[top_class_idx]) * 100.0, 1)

        # Class probabilities map (percentages and decimals)
        class_probs_pct = {}
        prob_dict_decimal = {}
        for idx, label in enumerate(class_labels):
            p_dec = float(probs[idx])
            prob_dict_decimal[label] = p_dec
            class_probs_pct[label] = round(p_dec * 100.0, 1)

        # Weighted Overall Risk Score Calculation:
        # Low = 20, Medium = 50, High = 75, Critical = 95
        weights = {"Low": 20.0, "Medium": 50.0, "High": 75.0, "Critical": 95.0}
        raw_weighted_score = (
            prob_dict_decimal.get("Low", 0.0) * weights["Low"] +
            prob_dict_decimal.get("Medium", 0.0) * weights["Medium"] +
            prob_dict_decimal.get("High", 0.0) * weights["High"] +
            prob_dict_decimal.get("Critical", 0.0) * weights["Critical"]
        )
        overall_risk_score = float(np.clip(round(raw_weighted_score, 1), 0.0, 100.0))

        # Final Risk Category derived from overall_risk_score ranges
        final_risk_category = get_risk_category_from_score(overall_risk_score)

        print(f"[DEBUG] Input DataFrame Shape: {df_input.shape}")
        print(f"[DEBUG] Class Probabilities: {class_probs_pct}")
        print(f"[DEBUG] Model Predicted Category: {model_predicted_category} | Confidence: {prediction_confidence}%")
        print(f"[DEBUG] Overall Risk Score: {overall_risk_score}% | Final Category: {final_risk_category}")

        return {
            "model_predicted_category": model_predicted_category,
            "risk_category": final_risk_category,
            "overall_risk_score": overall_risk_score,
            "prediction_confidence": prediction_confidence,
            "class_probabilities": class_probs_pct,
            "risk_score": prediction_confidence,        # Alias for backward compatibility
            "weighted_risk_score": overall_risk_score   # Alias for backward compatibility
        }

    except Exception as e:
        print(f"[ERROR] Inference Exception: {e}")
        return {
            "error": f"Model prediction could not be completed because model inference failed: {str(e)}"
        }

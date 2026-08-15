"""
CatBoost Model Inference Engine for Project Risk Prediction
Validates 20 frontend features, applies categorical mappings and numerical outlier capping,
and returns real-time CatBoost risk forecasts and class probabilities.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from models.model_loader import ModelLoader


def load_model_and_metadata():
    """
    Loads and returns:
      1. CatBoost model (Priority 1: models/catboost_risk_model.joblib, Priority 2: models/catboost_risk_model.cbm)
      2. Preprocessing metadata (preprocessing_objects/preprocessing_metadata.json)
      3. Outlier capper bounds (preprocessing_objects/outlier_capper.joblib)
      4. Categorical encoders (preprocessing_objects/categorical_encoders.joblib)
    """
    model = ModelLoader.load_model()
    metadata, capper, encoders = ModelLoader.load_preprocessing_objects()
    return model, metadata, capper, encoders


def predict_project_risk(input_dict):
    """
    Takes user input dictionary, validates 20 required features, applies encodings and capping,
    and executes CatBoost predict_proba().

    Returns dict:
      - risk_category: str ("Low", "Medium", "High", "Critical")
      - risk_score: float (highest class confidence percentage, e.g. 85.5%)
      - class_probabilities: dict of class percentages
      - probabilities: dict of class percentages (backward compatibility)
    """
    model, metadata, capper, encoders = load_model_and_metadata()

    if model is None or metadata is None:
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": "Model or preprocessing metadata not found."
        }

    # Extract 20 ML feature taxonomy
    feature_columns = metadata.get("feature_columns", [
        "project_type", "industry_sector", "methodology", "region", "priority",
        "planned_duration_days", "budget_usd", "requirement_changes_count",
        "vendor_dependency_count", "milestones_missed", "team_size",
        "team_avg_experience_years", "team_turnover_pct", "resource_availability_pct",
        "communication_score", "sponsor_engagement_score", "tech_complexity_score",
        "scope_clarity_score", "external_dependency_score", "defect_count"
    ])

    cat_cols = metadata.get("categorical_columns", [
        "project_type", "industry_sector", "methodology", "region", "priority"
    ])

    num_cols = metadata.get("numerical_columns", [
        "planned_duration_days", "budget_usd", "requirement_changes_count",
        "vendor_dependency_count", "milestones_missed", "team_size",
        "team_avg_experience_years", "team_turnover_pct", "resource_availability_pct",
        "communication_score", "sponsor_engagement_score", "tech_complexity_score",
        "scope_clarity_score", "external_dependency_score", "defect_count"
    ])

    # 1. Feature Presence Validation
    missing_features = [col for col in feature_columns if col not in input_dict]
    if missing_features:
        err_msg = f"Missing required ML features for prediction: {missing_features}"
        print(f"Validation Error: {err_msg}")
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": err_msg
        }

    # 2. Extract and Preprocess 20 Features
    processed_row = {}

    # 2a. Categorical Encoding (Use saved joblib encoders if present, else metadata mappings)
    cat_mappings = encoders if encoders else metadata.get("categorical_mappings", {})
    for col in cat_cols:
        val = str(input_dict.get(col, ""))
        mapping = cat_mappings.get(col, {})
        if val in mapping:
            processed_row[col] = mapping[val]
        elif mapping:
            # Fallback to first encoded index if value not seen
            processed_row[col] = list(mapping.values())[0]
        else:
            processed_row[col] = 0

    # 2b. Numerical Outlier Capping (Use saved joblib capper bounds if present, else metadata bounds)
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

    # 3. Build DataFrame with exact 20 training feature order
    df_input = pd.DataFrame([processed_row])[feature_columns]

    # Validate shape (1, 20)
    if df_input.shape != (1, 20):
        err_shape = f"Invalid feature matrix shape {df_input.shape}, expected (1, 20)"
        print(f"Validation Error: {err_shape}")
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": err_shape
        }

    # 4. CatBoost Inference via predict_proba()
    try:
        probs = model.predict_proba(df_input)[0]

        # Target class map
        inv_target_map = metadata.get(
            "inverse_target_mapping",
            {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"}
        )

        # Identify highest probability class
        top_class_idx = int(np.argmax(probs))
        raw_pred_str = str(top_class_idx)
        risk_category = inv_target_map.get(raw_pred_str, "Medium")

        # Highest class confidence percentage
        confidence_pct = round(float(probs[top_class_idx]) * 100.0, 1)

        # Weighted risk score percentage across classes (Low=15%, Medium=45%, High=75%, Critical=95%)
        weights = [0.15, 0.45, 0.75, 0.95]
        weighted_score = sum(p * w for p, w in zip(probs, weights)) * 100.0
        weighted_score = round(float(weighted_score), 1)

        # Class probabilities map
        class_probs = {
            inv_target_map.get(str(i), f"Class {i}"): round(float(p) * 100.0, 1)
            for i, p in enumerate(probs)
        }

        return {
            "risk_category": risk_category,
            "risk_score": confidence_pct,
            "weighted_risk_score": weighted_score,
            "class_probabilities": class_probs,
            "probabilities": class_probs
        }

    except Exception as e:
        print("Inference Exception:", e)
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": str(e)
        }

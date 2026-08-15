"""
CatBoost Model Inference Engine for Project Risk Prediction
Preprocesses raw user inputs and generates real-time CatBoost risk forecasts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from models.model_loader import ModelLoader


def load_model_and_metadata():
    """Loads and caches the trained 20-feature CatBoost model and preprocessing metadata via ModelLoader."""
    model = ModelLoader.load_model()
    metadata, _, _ = ModelLoader.load_preprocessing_objects()
    return model, metadata


def predict_project_risk(input_dict):
    """
    Takes user input dictionary, processes 20 features, and returns CatBoost risk prediction.
    Returns dict:
      - risk_category: str ("Low", "Medium", "High", "Critical")
      - risk_score: float (0.0 to 100.0)
      - probabilities: dict of class probabilities
    """
    model, metadata = load_model_and_metadata()

    if model is None or metadata is None:
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": "Model or preprocessing metadata not found."
        }

    cat_cols = metadata.get("categorical_columns", [])
    num_cols = metadata.get("numerical_columns", [])
    all_cols = metadata.get("feature_columns", cat_cols + num_cols)

    processed_row = {}

    # Handle Categoricals (Map string values to encoded integers)
    cat_mappings = metadata.get("categorical_mappings", {})
    for col in cat_cols:
        val = str(input_dict.get(col, ""))
        mapping = cat_mappings.get(col, {})
        if val in mapping:
            processed_row[col] = mapping[val]
        elif mapping:
            processed_row[col] = list(mapping.values())[0]
        else:
            processed_row[col] = 0

    # Handle Numericals with outlier bounds
    bounds = metadata.get("capper_bounds", {})
    for col in num_cols:
        try:
            val = float(input_dict.get(col, 0.0))
        except (ValueError, TypeError):
            val = 0.0

        if col in bounds:
            lb = bounds[col]["lower_bound"]
            ub = bounds[col]["upper_bound"]
            val = float(np.clip(val, lb, ub))
        processed_row[col] = val

    # Construct single-row DataFrame in exact feature order
    df_input = pd.DataFrame([processed_row])[all_cols]

    # CatBoost Inference
    try:
        preds = model.predict(df_input)
        probs = model.predict_proba(df_input)[0]

        inv_target_map = metadata.get("inverse_target_mapping", {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"})
        raw_pred_idx = str(preds[0][0]) if hasattr(preds[0], '__len__') else str(preds[0])

        risk_category = inv_target_map.get(raw_pred_idx, "Medium")

        # Compute weighted risk score percentage
        # Classes: 0: Low (15%), 1: Medium (45%), 2: High (75%), 3: Critical (95%)
        weights = [0.15, 0.45, 0.75, 0.95]
        risk_score = sum(p * w for p, w in zip(probs, weights)) * 100.0
        risk_score = round(float(risk_score), 1)

        prob_dict = {inv_target_map.get(str(i), f"Class {i}"): round(float(p) * 100, 1) for i, p in enumerate(probs)}

        return {
            "risk_category": risk_category,
            "risk_score": risk_score,
            "probabilities": prob_dict
        }

    except Exception as e:
        print("Inference Exception:", e)
        return {
            "risk_category": "Medium",
            "risk_score": 50.0,
            "error": str(e)
        }

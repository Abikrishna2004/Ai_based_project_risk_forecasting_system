"""
Model Loader Utility for AI-Based Project Risk Forecasting System
Loads the updated 20-feature CatBoost model and preprocessing objects with strict priority using robust Path resolution.
"""

import os
import json
import joblib
from pathlib import Path
from catboost import CatBoostClassifier

# Project Root Resolution via __file__
BASE_DIR = Path(__file__).resolve().parent.parent
BASELINE_MODEL_DIR = BASE_DIR / "models"
PREPROC_DIR = BASE_DIR / "preprocessing_objects"


class ModelLoader:
    _model = None
    _metadata = None
    _capper = None
    _encoders = None

    @classmethod
    def load_model(cls):
        """Loads and returns the updated 20-feature CatBoost model."""
        if cls._model is not None:
            return cls._model

        # Priority 1: Updated 20-feature Joblib model
        base_joblib = BASELINE_MODEL_DIR / "catboost_risk_model.joblib"

        if base_joblib.exists():
            try:
                cls._model = joblib.load(base_joblib)
                print(
                    "Loaded updated 20-feature CatBoost model "
                    "from 'models/catboost_risk_model.joblib'"
                )
                return cls._model
            except Exception as e:
                print(f"Notice loading Joblib model failed: {e}")

        # Priority 2: Updated 20-feature CBM model
        cbm_path = BASELINE_MODEL_DIR / "catboost_risk_model.cbm"

        if cbm_path.exists():
            try:
                cls._model = CatBoostClassifier()
                cls._model.load_model(str(cbm_path))
                print(
                    "Loaded updated 20-feature CatBoost model "
                    "from 'models/catboost_risk_model.cbm'"
                )
                return cls._model
            except Exception as e:
                print(f"Notice loading CBM model failed: {e}")

        raise FileNotFoundError(
            "No updated CatBoost model found in 'models/'!"
        )

    @classmethod
    def load_preprocessing_objects(cls):
        """Loads and returns preprocessing metadata, capper bounds, and encoders."""
        if cls._metadata is not None and cls._capper is not None and cls._encoders is not None:
            return cls._metadata, cls._capper, cls._encoders

        metadata_path = PREPROC_DIR / "preprocessing_metadata.json"
        capper_path = PREPROC_DIR / "outlier_capper.joblib"
        encoders_path = PREPROC_DIR / "categorical_encoders.joblib"

        # Load Metadata JSON
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cls._metadata = json.load(f)
        else:
            cls._metadata = {
                "feature_columns": [
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
                ],
                "categorical_columns": [
                    "project_type",
                    "industry_sector",
                    "methodology",
                    "region",
                    "priority"
                ],
                "numerical_columns": [
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
                ],
                "target_mapping": {
                    "Low": 0,
                    "Medium": 1,
                    "High": 2,
                    "Critical": 3
                },
                "inverse_target_mapping": {
                    "0": "Low",
                    "1": "Medium",
                    "2": "High",
                    "3": "Critical"
                }
            }

        if capper_path.exists():
            cls._capper = joblib.load(capper_path)
        else:
            cls._capper = {}

        if encoders_path.exists():
            cls._encoders = joblib.load(encoders_path)
        else:
            cls._encoders = {}

        return cls._metadata, cls._capper, cls._encoders

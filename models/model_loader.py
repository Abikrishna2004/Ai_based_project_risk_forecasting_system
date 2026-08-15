

import os
import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier

DATA_DIR = os.getcwd()
OPTIMIZED_MODEL_DIR = os.path.join(DATA_DIR, "Optimized_Model")
BASELINE_MODEL_DIR = os.path.join(DATA_DIR, "models")
PREPROC_DIR = os.path.join(DATA_DIR, "preprocessing_objects")


class ModelLoader:
    _model = None
    _metadata = None
    _capper = None
    _encoders = None

    @classmethod
    def load_model(cls):
        """Loads and returns the CatBoost model instance."""
        if cls._model is not None:
            return cls._model

        opt_path = os.path.join(OPTIMIZED_MODEL_DIR, "optimized_catboost_model.pkl")
        if os.path.exists(opt_path):
            try:
                cls._model = joblib.load(opt_path)
                print("Loaded CatBoost model from 'Optimized_Model/optimized_catboost_model.pkl'")
                return cls._model
            except Exception as e:
                print(f"Notice loading pkl model ({e}), checking alternative formats...")

        # Priority 2: Baseline Joblib Model
        base_joblib = os.path.join(BASELINE_MODEL_DIR, "catboost_risk_model.joblib")
        if os.path.exists(base_joblib):
            cls._model = joblib.load(base_joblib)
            print("Loaded CatBoost model from 'models/catboost_risk_model.joblib'")
            return cls._model

        # Priority 3: Native CBM Model
        cbm_path = os.path.join(BASELINE_MODEL_DIR, "catboost_risk_model.cbm")
        if os.path.exists(cbm_path):
            cls._model = CatBoostClassifier()
            cls._model.load_model(cbm_path)
            print("Loaded CatBoost model from 'models/catboost_risk_model.cbm'")
            return cls._model

        raise FileNotFoundError("No trained CatBoost model artifact found in 'Optimized_Model/' or 'models/'!")

    @classmethod
    def load_preprocessing_objects(cls):
        """Loads and returns preprocessing metadata, capper bounds, and encoders."""
        if cls._metadata is not None and cls._capper is not None:
            return cls._metadata, cls._capper, cls._encoders

        metadata_path = os.path.join(PREPROC_DIR, "preprocessing_metadata.json")
        capper_path = os.path.join(PREPROC_DIR, "outlier_capper.joblib")
        encoders_path = os.path.join(PREPROC_DIR, "categorical_encoders.joblib")

        # Load Metadata JSON
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                cls._metadata = json.load(f)
        else:
            cls._metadata = {
                "categorical_features": [
                    "project_type", "industry_sector", "methodology",
                    "region", "contract_type", "priority", "project_status"
                ],
                "numerical_features": [],
                "target_mapping": {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
            }

        if os.path.exists(capper_path):
            cls._capper = joblib.load(capper_path)
        else:
            cls._capper = {}

        if os.path.exists(encoders_path):
            cls._encoders = joblib.load(encoders_path)
        else:
            cls._encoders = {}

        return cls._metadata, cls._capper, cls._encoders

"""
FastAPI Backend - Predictions Router
Executes CatBoost risk forecasting on 20 features, stores prediction results in SQLite,
and manages individual/batch history deletion.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from backend.database import get_db
from backend.schemas import PredictionRequest
from utils.predictor import predict_project_risk

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_prediction(req: PredictionRequest, db: sqlite3.Connection = Depends(get_db)):
    """
    Executes 20-feature CatBoost ML prediction and saves the result to SQLite database.
    `project_name` is stored in the database but NEVER passed to the ML model.
    """
    input_dict = {
        "project_type": req.project_type,
        "industry_sector": req.industry_sector,
        "methodology": req.methodology,
        "region": req.region,
        "priority": req.priority,
        "planned_duration_days": req.planned_duration_days,
        "budget_usd": req.budget_usd,
        "requirement_changes_count": req.requirement_changes_count,
        "vendor_dependency_count": req.vendor_dependency_count,
        "milestones_missed": req.milestones_missed,
        "team_size": req.team_size,
        "team_avg_experience_years": req.team_avg_experience_years,
        "team_turnover_pct": req.team_turnover_pct,
        "resource_availability_pct": req.resource_availability_pct,
        "communication_score": req.communication_score,
        "sponsor_engagement_score": req.sponsor_engagement_score,
        "tech_complexity_score": req.tech_complexity_score,
        "scope_clarity_score": req.scope_clarity_score,
        "external_dependency_score": req.external_dependency_score,
        "defect_count": req.defect_count
    }

    # Run CatBoost Model Inference
    prediction_result = predict_project_risk(input_dict)

    if "error" in prediction_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=prediction_result["error"]
        )

    risk_category = prediction_result.get("risk_category", "Medium")
    prediction_confidence = prediction_result.get("prediction_confidence", prediction_result.get("risk_score", 50.0))
    overall_risk_score = prediction_result.get("overall_risk_score", prediction_result.get("weighted_risk_score", 50.0))
    class_probs = prediction_result.get("class_probabilities", {})

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor = db.cursor()

    try:
        cursor.execute("""
            INSERT INTO project_predictions (
                user_id, email, project_name, risk_level, risk_score, prediction_confidence, overall_risk_score, input_features_json, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(req.user_id),
            (req.email or "").strip().lower(),
            req.project_name.strip(),
            risk_category,
            float(prediction_confidence),
            float(prediction_confidence),
            float(overall_risk_score),
            json.dumps(input_dict),
            timestamp
        ))
        db.commit()
        pred_id = cursor.lastrowid

        cursor.execute("SELECT * FROM project_predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        record_dict = dict(row)
        record_dict["_id"] = str(record_dict["id"])
        record_dict["input_features"] = input_dict

        return {
            "success": True,
            "message": f"Prediction for project '{req.project_name}' created successfully!",
            "risk_category": risk_category,
            "prediction_confidence": prediction_confidence,
            "overall_risk_score": overall_risk_score,
            "risk_score": prediction_confidence,
            "weighted_risk_score": overall_risk_score,
            "class_probabilities": class_probs,
            "prediction_record": record_dict
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving prediction to database: {str(e)}"
        )


@router.get("/user/{user_id}")
def get_user_predictions(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Retrieves all project prediction records for a specific user."""
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM project_predictions WHERE user_id = ? ORDER BY id DESC
    """, (str(user_id),))
    rows = cursor.fetchall()

    predictions = []
    for r in rows:
        item = dict(r)
        item["_id"] = str(item["id"])
        if item.get("prediction_confidence") is None:
            item["prediction_confidence"] = item.get("risk_score", 0.0)
        if item.get("overall_risk_score") is None:
            item["overall_risk_score"] = item.get("risk_score", 0.0)

        try:
            item["input_features"] = json.loads(item["input_features_json"])
        except Exception:
            item["input_features"] = {}
        predictions.append(item)

    return {
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    }


@router.get("/{prediction_id}")
def get_prediction_by_id(prediction_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Retrieves a single project prediction record by ID."""
    cursor = db.cursor()
    cursor.execute("SELECT * FROM project_predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID '{prediction_id}' not found."
        )

    item = dict(row)
    item["_id"] = str(item["id"])
    if item.get("prediction_confidence") is None:
        item["prediction_confidence"] = item.get("risk_score", 0.0)
    if item.get("overall_risk_score") is None:
        item["overall_risk_score"] = item.get("risk_score", 0.0)

    try:
        item["input_features"] = json.loads(item["input_features_json"])
    except Exception:
        item["input_features"] = {}

    return {
        "success": True,
        "prediction": item
    }


@router.delete("/{prediction_id}")
def delete_prediction(prediction_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Deletes an individual prediction record by ID."""
    cursor = db.cursor()
    cursor.execute("SELECT id FROM project_predictions WHERE id = ?", (prediction_id,))
    if not cursor.fetchone():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID '{prediction_id}' not found."
        )

    cursor.execute("DELETE FROM project_predictions WHERE id = ?", (prediction_id,))
    db.commit()

    return {
        "success": True,
        "message": f"Prediction record #{prediction_id} deleted successfully."
    }


@router.delete("/user/{user_id}/all")
def delete_all_user_predictions(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Clears all prediction history records for a user."""
    cursor = db.cursor()
    cursor.execute("DELETE FROM project_predictions WHERE user_id = ?", (str(user_id),))
    deleted_count = cursor.rowcount
    db.commit()

    return {
        "success": True,
        "message": f"Cleared {deleted_count} prediction records for user '{user_id}'."
    }

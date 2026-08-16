"""
FastAPI Backend - Dashboard Metrics Router
Calculates real-time KPI metrics (total projects, risk category breakdown, average risk score) for the dashboard.
"""

import json
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, status
from backend.database import get_db
from backend.schemas import DashboardMetricsResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/metrics/{user_id}", response_model=DashboardMetricsResponse)
def get_dashboard_metrics(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Calculates real-time project risk metrics for a user's dashboard."""
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM project_predictions WHERE user_id = ? ORDER BY id DESC
    """, (str(user_id),))
    rows = cursor.fetchall()

    predictions = []
    for r in rows:
        item = dict(r)
        item["_id"] = str(item["id"])
        try:
            item["input_features"] = json.loads(item["input_features_json"])
        except Exception:
            item["input_features"] = {}
        predictions.append(item)

    total_projects = len(predictions)
    if total_projects == 0:
        return DashboardMetricsResponse(
            total_projects=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            avg_risk_score_pct="0%",
            avg_risk_score_num=0.0,
            predictions=[]
        )

    high_count = 0
    medium_count = 0
    low_count = 0
    total_score_sum = 0.0

    for item in predictions:
        lvl = str(item.get("risk_level", "")).lower()
        score = float(item.get("risk_score", 0.0))
        total_score_sum += score

        if "high" in lvl or "critical" in lvl:
            high_count += 1
        elif "medium" in lvl:
            medium_count += 1
        else:
            low_count += 1

    avg_score = round(total_score_sum / total_projects, 1)

    return DashboardMetricsResponse(
        total_projects=total_projects,
        high_risk_count=high_count,
        medium_risk_count=medium_count,
        low_risk_count=low_count,
        avg_risk_score_pct=f"{avg_score}%",
        avg_risk_score_num=avg_score,
        predictions=predictions
    )

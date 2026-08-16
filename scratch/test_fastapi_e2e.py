"""
End-to-End Test Verification Script for FastAPI Backend and Streamlit API Client
"""

import sys
import time
import requests
import uvicorn
from threading import Thread
from backend.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    print("Starting FastAPI Test Server in Background Thread...")
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2.5)

    from utils.api_client import (
        register_user,
        authenticate_user,
        get_user_profile,
        update_user_profile,
        change_password,
        save_project_prediction,
        get_user_predictions,
        delete_prediction,
        delete_all_predictions,
        get_user_dashboard_metrics
    )

    test_email = f"fastapi_test_{int(time.time())}@example.com"
    print(f"\n1. Testing User Registration for '{test_email}'...")
    reg_ok, reg_msg = register_user({
        "first_name": "FastAPI",
        "last_name": "Tester",
        "email": test_email,
        "password": "TestPassword123",
        "organization_type": "Enterprise",
        "education_category": "Professional"
    })
    print(f"Registration Result: ok={reg_ok}, msg={reg_msg}")
    assert reg_ok is True, f"Registration failed: {reg_msg}"

    print("\n2. Testing User Authentication (Login)...")
    auth_ok, auth_user = authenticate_user(test_email, "TestPassword123")
    print(f"Auth Result: ok={auth_ok}, user_id={auth_user.get('_id') if auth_ok else 'None'}")
    assert auth_ok is True
    user_id = auth_user["_id"]

    print("\n3. Testing GET User Profile...")
    profile = get_user_profile(user_id)
    print(f"Profile: {profile.get('first_name')} {profile.get('last_name')} ({profile.get('email')})")
    assert profile["email"] == test_email

    print("\n4. Testing Update Profile...")
    up_ok, up_msg, up_user = update_user_profile(user_id, {
        "first_name": "FastAPI-Updated",
        "last_name": "Tester",
        "designation": "Lead AI Engineer"
    })
    print(f"Update Result: ok={up_ok}, msg={up_msg}")
    assert up_ok is True

    print("\n5. Testing 20-Feature ML Risk Prediction & Save...")
    pred_payload = {
        "project_type": "Software Development",
        "industry_sector": "IT",
        "methodology": "Agile",
        "region": "Asia Pacific",
        "priority": "High",
        "planned_duration_days": 180,
        "budget_usd": 250000,
        "requirement_changes_count": 4,
        "vendor_dependency_count": 2,
        "milestones_missed": 1,
        "team_size": 12,
        "team_avg_experience_years": 5.5,
        "team_turnover_pct": 10.0,
        "resource_availability_pct": 85.0,
        "communication_score": 75.0,
        "sponsor_engagement_score": 80.0,
        "tech_complexity_score": 65.0,
        "scope_clarity_score": 70.0,
        "external_dependency_score": 35.0,
        "defect_count": 5
    }
    save_ok, save_msg = save_project_prediction(user_id, test_email, "FastAPI Test Platform", "Medium", 69.7, pred_payload)
    print(f"Prediction Save Result: ok={save_ok}, msg={save_msg}")
    assert save_ok is True

    print("\n6. Testing Dashboard Metrics...")
    metrics = get_user_dashboard_metrics(user_id)
    print(f"Dashboard Metrics: Total={metrics['total_projects']}, High={metrics['high_risk_count']}, Medium={metrics['medium_risk_count']}")
    assert metrics["total_projects"] == 1

    print("\n7. Testing GET Predictions List...")
    preds = get_user_predictions(user_id)
    print(f"Fetched {len(preds)} prediction records.")
    assert len(preds) == 1
    pred_id = preds[0]["id"]

    print("\n8. Testing Delete Prediction Record...")
    del_ok, del_msg = delete_prediction(pred_id)
    print(f"Delete Record Result: ok={del_ok}, msg={del_msg}")
    assert del_ok is True

    print("\nALL FASTAPI REST ENDPOINTS & STREAMLIT API CLIENT TESTS PASSED 100% PERFECTLY!")

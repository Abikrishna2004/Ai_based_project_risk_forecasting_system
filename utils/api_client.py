"""
Streamlit API Client Interface for AI-Based Project Risk Forecasting System
Communicates with FastAPI Backend REST API (http://127.0.0.1:8000) with automatic direct database fallback.
"""

import os
import sys
import json
import requests
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.database_client import (
    register_user as db_register_user,
    authenticate_user as db_authenticate_user,
    save_project_prediction as db_save_project_prediction,
    get_user_predictions as db_get_user_predictions,
    get_user_dashboard_metrics as db_get_user_dashboard_metrics,
    _get_db_connection,
    hash_password
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SEC = 5.0


def _is_backend_online() -> bool:
    """Checks if FastAPI backend server is reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 1. AUTHENTICATION
# -----------------------------------------------------------------------------
def register_user(user_data):
    """Registers a new user account via FastAPI or direct DB fallback."""
    if _is_backend_online():
        try:
            resp = requests.post(f"{API_BASE_URL}/auth/register", json=user_data, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code in (200, 201) and data.get("success"):
                return True, data.get("message", "Registration successful.")
            return False, data.get("detail", data.get("message", "Registration failed."))
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI register failed ({e}), using direct DB fallback.")

    return db_register_user(user_data)


def authenticate_user(email, password):
    """Authenticates user credentials via FastAPI or direct DB fallback."""
    if _is_backend_online():
        try:
            resp = requests.post(f"{API_BASE_URL}/auth/login", json={"email": email, "password": password}, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200 and data.get("success"):
                return True, data.get("user")
            return False, data.get("detail", data.get("message", "Authentication failed."))
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI authenticate failed ({e}), using direct DB fallback.")

    return db_authenticate_user(email, password)


# Aliases for backward compatibility
register_user_atlas = register_user
authenticate_user_atlas = authenticate_user


# -----------------------------------------------------------------------------
# 2. USER PROFILE MANAGEMENT
# -----------------------------------------------------------------------------
def get_user_profile(user_id):
    """Retrieves user profile information."""
    if _is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/users/profile/{user_id}", timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI get_profile failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["_id"] = str(d["id"])
        d.pop("password_hash", None)
        return d
    return None


def update_user_profile(user_id, profile_data):
    """Updates user profile information."""
    if _is_backend_online():
        try:
            resp = requests.put(f"{API_BASE_URL}/users/profile/{user_id}", json=profile_data, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200:
                return True, data.get("message", "Profile updated successfully."), data.get("user")
            return False, data.get("detail", "Failed to update profile."), None
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI update_profile failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE users SET
                first_name = ?, last_name = ?, organization_type = ?, education_category = ?,
                school_name = ?, standard = ?, university_name = ?, degree = ?,
                academic_year = ?, designation = ?, experience_level = ?
            WHERE id = ?
        """, (
            profile_data.get("first_name"), profile_data.get("last_name"), profile_data.get("organization_type"),
            profile_data.get("education_category"), profile_data.get("school_name"), profile_data.get("standard"),
            profile_data.get("university_name"), profile_data.get("degree"), profile_data.get("academic_year"),
            profile_data.get("designation"), profile_data.get("experience_level"), user_id
        ))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        updated_user = dict(cursor.fetchone())
        conn.close()
        updated_user["_id"] = str(updated_user["id"])
        updated_user.pop("password_hash", None)
        return True, "Profile updated successfully.", updated_user
    except Exception as e:
        conn.close()
        return False, f"Update error: {e}", None


def change_password(user_id, old_password, new_password):
    """Changes user password."""
    if _is_backend_online():
        try:
            payload = {"old_password": old_password, "new_password": new_password}
            resp = requests.put(f"{API_BASE_URL}/users/change-password/{user_id}", json=payload, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200:
                return True, data.get("message", "Password changed successfully.")
            return False, data.get("detail", "Password change failed.")
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI change_password failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "User not found."

    from backend.security import verify_password, hash_password_secure
    stored_hash = row["password_hash"]
    if not verify_password(old_password, stored_hash):
        conn.close()
        return False, "Current password is incorrect."

    new_hash = hash_password_secure(new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


def upload_profile_image(user_id, image_bytes, filename="avatar.jpg"):
    """Uploads user profile avatar image."""
    if _is_backend_online():
        try:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            resp = requests.post(f"{API_BASE_URL}/users/profile-image/{user_id}", files=files, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200:
                return True, data.get("message", "Image uploaded successfully."), data.get("profile_image")
            return False, data.get("detail", "Image upload failed."), None
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI upload_profile_image failed ({e}).")

    return False, "FastAPI backend server is offline.", None


# -----------------------------------------------------------------------------
# 3. PREDICTION & HISTORY MANAGEMENT
# -----------------------------------------------------------------------------
def save_project_prediction(user_id, email, project_name, risk_level, risk_score, input_features, prediction_confidence=None, overall_risk_score=None):
    """Saves project risk prediction result including separate confidence and overall risk score."""
    if _is_backend_online():
        try:
            payload = {
                "user_id": str(user_id),
                "email": email,
                "project_name": project_name,
                "project_type": input_features.get("project_type", "Software Development"),
                "industry_sector": input_features.get("industry_sector", "IT"),
                "methodology": input_features.get("methodology", "Agile"),
                "region": input_features.get("region", "Asia Pacific"),
                "priority": input_features.get("priority", "High"),
                "planned_duration_days": float(input_features.get("planned_duration_days", 180)),
                "budget_usd": float(input_features.get("budget_usd", 250000)),
                "requirement_changes_count": float(input_features.get("requirement_changes_count", 4)),
                "vendor_dependency_count": float(input_features.get("vendor_dependency_count", 2)),
                "milestones_missed": float(input_features.get("milestones_missed", 1)),
                "team_size": float(input_features.get("team_size", 12)),
                "team_avg_experience_years": float(input_features.get("team_avg_experience_years", 5.5)),
                "team_turnover_pct": float(input_features.get("team_turnover_pct", 10.0)),
                "resource_availability_pct": float(input_features.get("resource_availability_pct", 85.0)),
                "communication_score": float(input_features.get("communication_score", 75.0)),
                "sponsor_engagement_score": float(input_features.get("sponsor_engagement_score", 80.0)),
                "tech_complexity_score": float(input_features.get("tech_complexity_score", 65.0)),
                "scope_clarity_score": float(input_features.get("scope_clarity_score", 70.0)),
                "external_dependency_score": float(input_features.get("external_dependency_score", 35.0)),
                "defect_count": float(input_features.get("defect_count", 5.0))
            }
            resp = requests.post(f"{API_BASE_URL}/predictions", json=payload, timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code in (200, 201) and data.get("success"):
                return True, data.get("message", "Prediction saved successfully.")
            return False, data.get("detail", data.get("message", "Failed to save prediction."))
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI save_prediction failed ({e}), using direct DB fallback.")

    return db_save_project_prediction(user_id, email, project_name, risk_level, risk_score, input_features, prediction_confidence=prediction_confidence, overall_risk_score=overall_risk_score)


def get_user_predictions(user_id):
    """Retrieves all project prediction records for a user."""
    if _is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/predictions/user/{user_id}", timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("predictions", [])
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI get_predictions failed ({e}), using direct DB fallback.")

    return db_get_user_predictions(user_id)


def get_prediction_by_id(prediction_id):
    """Retrieves a single project prediction record by ID."""
    if _is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/predictions/{prediction_id}", timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("prediction")
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI get_prediction_by_id failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM project_predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
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
        return item
    return None


def delete_prediction(prediction_id):
    """Deletes an individual prediction record by ID."""
    if _is_backend_online():
        try:
            resp = requests.delete(f"{API_BASE_URL}/predictions/{prediction_id}", timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200:
                return True, data.get("message", "Prediction deleted.")
            return False, data.get("detail", "Failed to delete prediction.")
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI delete_prediction failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM project_predictions WHERE id = ?", (prediction_id,))
        db_count = cursor.rowcount
        conn.commit()
        conn.close()
        if db_count > 0:
            return True, f"Prediction record #{prediction_id} deleted successfully."
        return False, f"Prediction record #{prediction_id} not found."
    except Exception as e:
        conn.close()
        return False, f"Delete error: {e}"


def delete_all_predictions(user_id):
    """Clears all prediction history records for a user."""
    if _is_backend_online():
        try:
            resp = requests.delete(f"{API_BASE_URL}/predictions/user/{user_id}/all", timeout=TIMEOUT_SEC)
            data = resp.json()
            if resp.status_code == 200:
                return True, data.get("message", "All predictions cleared.")
            return False, data.get("detail", "Failed to clear predictions.")
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI delete_all_predictions failed ({e}).")

    conn = _get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM project_predictions WHERE user_id = ?", (str(user_id),))
        db_count = cursor.rowcount
        conn.commit()
        conn.close()
        return True, f"Cleared {db_count} prediction records for user."
    except Exception as e:
        conn.close()
        return False, f"Delete error: {e}"


# -----------------------------------------------------------------------------
# 4. DASHBOARD ANALYTICS
# -----------------------------------------------------------------------------
def get_user_dashboard_metrics(user_id):
    """Retrieves real-time dashboard risk metrics for a user."""
    if _is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/dashboard/metrics/{user_id}", timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[API CLIENT NOTICE] FastAPI get_dashboard_metrics failed ({e}), using direct DB fallback.")

    return db_get_user_dashboard_metrics(user_id)

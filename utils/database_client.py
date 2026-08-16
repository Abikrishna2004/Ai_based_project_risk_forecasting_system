"""
SQLite Database Engine for AI-Based Project Risk Forecasting System
Provides real-time data management for user authentication, registration, and risk prediction persistence.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_DIR = os.path.join(os.getcwd(), "data")
DB_FILE = os.path.join(DB_DIR, "project_risk.db")


def _get_db_connection():
    """Initializes and returns a SQLite database connection with row factory."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db_schema():
    """Creates database tables and executes safe column migrations if they do not exist."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            organization_type TEXT,
            education_category TEXT,
            school_name TEXT,
            standard TEXT,
            university_name TEXT,
            degree TEXT,
            academic_year TEXT,
            designation TEXT,
            experience_level TEXT,
            profile_image TEXT,
            created_at TEXT
        );
    """)

    # Safe Migration: Check if profile_image column exists
    cursor.execute("PRAGMA table_info(users);")
    user_cols = [row["name"] for row in cursor.fetchall()]
    if "profile_image" not in user_cols:
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_image TEXT;")
        except Exception:
            pass

    # Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            project_name TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score REAL NOT NULL,
            prediction_confidence REAL,
            overall_risk_score REAL,
            input_features_json TEXT NOT NULL,
            analyzed_at TEXT
        );
    """)

    # Safe Migrations for prediction_confidence and overall_risk_score
    cursor.execute("PRAGMA table_info(project_predictions);")
    pred_cols = [row["name"] for row in cursor.fetchall()]
    if "prediction_confidence" not in pred_cols:
        try:
            cursor.execute("ALTER TABLE project_predictions ADD COLUMN prediction_confidence REAL;")
        except Exception:
            pass

    if "overall_risk_score" not in pred_cols:
        try:
            cursor.execute("ALTER TABLE project_predictions ADD COLUMN overall_risk_score REAL;")
        except Exception:
            pass

    conn.commit()
    conn.close()


# Initialize database schema on module import
_init_db_schema()


def hash_password(password):
    """Computes SHA-256 hash for secure password storage."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def register_user(user_data, custom_uri=None):
    """Registers a new user in the enterprise database."""
    email = user_data.get("email", "").strip().lower()
    if not email:
        return False, "Email address is required."

    raw_password = user_data.get("password", "")
    password_digest = hash_password(raw_password)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    conn = _get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users (
                first_name, last_name, email, password_hash,
                organization_type, education_category, school_name, standard,
                university_name, degree, academic_year, designation, experience_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_data.get("first_name", "").strip(),
            user_data.get("last_name", "").strip(),
            email,
            password_digest,
            user_data.get("organization_type", "Startup"),
            user_data.get("education_category", "College / University Student"),
            user_data.get("school_name", ""),
            user_data.get("standard", ""),
            user_data.get("university_name", ""),
            user_data.get("degree", ""),
            user_data.get("academic_year", ""),
            user_data.get("designation", ""),
            user_data.get("experience_level", ""),
            timestamp
        ))
        conn.commit()
        conn.close()
        return True, f"Account registered successfully for '{email}'."

    except sqlite3.IntegrityError:
        conn.close()
        return False, f"User with email '{email}' is already registered."
    except Exception as e:
        conn.close()
        return False, f"Registration Error: {e}"


# Alias for backward compatibility
register_user_atlas = register_user


def authenticate_user(email, password, custom_uri=None):
    """Authenticates user credentials against the database."""
    email = email.strip().lower()

    conn = _get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,))
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return False, "No registered account found with this email."

    user_dict = dict(user_row)
    stored_hash = user_dict.get("password_hash", "")

    from backend.security import verify_password
    if verify_password(password, stored_hash):
        user_dict["_id"] = str(user_dict["id"])
        return True, user_dict

    return False, "Invalid password. Authentication failed."


# Alias for backward compatibility
authenticate_user_atlas = authenticate_user


def save_project_prediction(user_id, email, project_name, risk_level, risk_score, input_features, prediction_confidence=None, overall_risk_score=None, custom_uri=None):
    """Saves project risk prediction results including separate confidence and overall risk score."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    conf_val = float(prediction_confidence) if prediction_confidence is not None else float(risk_score)
    overall_val = float(overall_risk_score) if overall_risk_score is not None else float(risk_score)

    conn = _get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO project_predictions (
                user_id, email, project_name, risk_level, risk_score, prediction_confidence, overall_risk_score, input_features_json, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(user_id),
            email.strip().lower(),
            project_name.strip(),
            risk_level,
            float(conf_val),
            float(conf_val),
            float(overall_val),
            json.dumps(input_features),
            timestamp
        ))
        conn.commit()
        conn.close()
        return True, f"Prediction for project '{project_name}' saved successfully!"
    except Exception as e:
        conn.close()
        return False, f"Error saving prediction: {e}"


def get_user_predictions(user_id, custom_uri=None):
    """Retrieves all project prediction records for a given user."""
    conn = _get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM project_predictions WHERE user_id = ? ORDER BY id DESC
    """, (str(user_id),))

    rows = cursor.fetchall()
    conn.close()

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

    return predictions


def get_user_dashboard_metrics(user_id, custom_uri=None):
    """Calculates real-time project risk metrics for the user's dashboard."""
    predictions = get_user_predictions(user_id, custom_uri)

    total_projects = len(predictions)
    if total_projects == 0:
        return {
            "total_projects": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "avg_risk_score_pct": "0%",
            "avg_risk_score_num": 0.0,
            "predictions": []
        }

    high_count = 0
    medium_count = 0
    low_count = 0
    total_score_sum = 0.0

    for item in predictions:
        lvl = str(item.get("risk_level", "")).lower()
        score = float(item.get("overall_risk_score", item.get("risk_score", 0.0)))
        total_score_sum += score

        if "high" in lvl or "critical" in lvl:
            high_count += 1
        elif "medium" in lvl:
            medium_count += 1
        else:
            low_count += 1

    avg_score = round(total_score_sum / total_projects, 1)

    return {
        "total_projects": total_projects,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "avg_risk_score_pct": f"{avg_score}%",
        "avg_risk_score_num": avg_score,
        "predictions": predictions
    }

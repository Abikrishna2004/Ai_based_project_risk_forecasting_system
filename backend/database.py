"""
FastAPI Backend - Database Management Module
Connects to SQLite database at data/project_risk.db and initializes tables safely.
"""

import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data"
DB_FILE = DB_DIR / "project_risk.db"


def get_db_connection():
    """Returns a thread-safe SQLite connection with Row factory."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db():
    """FastAPI Dependency for database connection."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initializes SQLite database schema and performs safe migration for profile_image column."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
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

    # Safe Migration: Check if profile_image column exists, if not ADD IT
    cursor.execute("PRAGMA table_info(users);")
    columns = [row["name"] for row in cursor.fetchall()]
    if "profile_image" not in columns:
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_image TEXT;")
            print("[INFO] Added 'profile_image' column to users table.")
        except Exception as e:
            print(f"[NOTE] Migration notice for profile_image: {e}")

    # 2. Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            project_name TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score REAL NOT NULL,
            input_features_json TEXT NOT NULL,
            analyzed_at TEXT
        );
    """)

    conn.commit()
    conn.close()
    print("[SUCCESS] SQLite database initialized successfully at 'data/project_risk.db'.")

"""
FastAPI Backend - Dependencies & Authorization Helpers Module
"""

import sqlite3
from fastapi import Depends, HTTPException, status
from backend.database import get_db


def verify_user_exists(user_id: str, db: sqlite3.Connection = Depends(get_db)) -> dict:
    """
    Verifies user existence in SQLite database.
    Raises HTTP 404 if user not found.
    """
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found."
        )

    user_dict = dict(user_row)
    user_dict["_id"] = str(user_dict["id"])
    return user_dict

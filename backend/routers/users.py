"""
FastAPI Backend - User Profile Router
Manages User Profile GET, UPDATE, Change Password, and Profile Image Upload.
"""

import os
import shutil
import sqlite3
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from backend.database import get_db, BASE_DIR
from backend.schemas import UserProfileUpdateRequest, PasswordChangeRequest
from backend.security import hash_password_secure, verify_password

router = APIRouter(prefix="/users", tags=["Users"])
UPLOADS_DIR = BASE_DIR / "uploads" / "profile_images"


@router.get("/profile/{user_id}")
def get_user_profile(user_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Retrieves user profile details by user ID."""
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
    user_dict.pop("password_hash", None)
    return user_dict


@router.put("/profile/{user_id}")
def update_user_profile(user_id: str, req: UserProfileUpdateRequest, db: sqlite3.Connection = Depends(get_db)):
    """Updates user profile details."""
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found."
        )

    current_data = dict(user_row)
    updated_fields = {
        "first_name": req.first_name if req.first_name is not None else current_data.get("first_name"),
        "last_name": req.last_name if req.last_name is not None else current_data.get("last_name"),
        "organization_type": req.organization_type if req.organization_type is not None else current_data.get("organization_type"),
        "education_category": req.education_category if req.education_category is not None else current_data.get("education_category"),
        "school_name": req.school_name if req.school_name is not None else current_data.get("school_name"),
        "standard": req.standard if req.standard is not None else current_data.get("standard"),
        "university_name": req.university_name if req.university_name is not None else current_data.get("university_name"),
        "degree": req.degree if req.degree is not None else current_data.get("degree"),
        "academic_year": req.academic_year if req.academic_year is not None else current_data.get("academic_year"),
        "designation": req.designation if req.designation is not None else current_data.get("designation"),
        "experience_level": req.experience_level if req.experience_level is not None else current_data.get("experience_level"),
    }

    cursor.execute("""
        UPDATE users SET
            first_name = ?, last_name = ?, organization_type = ?, education_category = ?,
            school_name = ?, standard = ?, university_name = ?, degree = ?,
            academic_year = ?, designation = ?, experience_level = ?
        WHERE id = ?
    """, (
        updated_fields["first_name"], updated_fields["last_name"], updated_fields["organization_type"],
        updated_fields["education_category"], updated_fields["school_name"], updated_fields["standard"],
        updated_fields["university_name"], updated_fields["degree"], updated_fields["academic_year"],
        updated_fields["designation"], updated_fields["experience_level"], user_id
    ))
    db.commit()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    updated_user = dict(cursor.fetchone())
    updated_user["_id"] = str(updated_user["id"])
    updated_user.pop("password_hash", None)

    return {
        "success": True,
        "message": "User profile updated successfully.",
        "user": updated_user
    }


@router.put("/change-password/{user_id}")
def change_password(user_id: str, req: PasswordChangeRequest, db: sqlite3.Connection = Depends(get_db)):
    """Changes user password after validating old password."""
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found."
        )

    user_dict = dict(user_row)
    stored_hash = user_dict.get("password_hash", "")

    if not verify_password(req.old_password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect."
        )

    new_hash = hash_password_secure(req.new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    db.commit()

    return {
        "success": True,
        "message": "Password changed successfully."
    }


@router.post("/profile-image/{user_id}")
async def upload_profile_image(user_id: str, file: UploadFile = File(...), db: sqlite3.Connection = Depends(get_db)):
    """Uploads and updates user profile image."""
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found."
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    dest_filename = f"user_{user_id}{ext}"
    file_path = UPLOADS_DIR / dest_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/static/profile_images/{dest_filename}"
    cursor.execute("UPDATE users SET profile_image = ? WHERE id = ?", (image_url, user_id))
    db.commit()

    return {
        "success": True,
        "message": "Profile image uploaded successfully.",
        "profile_image": image_url
    }

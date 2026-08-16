"""
FastAPI Backend - Authentication Router
Handles User Registration (POST /auth/register) and Login (POST /auth/login).
"""

import sqlite3
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from backend.database import get_db
from backend.schemas import UserRegisterRequest, UserLoginRequest, AuthResponse
from backend.security import hash_password_secure, verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: UserRegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    """Registers a new user account with secure password hashing."""
    email = req.email.strip().lower()
    cursor = db.cursor()

    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,))
    if cursor.fetchone():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{email}' is already registered."
        )

    # Securely hash password
    pwd_hash = hash_password_secure(req.password)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        cursor.execute("""
            INSERT INTO users (
                first_name, last_name, email, password_hash,
                organization_type, education_category, school_name, standard,
                university_name, degree, academic_year, designation, experience_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            req.first_name.strip(),
            req.last_name.strip(),
            email,
            pwd_hash,
            req.organization_type or "Startup",
            req.education_category or "College / University Student",
            req.school_name or "",
            req.standard or "",
            req.university_name or "",
            req.degree or "",
            req.academic_year or "",
            req.designation or "",
            req.experience_level or "",
            timestamp
        ))
        db.commit()

        user_id = cursor.lastrowid
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()
        user_dict = dict(user_row)
        user_dict["_id"] = str(user_dict["id"])
        user_dict.pop("password_hash", None)

        return AuthResponse(
            success=True,
            message=f"Account registered successfully for '{email}'.",
            user=user_dict
        )

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{email}' is already registered."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration Error: {str(e)}"
        )


@router.post("/login", response_model=AuthResponse)
def login_user(req: UserLoginRequest, db: sqlite3.Connection = Depends(get_db)):
    """Authenticates user credentials against SQLite database."""
    email = req.email.strip().lower()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email,))
    user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No registered account found with this email."
        )

    user_dict = dict(user_row)
    stored_hash = user_dict.get("password_hash", "")

    if not verify_password(req.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password. Authentication failed."
        )

    user_dict["_id"] = str(user_dict["id"])
    user_dict.pop("password_hash", None)

    return AuthResponse(
        success=True,
        message=f"Welcome back, {user_dict.get('first_name', 'User')}!",
        user=user_dict
    )

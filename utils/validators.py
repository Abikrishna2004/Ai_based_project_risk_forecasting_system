"""
Form Validation Helpers for Authentication Portal
"""

import re


def is_valid_email(email):
    """Validates email format using regex."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None


def validate_password_strength(password):
    """
    Checks password strength rules:
    - Minimum 6 characters
    """
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, "Password strength OK."


def validate_registration_fields(first_name, last_name, email, password, confirm_password):
    """Validates required registration fields."""
    if not first_name.strip() or not last_name.strip():
        return False, "First Name and Last Name are required."

    if not is_valid_email(email):
        return False, "Please enter a valid email address."

    is_strong, pwd_msg = validate_password_strength(password)
    if not is_strong:
        return False, pwd_msg

    if password != confirm_password:
        return False, "Passwords do not match. Please confirm your password."

    return True, "All fields valid."

"""
FastAPI Backend - Security & Password Hashing Module
Implements secure password hashing and dual verification for legacy SHA-256 and salted PBKDF2 hashes.
"""

import os
import hashlib
import hmac

SECRET_SALT = os.getenv("AUTH_SECRET_SALT", "AiProjectRiskForecastingSystemSecretSalt2026")


def hash_password_legacy(password: str) -> str:
    """Computes legacy SHA-256 digest for compatibility."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def hash_password_secure(password: str) -> str:
    """Computes secure salted PBKDF2 HMAC SHA-256 hash."""
    salt = SECRET_SALT.encode('utf-8')
    derived = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return f"pbkdf2_sha256${derived.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verifies plain password against stored hash.
    Supports both new PBKDF2 hashes and legacy SHA-256 hashes.
    """
    if not stored_hash or not plain_password:
        return False

    # 1. PBKDF2 Salted Verification
    if stored_hash.startswith("pbkdf2_sha256$"):
        expected_hash = hash_password_secure(plain_password)
        return hmac.compare_digest(stored_hash, expected_hash)

    # 2. Legacy SHA-256 Verification
    legacy_hash = hash_password_legacy(plain_password)
    return hmac.compare_digest(stored_hash, legacy_hash)

"""
FastAPI Backend Main Application
AI-Based Project Risk Forecasting System REST API Server
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import init_db, BASE_DIR
from backend.routers import auth, users, predictions, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for database initialization on startup."""
    print("[STARTUP] Initializing SQLite database schema...")
    init_db()
    yield
    print("[SHUTDOWN] Fast API server stopping...")


app = FastAPI(
    title="AI-Based Project Risk Forecasting System REST API",
    description="Enterprise REST API for authentication, user profiles, 20-feature CatBoost risk predictions, and dashboard analytics.",
    version="2.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# CORS MIDDLEWARE CONFIGURATION
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Streamlit frontend and local/staging clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# STATIC FILES MOUNTING FOR PROFILE IMAGES
# -----------------------------------------------------------------------------
UPLOADS_DIR = BASE_DIR / "uploads" / "profile_images"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/profile_images", StaticFiles(directory=str(UPLOADS_DIR)), name="profile_images")

# -----------------------------------------------------------------------------
# ROUTER REGISTRATION
# -----------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)


@app.get("/", tags=["Health Check"])
def root():
    """API Root Health Check Endpoint."""
    return {
        "status": "online",
        "system": "AI-Based Project Risk Forecasting System REST API",
        "version": "2.0.0",
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

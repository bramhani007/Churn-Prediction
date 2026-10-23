import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-grocery-churn-prediction-2026")
    
    # SQLite Database location
    DATABASE_PATH = os.environ.get(
        "DATABASE_PATH",
        str(BASE_DIR / "database" / "churn.db")
    )
    
    # Machine Learning artifacts
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "churn_pipeline.joblib")
    )
    METRICS_PATH = os.environ.get(
        "METRICS_PATH",
        str(BASE_DIR / "models" / "model_metrics.json")
    )
    
    # Risk Classification Thresholds
    # Churn probability < RISK_THRESHOLD_LOW: Low Risk
    # RISK_THRESHOLD_LOW <= Churn probability < RISK_THRESHOLD_HIGH: Medium Risk
    # Churn probability >= RISK_THRESHOLD_HIGH: High Risk
    RISK_THRESHOLD_LOW = float(os.environ.get("RISK_THRESHOLD_LOW", "0.35"))
    RISK_THRESHOLD_HIGH = float(os.environ.get("RISK_THRESHOLD_HIGH", "0.65"))
    
    # Server configuration
    PORT = int(os.environ.get("PORT", "3000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    TESTING = False

class TestConfig(Config):
    """Configuration for automated testing."""
    TESTING = True
    DEBUG = False
    DATABASE_PATH = ":memory:"

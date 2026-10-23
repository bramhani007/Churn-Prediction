"""
Prediction service module handling ML model inference, heuristics fallback,
risk categorization, retention plan generation, and dashboard aggregations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import current_app

from database.db import query_db, execute_db
from ml.features import (
    FEATURE_COLUMNS,
    validate_features,
    classify_risk,
    get_retention_actions
)

logger = logging.getLogger(__name__)

# Cached model pipeline in memory
_CACHED_PIPELINE = None
_CACHED_MODEL_PATH = None

def get_loaded_model(model_path: str = None) -> Tuple[Any, bool, str]:
    """
    Load trained scikit-learn pipeline or indicate demo heuristic mode.
    Returns (pipeline, is_demo_mode, model_name).
    """
    global _CACHED_PIPELINE, _CACHED_MODEL_PATH
    
    if not model_path:
        try:
            model_path = current_app.config.get("MODEL_PATH")
        except RuntimeError:
            model_path = os.environ.get("MODEL_PATH", "models/churn_pipeline.joblib")

    target_path = Path(model_path)
    if target_path.exists():
        try:
            if _CACHED_PIPELINE is None or _CACHED_MODEL_PATH != str(target_path):
                _CACHED_PIPELINE = joblib.load(target_path)
                _CACHED_MODEL_PATH = str(target_path)
                logger.info("Loaded trained ML pipeline from %s", target_path)
            
            # Inspect model step name
            step_name = "Trained ML Classifier"
            if hasattr(_CACHED_PIPELINE, "named_steps") and "model" in _CACHED_PIPELINE.named_steps:
                step_name = type(_CACHED_PIPELINE.named_steps["model"]).__name__
            return _CACHED_PIPELINE, False, step_name
        except Exception as e:
            logger.error("Failed to load model file at %s: %s", target_path, e)
    
    # Clearly indicate demo mode fallback
    return None, True, "Demo Heuristic Baseline (Trained Model Absent)"

def predict_customer_churn(
    raw_features: Dict[str, Any],
    customer_id: Optional[str] = None,
    save_to_db: bool = True,
    db_path: str = None,
    model_path: str = None
) -> Dict[str, Any]:
    """
    Validate features, run prediction, classify risk, generate retention actions,
    and persist customer & prediction records in SQLite.
    """
    # 1. Input validation
    is_valid, errors, cleaned = validate_features(raw_features)
    if not is_valid:
        return {
            "success": False,
            "errors": errors
        }

    # Derive product/reorder fields if omitted
    if "total_products" not in cleaned:
        cleaned["total_products"] = int(cleaned["total_orders"] * cleaned["avg_basket_size"])
    if "total_reorders" not in cleaned:
        cleaned["total_reorders"] = int(cleaned["total_products"] * cleaned["reorder_ratio"])
    cleaned["total_reorders"] = min(cleaned["total_reorders"], cleaned["total_products"])

    # 2. Model inference
    pipeline, is_demo, model_name = get_loaded_model(model_path)
    
    if not is_demo and pipeline is not None:
        X_df = pd.DataFrame([cleaned])[FEATURE_COLUMNS]
        if hasattr(pipeline, "predict_proba"):
            prob = float(pipeline.predict_proba(X_df)[0, 1])
        else:
            pred = int(pipeline.predict(X_df)[0])
            prob = float(pred)
    else:
        # Transparent rule-based churn heuristic when model has not yet been trained
        recency = float(cleaned["recency_days"])
        interval = float(cleaned["avg_order_interval"])
        reorder_ratio = float(cleaned["reorder_ratio"])
        
        # Inactivity relative to interval
        recency_ratio = recency / max(interval, 4.0)
        prob = min(max(0.1 + (recency_ratio * 0.28) + (1.0 - reorder_ratio) * 0.22, 0.05), 0.95)

    prob = round(prob, 4)
    pred_class = 1 if prob >= 0.5 else 0

    # Risk thresholds
    low_thresh = 0.35
    high_thresh = 0.65
    try:
        low_thresh = current_app.config.get("RISK_THRESHOLD_LOW", 0.35)
        high_thresh = current_app.config.get("RISK_THRESHOLD_HIGH", 0.65)
    except RuntimeError:
        pass

    risk_category = classify_risk(prob, low_thresh, high_thresh)
    actions = get_retention_actions(risk_category, cleaned, prob)

    resolved_customer_id = customer_id.strip() if customer_id and customer_id.strip() else None

    # 3. Persist to SQLite
    prediction_id = None
    if save_to_db:
        # If no customer ID provided, generate a one-off assessment ID
        if not resolved_customer_id:
            resolved_customer_id = f"GUEST-{np.random.randint(10000, 99999)}"

        # Upsert customer record
        execute_db(
            """
            INSERT INTO customers 
            (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(customer_id) DO UPDATE SET
                total_orders=excluded.total_orders,
                recency_days=excluded.recency_days,
                avg_order_interval=excluded.avg_order_interval,
                avg_basket_size=excluded.avg_basket_size,
                reorder_ratio=excluded.reorder_ratio,
                total_products=excluded.total_products,
                total_reorders=excluded.total_reorders,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                resolved_customer_id,
                int(cleaned["total_orders"]),
                float(cleaned["recency_days"]),
                float(cleaned["avg_order_interval"]),
                float(cleaned["avg_basket_size"]),
                float(cleaned["reorder_ratio"]),
                int(cleaned["total_products"]),
                int(cleaned["total_reorders"])
            ),
            db_path=db_path
        )

        # Record prediction
        prediction_id = execute_db(
            """
            INSERT INTO predictions 
            (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_customer_id,
                prob,
                pred_class,
                risk_category,
                model_name,
                json.dumps(cleaned),
                json.dumps(actions)
            ),
            db_path=db_path
        )

    return {
        "success": True,
        "prediction_id": prediction_id,
        "customer_id": resolved_customer_id,
        "churn_probability": prob,
        "churn_percentage": round(prob * 100, 1),
        "predicted_class": pred_class,
        "risk_category": risk_category,
        "model_name": model_name,
        "is_demo_mode": is_demo,
        "features": cleaned,
        "retention_actions": actions,
        "thresholds": {
            "low": low_thresh,
            "high": high_thresh
        }
    }

def get_dashboard_metrics(db_path: str = None) -> Dict[str, Any]:
    """
    Fetch comprehensive KPI metrics, risk distributions, probability histograms,
    highest risk customers, and recent activity from SQLite.
    """
    # 1. Total customers scored (customers with at least one prediction)
    total_customers_row = query_db(
        "SELECT COUNT(*) as count FROM customers",
        one=True,
        db_path=db_path
    )
    total_customers = total_customers_row["count"] if total_customers_row else 0

    # 2. Total predictions performed
    total_predictions_row = query_db(
        "SELECT COUNT(*) as count FROM predictions",
        one=True,
        db_path=db_path
    )
    total_predictions = total_predictions_row["count"] if total_predictions_row else 0

    # 3. Latest predictions per customer for risk distribution
    risk_counts = query_db(
        """
        SELECT p.risk_category, COUNT(*) as count
        FROM predictions p
        INNER JOIN (
            SELECT customer_id, MAX(id) as max_id
            FROM predictions
            GROUP BY customer_id
        ) latest ON p.id = latest.max_id
        GROUP BY p.risk_category
        """,
        db_path=db_path
    )
    
    risk_dict = {"Low": 0, "Medium": 0, "High": 0}
    for r in risk_counts:
        cat = r["risk_category"]
        if cat in risk_dict:
            risk_dict[cat] = r["count"]

    # 4. Average predicted churn probability
    avg_prob_row = query_db(
        """
        SELECT AVG(p.churn_probability) as avg_prob
        FROM predictions p
        INNER JOIN (
            SELECT customer_id, MAX(id) as max_id
            FROM predictions
            GROUP BY customer_id
        ) latest ON p.id = latest.max_id
        """,
        one=True,
        db_path=db_path
    )
    avg_prob = round(float(avg_prob_row["avg_prob"] or 0.0), 4)

    # 5. Probability distribution histogram buckets for Bar Chart
    # Buckets: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    prob_buckets_row = query_db(
        """
        SELECT 
            SUM(CASE WHEN p.churn_probability < 0.20 THEN 1 ELSE 0 END) as b1,
            SUM(CASE WHEN p.churn_probability >= 0.20 AND p.churn_probability < 0.40 THEN 1 ELSE 0 END) as b2,
            SUM(CASE WHEN p.churn_probability >= 0.40 AND p.churn_probability < 0.60 THEN 1 ELSE 0 END) as b3,
            SUM(CASE WHEN p.churn_probability >= 0.60 AND p.churn_probability < 0.80 THEN 1 ELSE 0 END) as b4,
            SUM(CASE WHEN p.churn_probability >= 0.80 THEN 1 ELSE 0 END) as b5
        FROM predictions p
        INNER JOIN (
            SELECT customer_id, MAX(id) as max_id
            FROM predictions
            GROUP BY customer_id
        ) latest ON p.id = latest.max_id
        """,
        one=True,
        db_path=db_path
    )
    prob_histogram = [
        int(prob_buckets_row["b1"] or 0) if prob_buckets_row else 0,
        int(prob_buckets_row["b2"] or 0) if prob_buckets_row else 0,
        int(prob_buckets_row["b3"] or 0) if prob_buckets_row else 0,
        int(prob_buckets_row["b4"] or 0) if prob_buckets_row else 0,
        int(prob_buckets_row["b5"] or 0) if prob_buckets_row else 0
    ]

    # 6. Highest-risk customer table (top 8 highest churn probability)
    highest_risk_customers = query_db(
        """
        SELECT 
            c.customer_id,
            c.total_orders,
            c.recency_days,
            c.avg_order_interval,
            c.avg_basket_size,
            c.reorder_ratio,
            p.churn_probability,
            p.risk_category,
            p.created_at as predicted_at
        FROM customers c
        INNER JOIN (
            SELECT p1.customer_id, p1.churn_probability, p1.risk_category, p1.created_at
            FROM predictions p1
            INNER JOIN (
                SELECT customer_id, MAX(id) as max_id
                FROM predictions
                GROUP BY customer_id
            ) p2 ON p1.id = p2.max_id
        ) p ON c.customer_id = p.customer_id
        ORDER BY p.churn_probability DESC
        LIMIT 8
        """,
        db_path=db_path
    )

    # 7. Recent prediction activity (last 8)
    recent_activity = query_db(
        """
        SELECT 
            id,
            customer_id,
            churn_probability,
            predicted_class,
            risk_category,
            model_name,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 8
        """,
        db_path=db_path
    )

    # 8. Model Status
    _, is_demo, model_name = get_loaded_model()
    
    # Check for stored metrics
    metrics_file = None
    try:
        metrics_file = current_app.config.get("METRICS_PATH")
    except RuntimeError:
        metrics_file = os.environ.get("METRICS_PATH", "models/model_metrics.json")
    
    metrics_summary = None
    if metrics_file and Path(metrics_file).exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_summary = json.load(f)
        except Exception:
            pass

    return {
        "total_customers": total_customers,
        "high_risk_customers": risk_dict["High"],
        "medium_risk_customers": risk_dict["Medium"],
        "low_risk_customers": risk_dict["Low"],
        "total_predictions": total_predictions,
        "avg_churn_probability": avg_prob,
        "avg_churn_percentage": round(avg_prob * 100, 1),
        "risk_distribution": risk_dict,
        "probability_histogram": prob_histogram,
        "highest_risk_customers": highest_risk_customers,
        "recent_activity": recent_activity,
        "model_status": {
            "model_name": model_name,
            "is_demo_mode": is_demo,
            "metrics": metrics_summary
        }
    }

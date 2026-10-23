"""
Machine Learning Training Pipeline for Customer Churn Prediction.

Trains, evaluates, compares multiple algorithms, selects the best model based on ROC-AUC/F1,
saves the fitted pipeline to joblib, saves metrics to JSON, and records runs in SQLite.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from database.db import execute_db, init_db
from ml.create_demo_data import generate_synthetic_customers
from ml.evaluate import evaluate_classifier
from ml.features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Construct preprocessing and model candidate pipelines.
    Addresses class imbalance with balanced class weighting.
    """
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
        ]),
        "Random Forest Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=-1))
        ])
    }

def train_and_evaluate(
    data_path: str,
    model_output_path: str = "models/churn_pipeline.joblib",
    metrics_output_path: str = "models/model_metrics.json",
    test_size: float = 0.2,
    random_state: int = 42,
    db_path: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Execute full training workflow:
    1. Load customer feature dataset.
    2. Split features and target.
    3. Train each candidate pipeline.
    4. Compute evaluation metrics.
    5. Select optimal model using ROC-AUC (appropriate for customer churn).
    6. Persist pipeline and JSON metrics.
    7. Log training run in SQLite.
    """
    input_file = Path(data_path)
    if not input_file.exists():
        logger.warning("Training file %s not found. Generating synthetic demo dataset...", input_file)
        input_file.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df = generate_synthetic_customers(n_samples=2000, random_state=random_state)
        synthetic_df.to_csv(input_file, index=False)
        df = synthetic_df
    else:
        logger.info("Loading training data from %s...", input_file)
        df = pd.read_csv(input_file)

    # Validate required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Training dataset missing required feature columns: {missing_cols}")
    if "churn" not in df.columns:
        raise ValueError("Training dataset missing target column 'churn'.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["churn"].astype(int).values

    logger.info("Dataset shape: %s | Positive Churn Rate: %.2f%%", X.shape, (y.mean() * 100))

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    candidates = build_candidate_pipelines(random_state=random_state)
    all_metrics = []
    trained_pipelines = {}

    for name, pipeline in candidates.items():
        logger.info("Training candidate: %s...", name)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        metrics = evaluate_classifier(name, pipeline, X_test, y_test)
        logger.info(
            "[%s] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
            name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"]
        )
        all_metrics.append(metrics)

    # Selection Criterion:
    # In retail churn prediction, ROC-AUC measures discrimination across risk thresholds,
    # and F1-score balances missing true churners vs unnecessary retention spend.
    best_candidate = max(all_metrics, key=lambda m: (m["roc_auc"], m["f1_score"]))
    best_name = best_candidate["model_name"]
    best_pipeline = trained_pipelines[best_name]

    logger.info("Selected optimal production model: '%s' (ROC-AUC: %.4f)", best_name, best_candidate["roc_auc"])

    # Mark selection flag in metrics list
    for m in all_metrics:
        m["is_selected"] = (m["model_name"] == best_name)

    # Persist the selected model pipeline
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)
    logger.info("Saved best model pipeline to %s", model_output_path)

    # Persist metrics JSON
    metrics_summary = {
        "selected_model": best_name,
        "selection_metric": "ROC-AUC with F1-Score tie-breaker",
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "models": all_metrics
    }

    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved model evaluation metrics to %s", metrics_output_path)

    # Record in SQLite database
    try:
        init_db(db_path)
        run_id = execute_db(
            """
            INSERT INTO model_training_runs 
            (model_name, best_metric, train_size, test_size, parameters_json, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                best_name,
                f"ROC-AUC: {best_candidate['roc_auc']}",
                len(X_train),
                len(X_test),
                json.dumps(best_candidate),
                f"Trained on {len(df)} customer records."
            ),
            db_path=db_path
        )

        for m in all_metrics:
            execute_db(
                """
                INSERT INTO model_metrics 
                (run_id, model_name, accuracy, precision_score, recall_score, f1_score, roc_auc, confusion_matrix_json, is_selected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    m["model_name"],
                    m["accuracy"],
                    m["precision"],
                    m["recall"],
                    m["f1_score"],
                    m["roc_auc"],
                    json.dumps(m["confusion_matrix"]),
                    1 if m["is_selected"] else 0
                ),
                db_path=db_path
            )
        logger.info("Successfully recorded training run and metrics to SQLite database.")
    except Exception as e:
        logger.warning("Could not record training run to SQLite: %s", e)

    return best_name, metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer churn prediction models")
    parser.add_argument("--input", default="data/processed/customer_training.csv", help="Input processed CSV")
    parser.add_argument("--output", default="models/churn_pipeline.joblib", help="Model output path")
    parser.add_argument("--metrics", default="models/model_metrics.json", help="Metrics JSON output path")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.input,
        model_output_path=args.output,
        metrics_output_path=args.metrics,
        random_state=args.random_state
    )

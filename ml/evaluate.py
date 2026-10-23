"""
Model evaluation module for customer churn prediction.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_classifier(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate a fitted classifier or pipeline on test data.
    Returns standard retail churn metrics.
    """
    y_pred = model.predict(X_test)
    
    # Probabilities for positive class (churn=1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    # Ensure standard 2x2 shape
    if cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tn = int(cm[0, 0]) if cm.shape[0] > 0 else 0
        fp, fn, tp = 0, 0, 0

    roc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5

    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "raw": cm.tolist()
        }
    }
    return metrics

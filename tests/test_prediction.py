"""
Automated unit tests for features validation, risk classification, and prediction logic.
"""

from database.db import init_db
from ml.features import validate_features, classify_risk, get_retention_actions
from services.prediction_service import predict_customer_churn

def test_validate_features_valid():
    """Verify clean features pass validation."""
    valid_data = {
        "total_orders": 12,
        "recency_days": 8.0,
        "avg_order_interval": 9.5,
        "avg_basket_size": 14.0,
        "reorder_ratio": 0.65,
        "total_products": 168,
        "total_reorders": 109
    }
    is_valid, errors, cleaned = validate_features(valid_data)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["total_orders"] == 12
    assert cleaned["recency_days"] == 8.0

def test_validate_features_out_of_range():
    """Verify out-of-range features trigger descriptive errors."""
    invalid_data = {
        "total_orders": -5,
        "recency_days": 250.0,  # Max 180
        "avg_order_interval": 10.0,
        "avg_basket_size": 5.0,
        "reorder_ratio": 1.5,   # Max 1.0
        "total_products": 50,
        "total_reorders": 100   # Cannot exceed total_products
    }
    is_valid, errors, cleaned = validate_features(invalid_data)
    assert is_valid is False
    assert len(errors) >= 3

def test_classify_risk_thresholds():
    """Verify risk classification against configured thresholds."""
    assert classify_risk(0.15) == "Low"
    assert classify_risk(0.34) == "Low"
    assert classify_risk(0.35) == "Medium"
    assert classify_risk(0.64) == "Medium"
    assert classify_risk(0.65) == "High"
    assert classify_risk(0.92) == "High"

def test_retention_actions_generated():
    """Verify contextual retention strategies are produced for each risk category."""
    features = {
        "recency_days": 35.0,
        "avg_order_interval": 12.0,
        "reorder_ratio": 0.2,
        "total_orders": 5,
        "avg_basket_size": 4.0
    }
    high_actions = get_retention_actions("High", features, 0.85)
    assert len(high_actions) > 0
    assert any("Urgent" in a or "Win-Back" in a for a in high_actions)

    low_actions = get_retention_actions("Low", features, 0.15)
    assert len(low_actions) > 0
    assert any("Loyalty" in a or "Recipe" in a for a in low_actions)

def test_prediction_service_execution(tmp_path):
    """Verify end-to-end prediction execution with SQLite persistence."""
    test_db = str(tmp_path / "pred_test.db")
    init_db(test_db)
    raw_input = {
        "total_orders": 8,
        "recency_days": 18.0,
        "avg_order_interval": 10.0,
        "avg_basket_size": 9.0,
        "reorder_ratio": 0.45
    }
    res = predict_customer_churn(
        raw_features=raw_input,
        customer_id="TEST-USER-99",
        save_to_db=True,
        db_path=test_db
    )

    assert res["success"] is True
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_category"] in ("Low", "Medium", "High")
    assert res["customer_id"] == "TEST-USER-99"
    assert res["prediction_id"] is not None

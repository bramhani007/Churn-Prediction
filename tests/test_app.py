"""
Automated integration tests for Flask view routes, API endpoints, and error handling.
"""

import pytest
from app import create_app
from config import TestConfig
from database.db import init_db
from ml.create_demo_data import seed_database_with_demo_data

@pytest.fixture
def app_instance(tmp_path):
    test_db = str(tmp_path / "integration_test.db")
    
    class RuntimeTestConfig(TestConfig):
        DATABASE_PATH = test_db
        SECRET_KEY = "test-secret-key"

    app = create_app(RuntimeTestConfig)
    with app.app_context():
        init_db(test_db)
        seed_database_with_demo_data(db_path=test_db, n_customers=25)
    return app

@pytest.fixture
def client(app_instance):
    return app_instance.test_client()

def test_dashboard_route(client):
    """GET / should render 200 with dashboard metrics."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Customer Churn Intelligence" in response.data
    assert b"Total Customers" in response.data

def test_customers_route(client):
    """GET /customers should list customers and handle filters."""
    response = client.get("/customers")
    assert response.status_code == 200
    assert b"Customer Management" in response.data

    # Filter by risk
    risk_res = client.get("/customers?risk=high")
    assert risk_res.status_code == 200

def test_predict_form_get(client):
    """GET /predict should display the prediction form."""
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Customer Behavioral Features" in response.data
    assert b"Calculate Churn Probability" in response.data

def test_predict_form_post_success(client):
    """POST /predict with valid features scores and persists prediction."""
    payload = {
        "customer_id": "TEST-SUBMIT-1",
        "total_orders": "15",
        "recency_days": "10.5",
        "avg_order_interval": "8.0",
        "avg_basket_size": "12.0",
        "reorder_ratio": "0.60",
        "total_products": "180",
        "total_reorders": "108"
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert b"Prediction Evaluation Result" in response.data
    assert b"TEST-SUBMIT-1" in response.data

def test_predict_form_post_invalid(client):
    """POST /predict with out-of-range features displays error flash message."""
    payload = {
        "customer_id": "INVALID-CUST",
        "total_orders": "-1",
        "recency_days": "999",
        "avg_order_interval": "10",
        "avg_basket_size": "5",
        "reorder_ratio": "2.5"
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert b"out of allowable range" in response.data or b"danger" in response.data

def test_model_performance_route(client):
    """GET /model-performance should render benchmark page."""
    response = client.get("/model-performance")
    assert response.status_code == 200
    assert b"Machine Learning Benchmark" in response.data

def test_health_check_endpoint(client):
    """GET /health should return 200 JSON with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ("healthy", "degraded")
    assert data["database"] == "healthy"
    assert "model_status" in data

def test_api_metrics(client):
    """GET /api/metrics should return aggregated KPIs."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "total_customers" in data["data"]
    assert "risk_distribution" in data["data"]

def test_api_customers(client):
    """GET /api/customers should return paginated JSON."""
    response = client.get("/api/customers?per_page=5")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["data"]["customers"]) <= 5

def test_api_predictions_get_and_post(client):
    """Verify /api/predictions REST interface."""
    # GET
    get_res = client.get("/api/predictions?limit=5")
    assert get_res.status_code == 200

    # POST
    payload = {
        "customer_id": "API-CUST-88",
        "features": {
            "total_orders": 20,
            "recency_days": 5.0,
            "avg_order_interval": 7.0,
            "avg_basket_size": 15.0,
            "reorder_ratio": 0.75,
            "total_products": 300,
            "total_reorders": 225
        }
    }
    post_res = client.post("/api/predictions", json=payload)
    assert post_res.status_code == 201
    res_data = post_res.get_json()
    assert res_data["status"] == "success"
    assert res_data["data"]["customer_id"] == "API-CUST-88"

"""
Automated unit tests for SQLite database initialization, schemas, and queries.
"""

import sqlite3
import pytest
from database.db import init_db, query_db, execute_db

@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_churn.db")
    init_db(db_file)
    return db_file

def test_database_initialization(temp_db):
    """Verify that all expected tables and indexes are created."""
    conn = sqlite3.connect(temp_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()

    assert "customers" in tables
    assert "predictions" in tables
    assert "model_training_runs" in tables
    assert "model_metrics" in tables

def test_customer_insert_and_query(temp_db):
    """Verify inserting and querying customer profiles."""
    execute_db(
        """
        INSERT INTO customers 
        (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("TEST-CUST-1", 10, 5.0, 7.5, 12.0, 0.65, 120, 78),
        db_path=temp_db
    )

    cust = query_db("SELECT * FROM customers WHERE customer_id = ?", ("TEST-CUST-1",), one=True, db_path=temp_db)
    assert cust is not None
    assert cust["customer_id"] == "TEST-CUST-1"
    assert cust["total_orders"] == 10
    assert cust["recency_days"] == 5.0
    assert cust["reorder_ratio"] == 0.65

def test_prediction_foreign_key(temp_db):
    """Verify prediction record links correctly to customer."""
    execute_db(
        """
        INSERT INTO customers 
        (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("TEST-CUST-2", 8, 20.0, 10.0, 6.0, 0.40, 48, 19),
        db_path=temp_db
    )

    pred_id = execute_db(
        """
        INSERT INTO predictions 
        (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("TEST-CUST-2", 0.72, 1, "High", "Test Model", "{}", "[]"),
        db_path=temp_db
    )

    assert pred_id is not None
    preds = query_db("SELECT * FROM predictions WHERE customer_id = ?", ("TEST-CUST-2",), db_path=temp_db)
    assert len(preds) == 1
    assert preds[0]["churn_probability"] == 0.72
    assert preds[0]["risk_category"] == "High"

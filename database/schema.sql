-- SQLite Database Schema for Customer Churn Prediction System

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    total_orders INTEGER NOT NULL,
    recency_days REAL NOT NULL,
    avg_order_interval REAL NOT NULL,
    avg_basket_size REAL NOT NULL,
    reorder_ratio REAL NOT NULL,
    total_products INTEGER DEFAULT 0,
    total_reorders INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    predicted_class INTEGER NOT NULL,
    risk_category TEXT NOT NULL CHECK(risk_category IN ('Low', 'Medium', 'High')),
    model_name TEXT NOT NULL,
    features_json TEXT NOT NULL,
    retention_actions TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT NOT NULL,
    best_metric TEXT NOT NULL,
    train_size INTEGER NOT NULL,
    test_size INTEGER NOT NULL,
    parameters_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    model_name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    roc_auc REAL NOT NULL,
    confusion_matrix_json TEXT NOT NULL,
    is_selected BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES model_training_runs(id) ON DELETE SET NULL
);

-- Indexes for fast lookup and dashboard metrics
CREATE INDEX IF NOT EXISTS idx_customers_recency ON customers(recency_days);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_category);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_selected ON model_metrics(is_selected);

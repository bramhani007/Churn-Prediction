# System Architecture & Technical Specifications

**Machine Learning-Based Customer Churn Prediction System for an Online Grocery and Food Retail E-commerce Platform**

## 1. High-Level Architecture Overview

The system is designed as a modular, full-stack predictive analytics platform built with Python 3.11, Flask, SQLite, and scikit-learn.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Web Client (Browser / Mobile)                        │
│   HTML5 / CSS3 (Dark Navy & Emerald) / Bootstrap 5 / Chart.js          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Flask Backend Web Service                         │
│                                                                        │
│   ┌────────────────────┐   ┌───────────────────┐  ┌────────────────┐  │
│   │    View Routes     │   │   REST API Layer  │  │ Error Handlers │  │
│   │  / /customers etc. │   │ /api/metrics etc. │  │  404, 500, etc.│  │
│   └─────────┬──────────┘   └─────────┬─────────┘  └────────────────┘  │
│             │                        │                                 │
│             ▼                        ▼                                 │
│   ┌────────────────────────────────────────────┐                       │
│   │               Service Layer                │                       │
│   │    customer_service   prediction_service   │                       │
│   └──────────────────────┬─────────────────────┘                       │
└──────────────────────────┼─────────────────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────────────────────┐
│      SQLite Database      │ │        Machine Learning Engine           │
│   - customers             │ │   - Pipeline (Imputer, Scaler, Model)    │
│   - predictions           │ │   - Features Validation & Guardrails     │
│   - model_training_runs   │ │   - Baseline / Logistic / RF / HGB       │
│   - model_metrics         │ │   - Joblib persistence & JSON metrics    │
└───────────────────────────┘ └──────────────────────────────────────────┘
```

## 2. Component Design

### 2.1 Backend (Flask 3.x)
- **`app.py`**: Factory application initializing database tables, custom Jinja filters (e.g. `from_json`), HTML views, REST endpoints, and system health checks.
- **`config.py`**: Centralized configuration management using environment variables (`SECRET_KEY`, `DATABASE_PATH`, `MODEL_PATH`, `METRICS_PATH`, `RISK_THRESHOLD_LOW`, `RISK_THRESHOLD_HIGH`).
- **`services/`**:
  - `customer_service.py`: Encapsulates customer queries, pagination, search, risk filtering, and history.
  - `prediction_service.py`: Encapsulates model inference, transparent heuristic fallback, risk categorization, retention strategies, and dashboard metric calculations.

### 2.2 Database (SQLite 3)
- Built-in `sqlite3` with row factory enabled for dict access.
- Foreign keys enforced with `PRAGMA foreign_keys = ON;`.
- Parameterized SQL queries everywhere to guarantee SQL injection immunity.
- Automatic creation and seeding on boot if empty.

### 2.3 Machine Learning Pipeline (`scikit-learn` & `joblib`)
- **Features Pipeline:**
  1. `SimpleImputer(strategy='median')`: Handles missing behavioral values.
  2. `StandardScaler()`: Normalizes feature distributions.
  3. Classifier algorithm with balanced class weighting.
- **Evaluated Algorithms:**
  - Dummy Classifier (Stratified baseline)
  - Logistic Regression (Balanced L2 regularized)
  - Random Forest Classifier (Ensemble of decision trees)
  - HistGradientBoostingClassifier (Non-linear gradient boosted trees)
- **Selection Metric:** ROC-AUC with F1-Score tie-breaker to prioritize detecting genuine churners without excessive false alarms.

## 3. Render Production Deployment Topology
- **Server:** Gunicorn WSGI server (`gunicorn app:app`).
- **Health Check:** `/health` endpoint for Render zero-downtime health probing.
- **Persistence:** Render web services run on ephemeral disks by default. For multi-session SQLite persistence, attach a Render persistent disk and point `DATABASE_PATH` to `/var/data/churn.db`.

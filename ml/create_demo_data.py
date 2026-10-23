"""
Generates synthetic customer-level grocery purchasing data and seeds the SQLite database.
Enables full local development, evaluation, and smoke testing when raw Kaggle files are not yet downloaded.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from database.db import init_db, execute_db, query_db
from ml.features import FEATURE_COLUMNS, classify_risk, get_retention_actions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_synthetic_customers(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic customer purchasing behavioral features
    mimicking the Instacart grocery distribution.
    Features:
      - total_orders: 1 to 100
      - recency_days: 0 to 60
      - avg_order_interval: 3 to 35
      - avg_basket_size: 2 to 35
      - reorder_ratio: 0.1 to 0.95
      - total_products: calculated
      - total_reorders: calculated
      - churn: boolean target
    """
    rng = np.random.default_rng(random_state)

    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    
    # 3 cohorts: Loyal regulars, Occasional/declining, Inactive churners
    cohort_types = rng.choice(["loyal", "declining", "churned"], size=n_samples, p=[0.45, 0.30, 0.25])
    
    total_orders = []
    recency_days = []
    avg_intervals = []
    avg_baskets = []
    reorder_ratios = []
    churn_labels = []

    for cohort in cohort_types:
        if cohort == "loyal":
            orders = rng.integers(12, 85)
            recency = rng.uniform(1.0, 10.0)
            interval = rng.uniform(4.0, 12.0)
            basket = rng.uniform(8.0, 24.0)
            ratio = rng.uniform(0.55, 0.92)
            churn = 1 if rng.random() < 0.08 else 0
        elif cohort == "declining":
            orders = rng.integers(4, 25)
            recency = rng.uniform(14.0, 29.0)
            interval = rng.uniform(10.0, 22.0)
            basket = rng.uniform(4.0, 16.0)
            ratio = rng.uniform(0.30, 0.65)
            churn = 1 if rng.random() < 0.45 else 0
        else:  # churned
            orders = rng.integers(1, 10)
            recency = rng.uniform(30.0, 58.0)
            interval = rng.uniform(18.0, 35.0)
            basket = rng.uniform(2.0, 10.0)
            ratio = rng.uniform(0.05, 0.40)
            churn = 1 if rng.random() < 0.88 else 0

        total_orders.append(orders)
        recency_days.append(round(recency, 1))
        avg_intervals.append(round(interval, 1))
        avg_baskets.append(round(basket, 1))
        reorder_ratios.append(round(ratio, 4))
        churn_labels.append(churn)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "total_orders": total_orders,
        "recency_days": recency_days,
        "avg_order_interval": avg_intervals,
        "avg_basket_size": avg_baskets,
        "reorder_ratio": reorder_ratios,
        "churn": churn_labels
    })

    # Derive products & reorders consistently
    df["total_products"] = (df["total_orders"] * df["avg_basket_size"]).round().astype(int).clip(lower=1)
    df["total_reorders"] = (df["total_products"] * df["reorder_ratio"]).round().astype(int)
    # Ensure constraint holds
    df["total_reorders"] = np.minimum(df["total_reorders"], df["total_products"])

    return df

def seed_database_with_demo_data(db_path: str = None, n_customers: int = 150):
    """
    Initialize SQLite and populate initial customer profiles and prediction history.
    """
    init_db(db_path)
    
    # Check if customers already populated
    existing = query_db("SELECT COUNT(*) as cnt FROM customers", db_path=db_path, one=True)
    if existing and existing["cnt"] > 0:
        logger.info("Database already seeded with %d customers.", existing["cnt"])
        return

    logger.info("Seeding database with %d demo customers and predictions...", n_customers)
    df = generate_synthetic_customers(n_samples=n_customers, random_state=123)

    for _, row in df.iterrows():
        execute_db(
            """
            INSERT OR REPLACE INTO customers 
            (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["customer_id"],
                int(row["total_orders"]),
                float(row["recency_days"]),
                float(row["avg_order_interval"]),
                float(row["avg_basket_size"]),
                float(row["reorder_ratio"]),
                int(row["total_products"]),
                int(row["total_reorders"])
            ),
            db_path=db_path
        )

        # Generate a baseline initial prediction for rich dashboard insights
        # Probability based on recency and order pattern
        recency_penalty = min(row["recency_days"] / 45.0, 1.0) * 0.6
        interval_factor = min(row["avg_order_interval"] / 30.0, 1.0) * 0.2
        reorder_bonus = (1.0 - row["reorder_ratio"]) * 0.2
        prob = min(max(round(recency_penalty + interval_factor + reorder_bonus, 4), 0.02), 0.98)
        
        risk = classify_risk(prob)
        feat_dict = {col: row[col] for col in FEATURE_COLUMNS}
        actions = get_retention_actions(risk, feat_dict, prob)
        
        execute_db(
            """
            INSERT INTO predictions 
            (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["customer_id"],
                prob,
                1 if prob >= 0.5 else 0,
                risk,
                "Demo Baseline Classifier",
                json.dumps(feat_dict),
                json.dumps(actions)
            ),
            db_path=db_path
        )

    logger.info("Demo data seeded successfully into database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for grocery churn")
    parser.add_argument("--samples", type=int, default=1500, help="Number of synthetic customer records")
    parser.add_argument("--output", default="data/processed/customer_training.csv", help="Processed CSV path")
    parser.add_argument("--seed-db", action="store_true", help="Also seed the SQLite database")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_customers(n_samples=args.samples)
    df.to_csv(args.output, index=False)
    logger.info("Generated %d synthetic customer records at %s", len(df), args.output)

    if args.seed_db:
        seed_database_with_demo_data()

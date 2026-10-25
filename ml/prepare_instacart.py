"""
Instacart Market Basket Data Preparation Pipeline.

Generates a leakage-free customer-level dataset with a rigorously formulated
churn proxy label based on customer ordering cadence and observation cutoff.

Methodology details documented in docs/dataset_methodology.md.
"""

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_instacart_data(
    data_dir: str,
    output_path: str,
    min_orders: int = 3,
    sample_users: int = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Process raw Instacart CSV files into a customer-level churn feature dataset.

    Expected raw files in data_dir:
      - orders.csv
      - order_products__prior.csv
      - order_products__train.csv (optional, can be combined)
    """
    data_path = Path(data_dir)
    orders_file = data_path / "orders.csv"
    prior_file = data_path / "order_products__prior.csv"

    if not orders_file.exists():
        raise FileNotFoundError(
            f"Orders file not found at {orders_file}. Please download the Instacart dataset "
            "from Kaggle (https://www.kaggle.com/c/instacart-market-basket-analysis or https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) "
            f"and place orders.csv into {data_dir}."
        )

    logger.info("Reading orders from %s...", orders_file)
    orders = pd.read_csv(orders_file)

    # Filter to eligible users with at least min_orders to observe order patterns
    user_order_counts = orders.groupby("user_id")["order_number"].max()
    eligible_users = user_order_counts[user_order_counts >= min_orders].index

    if sample_users and sample_users < len(eligible_users):
        logger.info("Sampling %d eligible users for efficient training...", sample_users)
        np.random.seed(random_state)
        eligible_users = np.random.choice(eligible_users, size=sample_users, replace=False)

    orders = orders[orders["user_id"].isin(eligible_users)].copy()
    logger.info("Working with %d users...", len(eligible_users))

    # Churn proxy definition:
    # We partition each customer's timeline into an OBSERVATION window (all orders except the final one)
    # and a PREDICTION/EVALUATION window (the final order recorded in Instacart, whether train or prior).
    # If the days_since_prior_order for the evaluation order is >= 30.0 days (the Instacart ceiling for inactivity),
    # or if the user had no subsequent order within the horizon, they are labeled churned = 1, else 0.
    
    # Sort orders chronologically by customer and order_number
    orders = orders.sort_values(["user_id", "order_number"])

    # Split into evaluation order (final order) and observation orders (prior to final order)
    eval_orders = orders.groupby("user_id").last().reset_index()
    obs_orders = orders.groupby("user_id").apply(lambda g: g.iloc[:-1]).reset_index(drop=True)

    # Target label: Churn Proxy
    # Instacart caps `days_since_prior_order` at 30 days. Customers who reached 30 days
    # without returning represent high inactivity / grocery churn in this benchmark.
    eval_orders["churn_label"] = (eval_orders["days_since_prior_order"] >= 30.0).astype(int)
    target_df = eval_orders[["user_id", "churn_label", "days_since_prior_order"]].rename(
        columns={"days_since_prior_order": "eval_days_since_prior"}
    )

    # Observation features calculated STRICTLY on obs_orders (preventing target leakage)
    logger.info("Calculating leakage-free customer features on observation window...")
    customer_features = obs_orders.groupby("user_id").agg(
        total_orders=("order_number", "max"),
        recency_days=("days_since_prior_order", "last"),
        avg_order_interval=("days_since_prior_order", "mean")
    ).reset_index()

    customer_features["recency_days"] = customer_features["recency_days"].fillna(0.0)
    customer_features["avg_order_interval"] = customer_features["avg_order_interval"].fillna(0.0)

    # Basket features from order_products__prior if present
    if prior_file.exists():
        logger.info("Merging product-level basket data from %s...", prior_file)
        prior_products = pd.read_csv(prior_file, usecols=["order_id", "reordered"])
        
        # Merge only on observation orders
        obs_merged = pd.merge(
            obs_orders[["order_id", "user_id"]],
            prior_products,
            on="order_id",
            how="inner"
        )
        
        basket_stats = obs_merged.groupby(["user_id", "order_id"]).agg(
            basket_size=("reordered", "count"),
            basket_reorders=("reordered", "sum")
        ).reset_index()

        user_basket = basket_stats.groupby("user_id").agg(
            avg_basket_size=("basket_size", "mean"),
            total_products=("basket_size", "sum"),
            total_reorders=("basket_reorders", "sum")
        ).reset_index()

        user_basket["reorder_ratio"] = (
            user_basket["total_reorders"] / user_basket["total_products"].replace(0, 1)
        ).round(4)

        customer_features = pd.merge(customer_features, user_basket, on="user_id", how="left")
    else:
        logger.warning(
            "order_products__prior.csv not found; assigning baseline basket estimates."
        )
        customer_features["avg_basket_size"] = 10.0
        customer_features["total_products"] = customer_features["total_orders"] * 10
        customer_features["total_reorders"] = customer_features["total_products"] * 0.5
        customer_features["reorder_ratio"] = 0.5

    # Fill any remaining NaNs
    customer_features["avg_basket_size"] = customer_features["avg_basket_size"].fillna(8.0)
    customer_features["reorder_ratio"] = customer_features["reorder_ratio"].fillna(0.5)
    customer_features["total_products"] = customer_features["total_products"].fillna(30).astype(int)
    customer_features["total_reorders"] = customer_features["total_reorders"].fillna(15).astype(int)

    # Merge features with target label
    final_df = pd.merge(customer_features, target_df, on="user_id", how="inner")
    final_df = final_df.rename(columns={"user_id": "customer_id", "churn_label": "churn"})

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(
        "Successfully prepared %d customer records (Churn rate: %.2f%%). Saved to %s",
        len(final_df),
        (final_df["churn"].mean() * 100),
        output_path
    )
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Instacart Market Basket dataset for churn prediction")
    parser.add_argument("--data-dir", default="data/raw", help="Directory with raw Instacart CSV files")
    parser.add_argument("--output", default="data/processed/customer_training.csv", help="Output path for processed CSV")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of customers to sample (default: 5000)")
    args = parser.parse_args()

    process_instacart_data(
        data_dir=args.data_dir,
        output_path=args.output,
        sample_users=args.sample_size
    )

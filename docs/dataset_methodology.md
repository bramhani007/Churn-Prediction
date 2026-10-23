# Instacart Market Basket Dataset Methodology & Churn Proxy Formulation

## 1. Instacart Dataset Overview

The Instacart Market Basket dataset represents customer order histories across 3+ million grocery orders from over 200,000 anonymized Instacart users.

**Key constraints inherent to the Instacart dataset:**
1. **No Explicit Churn Flag:** The dataset records orders and products, not explicit cancellations or unsubscribes.
2. **No Real Calendar Dates:** Orders are sequenced via `order_number` with relative time represented by `days_since_prior_order` (capped at 30 days) and `order_dow` (day of week 0–6).
3. **No Monetary Pricing:** Instacart does not provide product price values or revenue metrics.

## 2. Leakage-Free Churn Proxy Formulation

To predict customer churn without data leakage, we formulate a time-sequenced observation window and evaluation cutoff:

### 2.1 The Observation Cutoff ($T_{cutoff}$)
For each eligible customer with $N \ge 3$ recorded orders:
- **Observation Window (Orders $1$ through $N-1$):** Used strictly to compute customer purchasing behavior features.
- **Evaluation Order (Order $N$):** Used strictly to assess the target outcome.

### 2.2 The Churn Proxy Label ($Y \in \{0, 1\}$)
In online grocery delivery, regular customers purchase on a weekly or bi-weekly cadence (mean 7–14 days). The Instacart dataset sets a ceiling of 30 days for `days_since_prior_order`.

- If the evaluation order's `days_since_prior_order` $\ge 30.0$ days (or customer has no subsequent order within the inactivity horizon), the customer has experienced significant disengagement/inactivity and is labeled **Churn = 1**.
- If the customer returned within standard grocery cadence ($< 30.0$ days), the customer is labeled **Retained = 0**.

### 2.3 Prevention of Feature Leakage
Features are derived **exclusively** from orders prior to the evaluation order:
- `total_orders`: Number of prior orders up to $N-1$.
- `recency_days`: Days since prior order at the observation boundary.
- `avg_order_interval`: Mean days elapsed between prior consecutive orders.
- `avg_basket_size`: Mean number of products per prior order.
- `reorder_ratio`: Total reordered items divided by total items across prior orders.
- `total_products`: Cumulative products purchased in prior orders.
- `total_reorders`: Cumulative reorders in prior orders.

**Strict constraint:** No monetary Average Order Value (AOV) is claimed or fabricated.

## 3. Training & Evaluation Protocol
- **Temporal Train/Test Split:** Preserves temporal ordering to validate generalization to future customers.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Selection Metric:** ROC-AUC is prioritized over accuracy because churn is imbalanced, and commercial retention teams require risk threshold tuning to optimize marketing budget allocations.

"""
Feature definitions, validation rules, and business logic for customer churn prediction.
Grocery e-commerce churn features strictly avoid target leakage and adhere to
Instacart Market Basket data constraints (no fabricated prices or calendar timestamps).
"""

from typing import Dict, Any, Tuple, List

# Consistent feature columns expected by the trained ML models
FEATURE_COLUMNS: List[str] = [
    "total_orders",
    "recency_days",
    "avg_order_interval",
    "avg_basket_size",
    "reorder_ratio",
    "total_products",
    "total_reorders"
]

# Validation rules for each customer feature
FEATURE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "total_orders": {
        "min": 1,
        "max": 150,
        "type": int,
        "description": "Total completed orders by customer"
    },
    "recency_days": {
        "min": 0.0,
        "max": 180.0,
        "type": float,
        "description": "Days since last recorded order"
    },
    "avg_order_interval": {
        "min": 0.0,
        "max": 60.0,
        "type": float,
        "description": "Average days elapsed between consecutive orders"
    },
    "avg_basket_size": {
        "min": 1.0,
        "max": 100.0,
        "type": float,
        "description": "Average number of grocery products per basket"
    },
    "reorder_ratio": {
        "min": 0.0,
        "max": 1.0,
        "type": float,
        "description": "Proportion of items previously ordered (0.0 to 1.0)"
    },
    "total_products": {
        "min": 1,
        "max": 5000,
        "type": int,
        "description": "Cumulative grocery products purchased"
    },
    "total_reorders": {
        "min": 0,
        "max": 5000,
        "type": int,
        "description": "Cumulative reordered grocery products"
    }
}

def validate_features(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate input customer features on backend.
    Automatically derives total_products and total_reorders if omitted.
    Returns (is_valid, list_of_error_messages, cleaned_data).
    """
    errors = []
    cleaned: Dict[str, Any] = {}

    # Copy input data
    processed_data = dict(data)

    # Core required features
    core_features = ["total_orders", "recency_days", "avg_order_interval", "avg_basket_size", "reorder_ratio"]
    for col in core_features:
        raw_val = processed_data.get(col)
        if raw_val is None or str(raw_val).strip() == "":
            errors.append(f"Missing required feature: '{col}'.")
            continue
        constraint = FEATURE_CONSTRAINTS[col]
        try:
            val = constraint["type"](raw_val)
        except (ValueError, TypeError):
            errors.append(f"'{col}' must be a valid {constraint['type'].__name__}.")
            continue
        if val < constraint["min"] or val > constraint["max"]:
            errors.append(
                f"'{col}' value {val} is out of allowable range [{constraint['min']}, {constraint['max']}]."
            )
            continue
        cleaned[col] = val

    if errors:
        return False, errors, cleaned

    # Auto-derive total_products if not supplied or empty
    raw_products = processed_data.get("total_products")
    if raw_products is None or str(raw_products).strip() == "":
        cleaned["total_products"] = max(1, int(round(cleaned["total_orders"] * cleaned["avg_basket_size"])))
    else:
        try:
            val = int(raw_products)
            if val < 1 or val > 5000:
                errors.append(f"'total_products' value {val} is out of allowable range [1, 5000].")
            else:
                cleaned["total_products"] = val
        except (ValueError, TypeError):
            errors.append("'total_products' must be a valid integer.")

    # Auto-derive total_reorders if not supplied or empty
    raw_reorders = processed_data.get("total_reorders")
    if raw_reorders is None or str(raw_reorders).strip() == "":
        prod_count = cleaned.get("total_products", 1)
        cleaned["total_reorders"] = int(round(prod_count * cleaned["reorder_ratio"]))
    else:
        try:
            val = int(raw_reorders)
            if val < 0 or val > 5000:
                errors.append(f"'total_reorders' value {val} is out of allowable range [0, 5000].")
            else:
                cleaned["total_reorders"] = val
        except (ValueError, TypeError):
            errors.append("'total_reorders' must be a valid integer.")

    # Cross-feature logical validation
    if "total_reorders" in cleaned and "total_products" in cleaned:
        cleaned["total_reorders"] = min(cleaned["total_reorders"], cleaned["total_products"])
        if cleaned["total_reorders"] > cleaned["total_products"]:
            errors.append("Total reorders cannot exceed total products purchased.")

    return len(errors) == 0, errors, cleaned

def classify_risk(probability: float, low_threshold: float = 0.35, high_threshold: float = 0.65) -> str:
    """
    Classify churn probability into Low, Medium, or High risk category.
    """
    if probability >= high_threshold:
        return "High"
    elif probability >= low_threshold:
        return "Medium"
    return "Low"

def get_retention_actions(risk_category: str, features: Dict[str, Any], probability: float) -> List[str]:
    """
    Generate actionable, contextual retail retention strategies
    based on customer purchasing behavioral signals.
    """
    actions = []
    recency = float(features.get("recency_days", 0))
    reorder_ratio = float(features.get("reorder_ratio", 0))
    avg_basket = float(features.get("avg_basket_size", 0))
    total_orders = int(features.get("total_orders", 0))

    if risk_category == "High":
        actions.append("⚡ Urgent Win-Back: Send personalized push notification with 20% discount on customer's top 5 favorite grocery essentials.")
        if recency > 30:
            actions.append("📅 Inactivity Intervention: Trigger 'We miss you' replenishment email highlighting fresh pantry items restocked.")
        if reorder_ratio < 0.3:
            actions.append("🎯 Assortment Exploration: Offer complimentary sample item on next order to discover recurring grocery favorites.")
        if total_orders >= 10:
            actions.append("💎 VIP Service Recovery: Assign dedicated support outreach or free delivery coupon for 3 consecutive orders.")
    elif risk_category == "Medium":
        actions.append("🔔 Routine Re-engagement: Schedule weekly basket-building reminder aligned with customer's typical order interval.")
        if reorder_ratio > 0.6:
            actions.append("🛒 Smart Reorder Prompt: Pre-populate digital shopping cart with previously ordered dairy, produce, and pantry staples.")
        if avg_basket < 8:
            actions.append("📦 Basket Builder Incentive: Free delivery perk when order basket exceeds $45 / 10 items.")
        actions.append("⭐ Loyalty Milestone: Reward with double loyalty points on next weekend grocery checkout.")
    else:  # Low risk
        actions.append("💚 Loyalty Retention: Continue regular seasonal recipe inspirations and new organic/local product highlights.")
        if total_orders > 15:
            actions.append("🌟 Early Access Privilege: Grant preview access to holiday promotions and exclusive farmer's market bundles.")
        actions.append("💬 Feedback Pulse: Send optional 1-click survey regarding delivery punctuality and produce freshness.")

    return actions

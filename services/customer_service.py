"""
Customer service module for database interactions and customer management.
"""

from typing import Dict, Any, List, Optional
from database.db import query_db, execute_db

def get_customers(
    page: int = 1,
    per_page: int = 15,
    search: str = "",
    risk_filter: str = "",
    sort_by: str = "recency_days",
    order: str = "desc",
    db_path: str = None
) -> Dict[str, Any]:
    """
    Fetch paginated, filtered, and sorted customer records with their latest prediction.
    """
    offset = (max(page, 1) - 1) * per_page
    
    # Allowed sort columns to prevent SQL injection
    allowed_sorts = {
        "customer_id": "c.customer_id",
        "total_orders": "c.total_orders",
        "recency_days": "c.recency_days",
        "avg_order_interval": "c.avg_order_interval",
        "avg_basket_size": "c.avg_basket_size",
        "reorder_ratio": "c.reorder_ratio",
        "churn_probability": "p.churn_probability",
        "created_at": "c.created_at"
    }
    sort_col = allowed_sorts.get(sort_by, "c.recency_days")
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"

    where_clauses = []
    params = []

    if search.strip():
        where_clauses.append("c.customer_id LIKE ?")
        params.append(f"%{search.strip()}%")

    if risk_filter.strip() and risk_filter.lower() != "all":
        where_clauses.append("p.risk_category = ?")
        params.append(risk_filter.capitalize())

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Count query
    count_query = f"""
        SELECT COUNT(DISTINCT c.customer_id) as total_count
        FROM customers c
        LEFT JOIN (
            SELECT p1.customer_id, p1.risk_category, p1.churn_probability
            FROM predictions p1
            INNER JOIN (
                SELECT customer_id, MAX(id) as max_id
                FROM predictions
                GROUP BY customer_id
            ) p2 ON p1.id = p2.max_id
        ) p ON c.customer_id = p.customer_id
        {where_sql}
    """
    count_res = query_db(count_query, params, one=True, db_path=db_path)
    total_count = count_res["total_count"] if count_res else 0
    total_pages = max(1, (total_count + per_page - 1) // per_page)

    # Main query
    data_query = f"""
        SELECT 
            c.customer_id,
            c.total_orders,
            c.recency_days,
            c.avg_order_interval,
            c.avg_basket_size,
            c.reorder_ratio,
            c.total_products,
            c.total_reorders,
            c.created_at,
            p.churn_probability,
            p.risk_category,
            p.created_at as last_predicted_at
        FROM customers c
        LEFT JOIN (
            SELECT p1.customer_id, p1.risk_category, p1.churn_probability, p1.created_at
            FROM predictions p1
            INNER JOIN (
                SELECT customer_id, MAX(id) as max_id
                FROM predictions
                GROUP BY customer_id
            ) p2 ON p1.id = p2.max_id
        ) p ON c.customer_id = p.customer_id
        {where_sql}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """
    query_params = list(params) + [per_page, offset]
    rows = query_db(data_query, query_params, db_path=db_path)

    return {
        "customers": rows,
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "search": search,
        "risk_filter": risk_filter,
        "sort_by": sort_by,
        "order": order
    }

def get_customer_detail(customer_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive customer profile and prediction history.
    """
    customer = query_db(
        "SELECT * FROM customers WHERE customer_id = ?",
        (customer_id,),
        one=True,
        db_path=db_path
    )
    if not customer:
        return None

    predictions = query_db(
        """
        SELECT id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions, created_at
        FROM predictions 
        WHERE customer_id = ?
        ORDER BY id DESC
        """,
        (customer_id,),
        db_path=db_path
    )

    return {
        "customer": customer,
        "predictions": predictions,
        "latest_prediction": predictions[0] if predictions else None
    }

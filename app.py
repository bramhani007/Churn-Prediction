"""
Flask Application Entry Point:
Machine Learning-Based Customer Churn Prediction System for an Online Grocery and Food Retail E-commerce Platform.
"""

import json
import logging
import os
from pathlib import Path
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash
)

from config import Config
from database.db import close_db, init_db, query_db
from ml.create_demo_data import seed_database_with_demo_data
from ml.features import FEATURE_COLUMNS, FEATURE_CONSTRAINTS
from services.customer_service import get_customers, get_customer_detail
from services.prediction_service import (
    get_dashboard_metrics,
    predict_customer_churn,
    get_loaded_model
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_app")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Teardown database connection
    app.teardown_appcontext(close_db)

    # Custom Jinja template filters
    @app.template_filter('from_json')
    def filter_from_json(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return [val]
        return val or []

    # Initialize database tables and seed sample data if empty
    with app.app_context():
        try:
            init_db()
            # If database has no customers, populate demo records
            cust_check = query_db("SELECT COUNT(*) as count FROM customers", one=True)
            if not cust_check or cust_check["count"] == 0:
                logger.info("Fresh database detected; automatically seeding demo grocery customers...")
                seed_database_with_demo_data(n_customers=120)
        except Exception as e:
            logger.error("Database initialization error: %s", e)

    # -------------------------------------------------------------
    # HTML View Routes
    # -------------------------------------------------------------

    @app.route("/")
    def dashboard():
        """Main executive dashboard showing churn KPIs and risk breakdowns."""
        metrics = get_dashboard_metrics()
        return render_template(
            "dashboard.html",
            metrics=metrics,
            active_page="dashboard"
        )

    @app.route("/customers")
    def customers_list():
        """Customer directory with search, risk filters, sorting, and pagination."""
        page = request.args.get("page", 1, type=int)
        search = request.args.get("search", "", type=str)
        risk_filter = request.args.get("risk", "", type=str)
        sort_by = request.args.get("sort", "recency_days", type=str)
        order = request.args.get("order", "desc", type=str)

        result = get_customers(
            page=page,
            per_page=15,
            search=search,
            risk_filter=risk_filter,
            sort_by=sort_by,
            order=order
        )
        return render_template(
            "customers.html",
            data=result,
            active_page="customers"
        )

    @app.route("/customers/<customer_id>")
    def customer_detail(customer_id):
        """Individual customer profile, features, and full prediction history."""
        detail = get_customer_detail(customer_id)
        if not detail:
            flash(f"Customer '{customer_id}' was not found in the database.", "warning")
            return redirect(url_for("customers_list"))
        
        return render_template(
            "customer_detail.html",
            customer=detail["customer"],
            predictions=detail["predictions"],
            latest_prediction=detail["latest_prediction"],
            active_page="customers"
        )

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        """Interactive prediction form and churn scoring result view."""
        prediction_result = None
        form_data = {}
        errors = []

        if request.method == "POST":
            # Extract form values
            customer_id = request.form.get("customer_id", "").strip()
            raw_features = {}
            for col in FEATURE_COLUMNS:
                raw_features[col] = request.form.get(col, "").strip()
                form_data[col] = raw_features[col]
            form_data["customer_id"] = customer_id

            res = predict_customer_churn(
                raw_features=raw_features,
                customer_id=customer_id,
                save_to_db=True
            )

            if res.get("success"):
                prediction_result = res
                flash("Customer scored successfully and saved to database.", "success")
            else:
                errors = res.get("errors", ["Prediction failed due to invalid inputs."])
                for err in errors:
                    flash(err, "danger")

        # Pre-fill feature constraints metadata for form hints
        _, is_demo, model_name = get_loaded_model()
        return render_template(
            "predict.html",
            result=prediction_result,
            form_data=form_data,
            errors=errors,
            constraints=FEATURE_CONSTRAINTS,
            is_demo=is_demo,
            model_name=model_name,
            active_page="predict"
        )

    @app.route("/model-performance")
    def model_performance():
        """ML evaluation metrics, baseline comparison table, and methodology."""
        metrics_path = app.config.get("METRICS_PATH")
        metrics_data = None
        if Path(metrics_path).exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception as e:
                logger.error("Error loading metrics JSON: %s", e)

        # Also fetch recent runs from SQLite
        training_runs = query_db(
            """
            SELECT id, run_timestamp, model_name, best_metric, train_size, test_size, notes
            FROM model_training_runs
            ORDER BY id DESC
            LIMIT 10
            """
        )

        all_model_metrics = query_db(
            """
            SELECT m.*, r.run_timestamp
            FROM model_metrics m
            LEFT JOIN model_training_runs r ON m.run_id = r.id
            ORDER BY m.id DESC
            LIMIT 20
            """
        )

        _, is_demo, model_name = get_loaded_model()

        return render_template(
            "model_performance.html",
            metrics=metrics_data,
            training_runs=training_runs,
            all_model_metrics=all_model_metrics,
            is_demo=is_demo,
            model_name=model_name,
            active_page="model-performance"
        )

    # -------------------------------------------------------------
    # REST API Endpoints
    # -------------------------------------------------------------

    @app.route("/api/metrics")
    def api_metrics():
        """Return executive KPI dashboard data in JSON format."""
        try:
            metrics = get_dashboard_metrics()
            return jsonify({"status": "success", "data": metrics}), 200
        except Exception as e:
            logger.error("API Metrics error: %s", e)
            return jsonify({"status": "error", "message": "Failed to retrieve metrics."}), 500

    @app.route("/api/customers")
    def api_customers():
        """Return paginated customer data in JSON."""
        page = request.args.get("page", 1, type=int)
        per_page = min(request.args.get("per_page", 20, type=int), 100)
        search = request.args.get("search", "", type=str)
        risk_filter = request.args.get("risk", "", type=str)

        try:
            result = get_customers(page=page, per_page=per_page, search=search, risk_filter=risk_filter)
            return jsonify({"status": "success", "data": result}), 200
        except Exception as e:
            logger.error("API Customers error: %s", e)
            return jsonify({"status": "error", "message": "Failed to retrieve customers."}), 500

    @app.route("/api/predictions", methods=["GET", "POST"])
    def api_predictions():
        """Retrieve recent predictions or score customer via JSON payload."""
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            customer_id = payload.get("customer_id")
            features = payload.get("features", payload)
            
            res = predict_customer_churn(
                raw_features=features,
                customer_id=customer_id,
                save_to_db=True
            )
            if res.get("success"):
                return jsonify({"status": "success", "data": res}), 201
            return jsonify({"status": "error", "errors": res.get("errors")}), 400

        # GET predictions
        limit = min(request.args.get("limit", 20, type=int), 100)
        predictions = query_db(
            """
            SELECT id, customer_id, churn_probability, predicted_class, risk_category, model_name, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return jsonify({"status": "success", "data": predictions}), 200

    @app.route("/health")
    def health():
        """Render health check endpoint verifying database connectivity."""
        try:
            db_check = query_db("SELECT 1 as healthy", one=True)
            db_status = "healthy" if db_check and db_check["healthy"] == 1 else "unhealthy"
            _, is_demo, model_name = get_loaded_model()
            
            return jsonify({
                "status": "healthy" if db_status == "healthy" else "degraded",
                "database": db_status,
                "model_status": {
                    "model_name": model_name,
                    "is_demo_mode": is_demo
                },
                "service": "customer-churn-prediction"
            }), 200
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return jsonify({
                "status": "unhealthy",
                "error": "Database connectivity check failed"
            }), 503

    # -------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template(
            "base.html",
            error_code=404,
            error_title="Page Not Found",
            error_message="The requested page could not be located."
        ), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error("Unhandled internal server error: %s", e)
        return render_template(
            "base.html",
            error_code=500,
            error_title="Server Error",
            error_message="An unexpected internal error occurred. Please try again later."
        ), 500

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    logger.info("Starting Customer Churn Prediction Server on port %d...", port)
    app.run(host="0.0.0.0", port=port, debug=debug)

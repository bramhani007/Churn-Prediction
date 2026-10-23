# Machine Learning-Based Customer Churn Prediction System for Online Grocery and Food Retail

A full-stack, machine learning-driven customer churn prediction web application built for an online grocery and food retail e-commerce platform.

The system analyzes customer purchasing behavior (cadence, basket size, recency, reorder ratios), predicts the probability of customer churn using scikit-learn models, classifies customers into risk tiers (Low, Medium, High), stores evaluations in an SQLite database, and presents actionable insights on an executive SaaS dashboard.

---

## 1. Technology Stack

- **Programming Language:** Python 3.11
- **Backend Framework:** Flask 3.x
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5, Bootstrap Icons
- **Visualizations:** Chart.js
- **Database:** SQLite 3 (using Python's built-in `sqlite3`)
- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Model Algorithms:**
  - Logistic Regression (Balanced L2 regularized)
  - Random Forest Classifier (Balanced ensemble)
- **Production Server:** Gunicorn
- **Deployment Target:** GitHub & Render Web Service

---

## 2. Quick Start Guide (Windows PowerShell)

Follow these exact Windows PowerShell commands to clone, set up, and run the project locally.

### Step 1: Open PowerShell and Navigate to Project Directory
```powershell
cd customer-churn-prediction
```

### Step 2: Create and Activate Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(Note: If PowerShell displays an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and reactivate).*

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### Step 5: Generate Demo Data & Train Models (Zero-Config Setup)
The application automatically seeds SQLite database with demo data on initial launch. You can also generate synthetic training data and train the candidate ML models immediately:
```powershell
python -m ml.create_demo_data --samples 1500 --seed-db
python -m ml.train --input data/processed/customer_training.csv
```

### Step 6: Start the Flask Application
```powershell
python app.py
```
Open your browser and navigate to:
**http://localhost:3000**

---

## 3. Machine Learning Training Pipeline

The training pipeline trains 4 candidate algorithms with balanced class weights and a unified imputer/scaler preprocessing pipeline. It evaluates them using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**, selecting the model with the highest ROC-AUC with an F1-score tie-breaker.

### Run Model Training:
```powershell
python -m ml.train --input data/processed/customer_training.csv --output models/churn_pipeline.joblib --metrics models/model_metrics.json
```

Artifacts generated:
- `models/churn_pipeline.joblib`: The fitted scikit-learn pipeline.
- `models/model_metrics.json`: Full benchmark metrics, confusion matrices, and model rankings.
- Recorded training run logged in SQLite table `model_training_runs` and `model_metrics`.

---

## 4. Using Real Instacart Market Basket Data

The application supports the official Kaggle Instacart Market Basket dataset:
[https://www.kaggle.com/competitions/instacart-market-basket-analysis/data](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)

### Instructions:
1. Download `orders.csv` and `order_products__prior.csv` from Kaggle.
2. Place them into the `data/raw/` directory:
   ```
   data/raw/orders.csv
   data/raw/order_products__prior.csv
   ```
3. Run the leak-free data preparation pipeline:
   ```powershell
   python -m ml.prepare_instacart --data-dir data/raw --output data/processed/customer_training.csv --sample-size 10000
   ```
4. Retrain the machine learning pipeline:
   ```powershell
   python -m ml.train --input data/processed/customer_training.csv
   ```

### Churn Proxy & Leakage Prevention Methodology:
- **Observation Cutoff:** Customer behavioral features (total orders, recency, average order interval, average basket size, reorder ratio) are derived **strictly from orders $1$ to $N-1$**.
- **Target Formulation:** Churn proxy is evaluated on order $N$. If `days_since_prior_order` $\ge 30.0$ days (the Instacart ceiling), the customer is marked as churned ($1$), else retained ($0$).
- **No Fabricated Pricing:** Instacart contains no product prices. Average Order Value (AOV) is not fabricated; basket size and reorder ratios are used instead.

---

## 5. Running Automated Tests

Run the full pytest suite covering database schemas, feature constraints, predictions, and Flask routes:

```powershell
pytest -v
```

To run individual test modules:
```powershell
pytest tests/test_database.py -v
pytest tests/test_prediction.py -v
pytest tests/test_app.py -v
```

---

## 6. Render Deployment Instructions

This project includes `Procfile`, `requirements.txt`, `.python-version`, and environment configurations ready for 1-click deployment to [Render](https://render.com).

### Step 1: Push Code to GitHub
```powershell
git init
git add .
git commit -m "Deploy Customer Churn Prediction System"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-churn-prediction.git
git push -u origin main
```

### Step 2: Create Web Service on Render
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Configure service settings:
   - **Name:** `grocery-churn-prediction`
   - **Environment:** `Python 3`
   - **Region:** Any (e.g. Oregon or Frankfurt)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

### Step 3: Configure Environment Variables
Under the **Environment** tab on Render, add:
- `FLASK_ENV` = `production`
- `SECRET_KEY` = `generate-a-long-random-string`
- `DATABASE_PATH` = `database/churn.db`
- `RISK_THRESHOLD_LOW` = `0.35`
- `RISK_THRESHOLD_HIGH` = `0.65`

### Step 4: Verify Deployment & Health Check
Once deployed, verify the service by visiting:
- Dashboard: `https://your-service-name.onrender.com/`
- Health check: `https://your-service-name.onrender.com/health`

### ⚠️ SQLite Persistence Warning on Render:
Render web services run on an ephemeral root filesystem. If the container restarts or redeploys, SQLite files stored in the container will reset.
- **For persistent SQLite storage on Render:** Attach a Render Persistent Disk (e.g. mounted at `/var/data`) and set `DATABASE_PATH=/var/data/churn.db`.
- **For enterprise multi-instance clustering:** Managed PostgreSQL is recommended for concurrent multi-worker writes, whereas SQLite serves single-instance architectures.

---

## 7. Troubleshooting

- **Missing packages / ModuleNotFoundError:**
  Ensure your virtual environment is activated (`.\venv\Scripts\Activate.ps1`) before running `pip install -r requirements.txt`.
- **Database lock or permission issues:**
  Ensure the `database/` folder is writable. SQLite creates temporary journal files adjacent to `churn.db`.
- **Model warning "Demo Baseline Heuristics":**
  Run `python -m ml.train` to generate `models/churn_pipeline.joblib`.

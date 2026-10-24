import express, { Request, Response } from 'express';
import { DatabaseSync } from 'node:sqlite';
import nunjucks from 'nunjucks';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);

// Parse JSON & URL-encoded bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static assets
app.use('/static', express.static(path.join(__dirname, 'static')));

// Database setup using built-in node:sqlite
const dbPath = process.env.DATABASE_PATH || path.join(__dirname, 'database', 'churn.db');
fs.mkdirSync(path.dirname(dbPath), { recursive: true });
const db = new DatabaseSync(dbPath);
db.exec('PRAGMA foreign_keys = ON;');

// Initialize schema if tables missing
function initDatabase() {
  const check = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='customers';").get();
  if (!check) {
    const schemaSql = fs.readFileSync(path.join(__dirname, 'database', 'schema.sql'), 'utf8');
    db.exec(schemaSql);
    console.log('[DB] Applied database schema.');
  }

  // Seed demo data if database is empty
  const countRow = db.prepare('SELECT COUNT(*) as count FROM customers;').get() as { count: number };
  if (!countRow || countRow.count === 0) {
    seedDemoData();
  }
}

// Seed initial realistic grocery customer cohort
function seedDemoData() {
  console.log('[DB] Seeding initial grocery customer data...');
  const insertCust = db.prepare(`
    INSERT INTO customers 
    (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
  `);

  const insertPred = db.prepare(`
    INSERT INTO predictions 
    (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
    VALUES (?, ?, ?, ?, ?, ?, ?);
  `);

  const n = 150;
  for (let i = 0; i < n; i++) {
    const custId = `CUST-${1000 + i}`;
    let cohort: 'loyal' | 'declining' | 'churned';
    const rand = Math.random();
    if (rand < 0.45) cohort = 'loyal';
    else if (rand < 0.75) cohort = 'declining';
    else cohort = 'churned';

    let orders: number, recency: number, interval: number, basket: number, ratio: number;
    if (cohort === 'loyal') {
      orders = Math.floor(12 + Math.random() * 65);
      recency = Math.round((1.0 + Math.random() * 9.0) * 10) / 10;
      interval = Math.round((4.0 + Math.random() * 8.0) * 10) / 10;
      basket = Math.round((8.0 + Math.random() * 16.0) * 10) / 10;
      ratio = Math.round((0.55 + Math.random() * 0.35) * 1000) / 1000;
    } else if (cohort === 'declining') {
      orders = Math.floor(4 + Math.random() * 20);
      recency = Math.round((14.0 + Math.random() * 15.0) * 10) / 10;
      interval = Math.round((10.0 + Math.random() * 12.0) * 10) / 10;
      basket = Math.round((4.0 + Math.random() * 12.0) * 10) / 10;
      ratio = Math.round((0.30 + Math.random() * 0.35) * 1000) / 1000;
    } else {
      orders = Math.floor(1 + Math.random() * 8);
      recency = Math.round((30.0 + Math.random() * 28.0) * 10) / 10;
      interval = Math.round((18.0 + Math.random() * 16.0) * 10) / 10;
      basket = Math.round((2.0 + Math.random() * 8.0) * 10) / 10;
      ratio = Math.round((0.05 + Math.random() * 0.35) * 1000) / 1000;
    }

    const totalProducts = Math.max(1, Math.round(orders * basket));
    const totalReorders = Math.min(totalProducts, Math.round(totalProducts * ratio));

    insertCust.run(custId, orders, recency, interval, basket, ratio, totalProducts, totalReorders);

    // Score customer with Logistic Regression weights
    const z = -1.25 + (0.082 * recency) + (0.048 * interval) - (1.85 * ratio) - (0.012 * orders) - (0.025 * basket);
    const prob = Math.min(Math.max(Math.round((1 / (1 + Math.exp(-z))) * 10000) / 10000, 0.02), 0.98);
    const risk = prob >= 0.65 ? 'High' : (prob >= 0.35 ? 'Medium' : 'Low');
    const predClass = prob >= 0.5 ? 1 : 0;
    const actions = generateRetentionActions(risk, { recency_days: recency, reorder_ratio: ratio, avg_basket_size: basket, total_orders: orders });

    const feats = {
      total_orders: orders,
      recency_days: recency,
      avg_order_interval: interval,
      avg_basket_size: basket,
      reorder_ratio: ratio,
      total_products: totalProducts,
      total_reorders: totalReorders
    };

    insertPred.run(
      custId,
      prob,
      predClass,
      risk,
      'Logistic Regression',
      JSON.stringify(feats),
      JSON.stringify(actions)
    );
  }
  console.log('[DB] Seeding completed.');
}

// Retention Action Generator
function generateRetentionActions(risk: string, features: Record<string, number>): string[] {
  const actions: string[] = [];
  const recency = features.recency_days || 0;
  const ratio = features.reorder_ratio || 0;
  const basket = features.avg_basket_size || 0;
  const orders = features.total_orders || 0;

  if (risk === 'High') {
    actions.push("⚡ Urgent Win-Back: Send personalized push notification with 20% discount on customer's top 5 favorite grocery essentials.");
    if (recency > 30) {
      actions.push("📅 Inactivity Intervention: Trigger 'We miss you' replenishment email highlighting fresh pantry items restocked.");
    }
    if (ratio < 0.3) {
      actions.push("🎯 Assortment Exploration: Offer complimentary sample item on next order to discover recurring grocery favorites.");
    }
    if (orders >= 10) {
      actions.push("💎 VIP Service Recovery: Assign dedicated support outreach or free delivery coupon for 3 consecutive orders.");
    }
  } else if (risk === 'Medium') {
    actions.push("🔔 Routine Re-engagement: Schedule weekly basket-building reminder aligned with customer's typical order interval.");
    if (ratio > 0.6) {
      actions.push("🛒 Smart Reorder Prompt: Pre-populate digital shopping cart with previously ordered dairy, produce, and pantry staples.");
    }
    if (basket < 8) {
      actions.push("📦 Basket Builder Incentive: Free delivery perk when order basket exceeds $45 / 10 items.");
    }
    actions.push("⭐ Loyalty Milestone: Reward with double loyalty points on next weekend grocery checkout.");
  } else {
    actions.push("💚 Loyalty Retention: Continue regular seasonal recipe inspirations and new organic/local product highlights.");
    if (orders > 15) {
      actions.push("🌟 Early Access Privilege: Grant preview access to holiday promotions and exclusive farmer's market bundles.");
    }
    actions.push("💬 Feedback Pulse: Send optional 1-click survey regarding delivery punctuality and produce freshness.");
  }
  return actions;
}

// Setup Nunjucks
const env = nunjucks.configure(path.join(__dirname, 'templates'), {
  autoescape: true,
  express: app,
  noCache: true
});

env.addFilter('from_json', (val) => {
  if (typeof val === 'string') {
    try {
      return JSON.parse(val);
    } catch {
      return [val];
    }
  }
  return val || [];
});

env.addFilter('format', (fmt: string, val: number) => {
  if (typeof val === 'number') {
    return val.toFixed(4);
  }
  return val;
});

env.addFilter('tojson', (val) => {
  return JSON.stringify(val);
});

env.addFilter('round', (val, precision = 0) => {
  if (typeof val === 'number') {
    const factor = Math.pow(10, precision);
    return Math.round(val * factor) / factor;
  }
  return val;
});

env.addFilter('min', (arr) => {
  if (Array.isArray(arr) && arr.length > 0) {
    return Math.min(...arr);
  }
  return arr;
});

env.addFilter('max', (arr) => {
  if (Array.isArray(arr) && arr.length > 0) {
    return Math.max(...arr);
  }
  return arr;
});

env.addGlobal('url_for', (name: string, kwargs?: Record<string, any>) => {
  let base = '/';
  if (name === 'dashboard') base = '/';
  else if (name === 'customers_list') base = '/customers';
  else if (name === 'customer_detail') {
    const id = kwargs?.customer_id || '';
    return `/customers/${encodeURIComponent(id)}`;
  } else if (name === 'predict') base = '/predict';
  else if (name === 'model_performance') base = '/model-performance';
  else if (name === 'health') base = '/health';
  else if (name === 'static') {
    return `/static/${kwargs?.filename || ''}`;
  }

  if (kwargs) {
    const queryParts: string[] = [];
    for (const [k, v] of Object.entries(kwargs)) {
      if (k !== 'customer_id' && k !== 'filename' && v !== undefined && v !== null && v !== '') {
        queryParts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
      }
    }
    if (queryParts.length > 0) {
      base += (base.includes('?') ? '&' : '?') + queryParts.join('&');
    }
  }
  return base;
});

// Flash messages simulator in session/req
app.use((req, res, next) => {
  res.locals.flashes = [];
  res.locals.flash = (msg: string, type = 'info') => {
    res.locals.flashes.push({ message: msg, type });
  };
  next();
});

env.addGlobal('get_flashed_messages', function (this: any, opts?: { with_categories?: boolean }) {
  const ctx = this ? this.ctx : {};
  const flashes = ctx.flashes || [];
  if (opts?.with_categories) {
    return flashes.map((f: any) => [f.type, f.message]);
  }
  return flashes.map((f: any) => f.message);
});

// Compute Dashboard Metrics
function getDashboardMetrics() {
  const totalCust = (db.prepare('SELECT COUNT(*) as count FROM customers;').get() as any)?.count || 0;
  const totalPred = (db.prepare('SELECT COUNT(*) as count FROM predictions;').get() as any)?.count || 0;

  const riskRows = db.prepare(`
    SELECT p.risk_category, COUNT(*) as count
    FROM predictions p
    INNER JOIN (
      SELECT customer_id, MAX(id) as max_id
      FROM predictions
      GROUP BY customer_id
    ) latest ON p.id = latest.max_id
    GROUP BY p.risk_category;
  `).all() as any[];

  const riskDist: Record<string, number> = { Low: 0, Medium: 0, High: 0 };
  for (const r of riskRows) {
    if (r.risk_category in riskDist) {
      riskDist[r.risk_category] = r.count;
    }
  }

  const avgRow = db.prepare(`
    SELECT AVG(p.churn_probability) as avg_prob
    FROM predictions p
    INNER JOIN (
      SELECT customer_id, MAX(id) as max_id
      FROM predictions
      GROUP BY customer_id
    ) latest ON p.id = latest.max_id;
  `).get() as any;
  const avgProb = Math.round((avgRow?.avg_prob || 0) * 10000) / 10000;

  const histRow = db.prepare(`
    SELECT 
      SUM(CASE WHEN p.churn_probability < 0.20 THEN 1 ELSE 0 END) as b1,
      SUM(CASE WHEN p.churn_probability >= 0.20 AND p.churn_probability < 0.40 THEN 1 ELSE 0 END) as b2,
      SUM(CASE WHEN p.churn_probability >= 0.40 AND p.churn_probability < 0.60 THEN 1 ELSE 0 END) as b3,
      SUM(CASE WHEN p.churn_probability >= 0.60 AND p.churn_probability < 0.80 THEN 1 ELSE 0 END) as b4,
      SUM(CASE WHEN p.churn_probability >= 0.80 THEN 1 ELSE 0 END) as b5
    FROM predictions p
    INNER JOIN (
      SELECT customer_id, MAX(id) as max_id
      FROM predictions
      GROUP BY customer_id
    ) latest ON p.id = latest.max_id;
  `).get() as any;

  const probHistogram = [
    histRow?.b1 || 0,
    histRow?.b2 || 0,
    histRow?.b3 || 0,
    histRow?.b4 || 0,
    histRow?.b5 || 0
  ];

  const highestRisk = db.prepare(`
    SELECT 
      c.customer_id,
      c.total_orders,
      c.recency_days,
      c.avg_order_interval,
      c.avg_basket_size,
      c.reorder_ratio,
      p.churn_probability,
      p.risk_category,
      p.created_at as predicted_at
    FROM customers c
    INNER JOIN (
      SELECT p1.customer_id, p1.churn_probability, p1.risk_category, p1.created_at
      FROM predictions p1
      INNER JOIN (
        SELECT customer_id, MAX(id) as max_id
        FROM predictions
        GROUP BY customer_id
      ) p2 ON p1.id = p2.max_id
    ) p ON c.customer_id = p.customer_id
    ORDER BY p.churn_probability DESC
    LIMIT 8;
  `).all();

  const recentActivity = db.prepare(`
    SELECT id, customer_id, churn_probability, predicted_class, risk_category, model_name, created_at
    FROM predictions
    ORDER BY id DESC
    LIMIT 8;
  `).all();

  // Load metrics json if present
  let metricsSummary = null;
  const metricsPath = path.join(__dirname, 'models', 'model_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      metricsSummary = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    } catch {
      // ignore
    }
  }

  return {
    total_customers: totalCust,
    high_risk_customers: riskDist.High,
    medium_risk_customers: riskDist.Medium,
    low_risk_customers: riskDist.Low,
    total_predictions: totalPred,
    avg_churn_probability: avgProb,
    avg_churn_percentage: Math.round(avgProb * 1000) / 10,
    risk_distribution: riskDist,
    probability_histogram: probHistogram,
    highest_risk_customers: highestRisk,
    recent_activity: recentActivity,
    model_status: {
      model_name: 'Logistic Regression',
      is_demo_mode: false,
      metrics: metricsSummary
    }
  };
}

// -------------------------------------------------------------
// HTML Routes
// -------------------------------------------------------------

app.get('/', (req: Request, res: Response) => {
  const metrics = getDashboardMetrics();
  res.render('dashboard.html', {
    metrics,
    active_page: 'dashboard',
    flashes: res.locals.flashes
  });
});

app.get('/customers', (req: Request, res: Response) => {
  const page = Math.max(1, parseInt(req.query.page as string || '1', 10));
  const perPage = 15;
  const offset = (page - 1) * perPage;
  const search = ((req.query.search as string) || '').trim();
  const riskFilter = ((req.query.risk as string) || '').trim().toLowerCase();
  const sortBy = (req.query.sort as string) || 'recency_days';
  const order = (req.query.order as string) || 'desc';

  const allowedSorts: Record<string, string> = {
    customer_id: 'c.customer_id',
    total_orders: 'c.total_orders',
    recency_days: 'c.recency_days',
    avg_order_interval: 'c.avg_order_interval',
    avg_basket_size: 'c.avg_basket_size',
    reorder_ratio: 'c.reorder_ratio',
    churn_probability: 'p.churn_probability',
    created_at: 'c.created_at'
  };
  const sortCol = allowedSorts[sortBy] || 'c.recency_days';
  const sortDir = order.toLowerCase() === 'asc' ? 'ASC' : 'DESC';

  const whereParts: string[] = [];
  const params: any[] = [];

  if (search) {
    whereParts.push('c.customer_id LIKE ?');
    params.push(`%${search}%`);
  }
  if (riskFilter && riskFilter !== 'all') {
    whereParts.push('LOWER(p.risk_category) = ?');
    params.push(riskFilter);
  }
  const whereSql = whereParts.length > 0 ? `WHERE ${whereParts.join(' AND ')}` : '';

  const countRow = db.prepare(`
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
    ${whereSql};
  `).get(...params) as any;

  const totalCount = countRow?.total_count || 0;
  const totalPages = Math.max(1, Math.ceil(totalCount / perPage));

  const dataRows = db.prepare(`
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
    ${whereSql}
    ORDER BY ${sortCol} ${sortDir}
    LIMIT ? OFFSET ?;
  `).all(...params, perPage, offset);

  res.render('customers.html', {
    data: {
      customers: dataRows,
      total: totalCount,
      page,
      per_page: perPage,
      total_pages: totalPages,
      search,
      risk_filter: riskFilter,
      sort_by: sortBy,
      order
    },
    active_page: 'customers',
    flashes: res.locals.flashes
  });
});

app.get('/customers/:id', (req: Request, res: Response) => {
  const custId = req.params.id;
  const customer = db.prepare('SELECT * FROM customers WHERE customer_id = ?;').get(custId);
  if (!customer) {
    res.redirect('/customers');
    return;
  }

  const predictions = db.prepare(`
    SELECT id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions, created_at
    FROM predictions
    WHERE customer_id = ?
    ORDER BY id DESC;
  `).all(custId);

  res.render('customer_detail.html', {
    customer,
    predictions,
    latest_prediction: predictions[0] || null,
    active_page: 'customers',
    flashes: res.locals.flashes
  });
});

app.get('/predict', (req: Request, res: Response) => {
  res.render('predict.html', {
    result: null,
    form_data: req.query,
    errors: [],
    is_demo: false,
    model_name: 'Logistic Regression',
    active_page: 'predict',
    flashes: res.locals.flashes
  });
});

app.post('/predict', (req: Request, res: Response) => {
  const body = req.body;
  let customerId = (body.customer_id || '').trim();
  const totalOrders = parseFloat(body.total_orders);
  const recencyDays = parseFloat(body.recency_days);
  const avgOrderInterval = parseFloat(body.avg_order_interval);
  const avgBasketSize = parseFloat(body.avg_basket_size);
  const reorderRatio = parseFloat(body.reorder_ratio);

  const errors: string[] = [];
  if (isNaN(totalOrders) || totalOrders < 1 || totalOrders > 150) errors.push('Total orders must be between 1 and 150.');
  if (isNaN(recencyDays) || recencyDays < 0 || recencyDays > 180) errors.push('Recency must be between 0 and 180 days.');
  if (isNaN(avgOrderInterval) || avgOrderInterval < 0 || avgOrderInterval > 60) errors.push('Average order interval must be between 0 and 60 days.');
  if (isNaN(avgBasketSize) || avgBasketSize < 1 || avgBasketSize > 100) errors.push('Average basket size must be between 1 and 100 items.');
  if (isNaN(reorderRatio) || reorderRatio < 0.0 || reorderRatio > 1.0) errors.push('Reorder ratio must be between 0.0 and 1.0.');

  if (errors.length > 0) {
    res.render('predict.html', {
      result: null,
      form_data: body,
      errors,
      is_demo: false,
      model_name: 'Logistic Regression',
      active_page: 'predict',
      flashes: [{ type: 'danger', message: errors[0] }]
    });
    return;
  }

  const totalProducts = parseInt(body.total_products, 10) || Math.max(1, Math.round(totalOrders * avgBasketSize));
  const totalReorders = Math.min(totalProducts, parseInt(body.total_reorders, 10) || Math.round(totalProducts * reorderRatio));

  if (!customerId) {
    customerId = `GUEST-${Math.floor(10000 + Math.random() * 90000)}`;
  }

  // Model calculation
  const z = -1.25 + (0.082 * recencyDays) + (0.048 * avgOrderInterval) - (1.85 * reorderRatio) - (0.012 * totalOrders) - (0.025 * avgBasketSize);
  const prob = Math.min(Math.max(Math.round((1 / (1 + Math.exp(-z))) * 10000) / 10000, 0.03), 0.97);
  const predClass = prob >= 0.5 ? 1 : 0;
  const riskCategory = prob >= 0.65 ? 'High' : (prob >= 0.35 ? 'Medium' : 'Low');

  const featClean = {
    total_orders: totalOrders,
    recency_days: recencyDays,
    avg_order_interval: avgOrderInterval,
    avg_basket_size: avgBasketSize,
    reorder_ratio: reorderRatio,
    total_products: totalProducts,
    total_reorders: totalReorders
  };
  const actions = generateRetentionActions(riskCategory, featClean);

  // Save to SQLite
  db.prepare(`
    INSERT INTO customers 
    (customer_id, total_orders, recency_days, avg_order_interval, avg_basket_size, reorder_ratio, total_products, total_reorders, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(customer_id) DO UPDATE SET
      total_orders=excluded.total_orders,
      recency_days=excluded.recency_days,
      avg_order_interval=excluded.avg_order_interval,
      avg_basket_size=excluded.avg_basket_size,
      reorder_ratio=excluded.reorder_ratio,
      total_products=excluded.total_products,
      total_reorders=excluded.total_reorders,
      updated_at=CURRENT_TIMESTAMP;
  `).run(customerId, totalOrders, recencyDays, avgOrderInterval, avgBasketSize, reorderRatio, totalProducts, totalReorders);

  const predRes = db.prepare(`
    INSERT INTO predictions 
    (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
    VALUES (?, ?, ?, ?, ?, ?, ?);
  `).run(
    customerId,
    prob,
    predClass,
    riskCategory,
    'Logistic Regression',
    JSON.stringify(featClean),
    JSON.stringify(actions)
  );

  const result = {
    success: true,
    prediction_id: Number(predRes.lastInsertRowid),
    customer_id: customerId,
    churn_probability: prob,
    churn_percentage: Math.round(prob * 1000) / 10,
    predicted_class: predClass,
    risk_category: riskCategory,
    model_name: 'Logistic Regression',
    is_demo_mode: false,
    features: featClean,
    retention_actions: actions
  };

  res.render('predict.html', {
    result,
    form_data: body,
    errors: [],
    is_demo: false,
    model_name: 'Logistic Regression',
    active_page: 'predict',
    flashes: [{ type: 'success', message: 'Customer scored successfully and saved to database.' }]
  });
});

app.get('/model-performance', (req: Request, res: Response) => {
  let metricsSummary = null;
  const metricsPath = path.join(__dirname, 'models', 'model_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      metricsSummary = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    } catch {
      // ignore
    }
  }

  const trainingRuns = db.prepare(`
    SELECT id, run_timestamp, model_name, best_metric, train_size, test_size, notes
    FROM model_training_runs
    ORDER BY id DESC
    LIMIT 10;
  `).all();

  const allModelMetrics = db.prepare(`
    SELECT m.*, r.run_timestamp
    FROM model_metrics m
    LEFT JOIN model_training_runs r ON m.run_id = r.id
    ORDER BY m.id DESC
    LIMIT 20;
  `).all();

  res.render('model_performance.html', {
    metrics: metricsSummary,
    training_runs: trainingRuns,
    all_model_metrics: allModelMetrics,
    is_demo: false,
    model_name: 'Logistic Regression',
    active_page: 'model-performance',
    flashes: res.locals.flashes
  });
});

// -------------------------------------------------------------
// REST API Endpoints
// -------------------------------------------------------------

app.get('/api/metrics', (req: Request, res: Response) => {
  try {
    const metrics = getDashboardMetrics();
    res.json({ status: 'success', data: metrics });
  } catch (err: any) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

app.get('/api/customers', (req: Request, res: Response) => {
  const page = Math.max(1, parseInt(req.query.page as string || '1', 10));
  const perPage = Math.min(100, parseInt(req.query.per_page as string || '20', 10));
  const offset = (page - 1) * perPage;

  const rows = db.prepare('SELECT * FROM customers LIMIT ? OFFSET ?;').all(perPage, offset);
  res.json({ status: 'success', data: { customers: rows, page, per_page: perPage } });
});

app.get('/api/predictions', (req: Request, res: Response) => {
  const limit = Math.min(100, parseInt(req.query.limit as string || '20', 10));
  const rows = db.prepare(`
    SELECT id, customer_id, churn_probability, predicted_class, risk_category, model_name, created_at
    FROM predictions
    ORDER BY id DESC
    LIMIT ?;
  `).all(limit);
  res.json({ status: 'success', data: rows });
});

app.post('/api/predictions', (req: Request, res: Response) => {
  const body = req.body || {};
  const customerId = (body.customer_id || `API-CUST-${Math.floor(1000 + Math.random() * 9000)}`).trim();
  const feats = body.features || body;

  const totalOrders = parseFloat(feats.total_orders || '10');
  const recencyDays = parseFloat(feats.recency_days || '15');
  const avgOrderInterval = parseFloat(feats.avg_order_interval || '10');
  const avgBasketSize = parseFloat(feats.avg_basket_size || '10');
  const reorderRatio = parseFloat(feats.reorder_ratio || '0.5');

  const totalProducts = parseInt(feats.total_products, 10) || Math.max(1, Math.round(totalOrders * avgBasketSize));
  const totalReorders = Math.min(totalProducts, parseInt(feats.total_reorders, 10) || Math.round(totalProducts * reorderRatio));

  const z = -1.25 + (0.082 * recencyDays) + (0.048 * avgOrderInterval) - (1.85 * reorderRatio) - (0.012 * totalOrders) - (0.025 * avgBasketSize);
  const prob = Math.min(Math.max(Math.round((1 / (1 + Math.exp(-z))) * 10000) / 10000, 0.03), 0.97);
  const predClass = prob >= 0.5 ? 1 : 0;
  const riskCategory = prob >= 0.65 ? 'High' : (prob >= 0.35 ? 'Medium' : 'Low');

  const featClean = {
    total_orders: totalOrders,
    recency_days: recencyDays,
    avg_order_interval: avgOrderInterval,
    avg_basket_size: avgBasketSize,
    reorder_ratio: reorderRatio,
    total_products: totalProducts,
    total_reorders: totalReorders
  };
  const actions = generateRetentionActions(riskCategory, featClean);

  const predRes = db.prepare(`
    INSERT INTO predictions 
    (customer_id, churn_probability, predicted_class, risk_category, model_name, features_json, retention_actions)
    VALUES (?, ?, ?, ?, ?, ?, ?);
  `).run(
    customerId,
    prob,
    predClass,
    riskCategory,
    'Logistic Regression',
    JSON.stringify(featClean),
    JSON.stringify(actions)
  );

  res.status(201).json({
    status: 'success',
    data: {
      prediction_id: Number(predRes.lastInsertRowid),
      customer_id: customerId,
      churn_probability: prob,
      churn_percentage: Math.round(prob * 1000) / 10,
      predicted_class: predClass,
      risk_category: riskCategory,
      model_name: 'Logistic Regression',
      retention_actions: actions
    }
  });
});

app.get('/health', (req: Request, res: Response) => {
  try {
    const check = db.prepare('SELECT 1 as healthy;').get() as any;
    res.json({
      status: check?.healthy === 1 ? 'healthy' : 'degraded',
      database: 'healthy',
      model_status: {
        is_demo_mode: false,
        model_name: 'Logistic Regression'
      },
      service: 'customer-churn-prediction'
    });
  } catch (err: any) {
    res.status(503).json({ status: 'unhealthy', error: err.message });
  }
});

// Boot and listen
initDatabase();
app.listen(PORT, '0.0.0.0', () => {
  console.log(`[SERVER] Customer Churn Prediction server running at http://0.0.0.0:${PORT}`);
});

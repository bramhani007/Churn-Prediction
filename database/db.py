import sqlite3
import os
import logging
from pathlib import Path
from flask import g, current_app

logger = logging.getLogger(__name__)

def resolve_db_path(custom_path=None):
    if custom_path:
        return custom_path
    try:
        return current_app.config.get("DATABASE_PATH")
    except RuntimeError:
        return os.environ.get(
            "DATABASE_PATH",
            str(Path(__file__).resolve().parent / "churn.db")
        )

def get_db(db_path=None):
    """Return an active SQLite database connection with row factory enabled."""
    target_path = resolve_db_path(db_path)
    
    # If in Flask request context and no custom path, use application context `g`
    if not db_path:
        try:
            if "db" not in g:
                if target_path != ":memory:":
                    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                g.db = sqlite3.connect(target_path, detect_types=sqlite3.PARSE_DECLTYPES)
                g.db.row_factory = sqlite3.Row
                g.db.execute("PRAGMA foreign_keys = ON;")
            return g.db
        except RuntimeError:
            pass  # Outside request context

    if target_path != ":memory:":
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def close_db(e=None):
    """Close the database connection at the end of the request."""
    try:
        db = g.pop("db", None)
        if db is not None:
            db.close()
    except RuntimeError:
        pass

def init_db(db_path=None):
    """Initialize database tables and indexes from schema.sql."""
    target_path = resolve_db_path(db_path)
    schema_file = Path(__file__).resolve().parent / "schema.sql"
    
    logger.info("Initializing SQLite database at %s", target_path)
    if target_path != ":memory:":
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(schema_file, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    logger.info("Database schema applied successfully.")

def query_db(query, args=(), one=False, db_path=None):
    """Execute a read query with parameters and return dict rows."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    # Close standalone connection if not in Flask request context
    try:
        if "db" not in g:
            conn.close()
    except RuntimeError:
        conn.close()
    
    results = [dict(row) for row in rv]
    return (results[0] if results else None) if one else results

def execute_db(query, args=(), db_path=None):
    """Execute an insert/update/delete statement with parameters."""
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute(query, args)
    conn.commit()
    last_id = cur.lastrowid
    cur.close()
    try:
        if "db" not in g:
            conn.close()
    except RuntimeError:
        conn.close()
    return last_id

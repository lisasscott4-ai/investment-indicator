import hashlib
import os
import secrets
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool, NullPool

# ── Engine setup ──────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///investment.db')

# Railway/Render supply postgres:// URLs; SQLAlchemy needs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

IS_POSTGRES = DATABASE_URL.startswith('postgresql')

if IS_POSTGRES:
    engine = create_engine(DATABASE_URL, poolclass=NullPool)
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )

@contextmanager
def get_db():
    with engine.begin() as conn:
        yield conn

# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(result):
    row = result.mappings().fetchone()
    return dict(row) if row else None

def _rows(result):
    return [dict(r) for r in result.mappings().all()]

def _insert_ignore(conn, sql_sqlite, sql_pg, params):
    sql = sql_pg if IS_POSTGRES else sql_sqlite
    conn.execute(text(sql), params)

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL UNIQUE,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',
    email         TEXT,
    birthday      TEXT,
    phone         TEXT,
    address       TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id         SERIAL PRIMARY KEY,
    token      TEXT NOT NULL UNIQUE,
    user_id    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS assets (
    id                  SERIAL PRIMARY KEY,
    user_id             TEXT NOT NULL DEFAULT 'anon',
    ticker              TEXT NOT NULL,
    name                TEXT,
    market              TEXT NOT NULL DEFAULT 'US',
    active              INTEGER DEFAULT 1,
    sharesies_available INTEGER DEFAULT 0,
    added_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS snapshots (
    id               SERIAL PRIMARY KEY,
    asset_id         INTEGER REFERENCES assets(id),
    timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price            REAL,
    price_change_pct REAL,
    momentum_score   REAL,
    financial_score  REAL,
    sentiment_score  REAL,
    industry_score   REAL,
    valuation_score  REAL,
    total_score      REAL,
    risk_level       TEXT,
    confidence       INTEGER,
    time_horizon     TEXT,
    reasoning_json   TEXT,
    signals_json     TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id             SERIAL PRIMARY KEY,
    user_id        TEXT NOT NULL,
    batch_id       TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    market         TEXT NOT NULL,
    company_name   TEXT,
    why_interesting TEXT,
    theme          TEXT,
    industry       TEXT,
    price_at_rec   REAL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL UNIQUE,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',
    email         TEXT,
    birthday      TEXT,
    phone         TEXT,
    address       TEXT,
    created_at    TEXT DEFAULT (CURRENT_TIMESTAMP)
);
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token      TEXT NOT NULL UNIQUE,
    user_id    TEXT NOT NULL,
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP)
);
CREATE TABLE IF NOT EXISTS assets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL DEFAULT 'anon',
    ticker              TEXT NOT NULL,
    name                TEXT,
    market              TEXT NOT NULL DEFAULT 'US',
    active              INTEGER DEFAULT 1,
    sharesies_available INTEGER DEFAULT 0,
    added_at            TEXT DEFAULT (CURRENT_TIMESTAMP)
);
CREATE TABLE IF NOT EXISTS snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id         INTEGER REFERENCES assets(id),
    timestamp        TEXT DEFAULT (CURRENT_TIMESTAMP),
    price            REAL,
    price_change_pct REAL,
    momentum_score   REAL,
    financial_score  REAL,
    sentiment_score  REAL,
    industry_score   REAL,
    valuation_score  REAL,
    total_score      REAL,
    risk_level       TEXT,
    confidence       INTEGER,
    time_horizon     TEXT,
    reasoning_json   TEXT,
    signals_json     TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    batch_id        TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    market          TEXT NOT NULL,
    company_name    TEXT,
    why_interesting TEXT,
    theme           TEXT,
    industry        TEXT,
    price_at_rec    REAL,
    created_at      TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""

def init_db():
    schema = _SCHEMA_PG if IS_POSTGRES else _SCHEMA_SQLITE
    with engine.begin() as conn:
        for stmt in schema.strip().split(';'):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

        # Migrations for existing databases
        for sql in [
            "ALTER TABLE assets ADD COLUMN sharesies_available INTEGER DEFAULT 0",
            "ALTER TABLE assets ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anon'",
            "ALTER TABLE recommendations ADD COLUMN industry TEXT",
            "ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'pending'",
            "ALTER TABLE users ADD COLUMN email TEXT",
            "ALTER TABLE users ADD COLUMN birthday TEXT",
            "ALTER TABLE users ADD COLUMN phone TEXT",
            "ALTER TABLE users ADD COLUMN address TEXT",
            # Existing admin accounts should be active
            "UPDATE users SET status = 'active' WHERE is_admin = 1 AND (status IS NULL OR status = 'pending')",
        ]:
            try:
                conn.execute(text(sql))
            except Exception:
                pass

        # Unique index per user
        try:
            conn.execute(text("DROP INDEX IF EXISTS idx_assets_ticker_market"))
        except Exception:
            pass
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_ticker_market_user "
                "ON assets(ticker, market, user_id)"
            ))
        except Exception:
            pass

# ── Auth ──────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
    return salt.hex() + ':' + key.hex()

def _check_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split(':', 1)
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
    return secrets.compare_digest(key.hex(), key_hex)

def create_user(username: str, password: str, is_admin: bool = False,
                email: str = None, birthday: str = None,
                phone: str = None, address: str = None) -> str:
    user_id = secrets.token_urlsafe(16)
    password_hash = _hash_password(password)
    # Admins are active immediately; regular users need approval
    status = 'active' if is_admin else 'pending'
    try:
        with get_db() as conn:
            conn.execute(text(
                "INSERT INTO users (user_id, username, password_hash, is_admin, status, "
                "email, birthday, phone, address) "
                "VALUES (:uid, :username, :pw, :admin, :status, :email, :birthday, :phone, :address)"
            ), {'uid': user_id, 'username': username.lower().strip(),
                'pw': password_hash, 'admin': 1 if is_admin else 0,
                'status': status, 'email': email, 'birthday': birthday,
                'phone': phone, 'address': address})
    except Exception as e:
        if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
            raise ValueError("Username already taken")
        raise
    return user_id

def authenticate_user(username: str, password: str):
    with get_db() as conn:
        row = _row(conn.execute(
            text("SELECT user_id, password_hash, status FROM users WHERE username = :u"),
            {'u': username.lower().strip()}
        ))
    if not row or not _check_password(password, row['password_hash']):
        return None, None
    return row['user_id'], row.get('status', 'active')

def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute(
            text("INSERT INTO sessions (token, user_id) VALUES (:token, :uid)"),
            {'token': token, 'uid': user_id}
        )
    return token

def get_session_user(token: str):
    with get_db() as conn:
        row = _row(conn.execute(
            text("SELECT user_id FROM sessions WHERE token = :t"),
            {'t': token}
        ))
    return row['user_id'] if row else None

def delete_session(token: str):
    with get_db() as conn:
        conn.execute(text("DELETE FROM sessions WHERE token = :t"), {'t': token})

def get_username(user_id: str):
    with get_db() as conn:
        row = _row(conn.execute(
            text("SELECT username FROM users WHERE user_id = :uid"),
            {'uid': user_id}
        ))
    return row['username'] if row else None

def is_admin(user_id: str) -> bool:
    with get_db() as conn:
        row = _row(conn.execute(
            text("SELECT is_admin FROM users WHERE user_id = :uid"),
            {'uid': user_id}
        ))
    return bool(row and row['is_admin'])

def admin_exists() -> bool:
    with get_db() as conn:
        row = _row(conn.execute(text("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1")))
    return row is not None

def get_pending_users():
    with get_db() as conn:
        return _rows(conn.execute(text(
            "SELECT user_id, username, email, birthday, phone, address, created_at "
            "FROM users WHERE status = 'pending' ORDER BY created_at ASC"
        )))

def approve_user(user_id: str):
    with get_db() as conn:
        conn.execute(text(
            "UPDATE users SET status = 'active' WHERE user_id = :uid"
        ), {'uid': user_id})

def reject_user(user_id: str):
    with get_db() as conn:
        conn.execute(text(
            "UPDATE users SET status = 'rejected' WHERE user_id = :uid"
        ), {'uid': user_id})

def get_all_users():
    with get_db() as conn:
        return _rows(conn.execute(text("""
            SELECT
                u.user_id, u.username, u.is_admin, u.created_at,
                COUNT(DISTINCT a.id)  AS asset_count,
                MAX(s.timestamp)      AS last_active,
                COUNT(DISTINCT s.id)  AS snapshot_count,
                COUNT(DISTINCT r.id)  AS rec_count
            FROM users u
            LEFT JOIN assets a ON a.user_id = u.user_id AND a.active = 1
            LEFT JOIN snapshots s ON s.asset_id = a.id
            LEFT JOIN recommendations r ON r.user_id = u.user_id
            GROUP BY u.user_id, u.username, u.is_admin, u.created_at
            ORDER BY u.created_at DESC
        """)))

def get_platform_stats():
    with get_db() as conn:
        def scalar(sql):
            return conn.execute(text(sql)).scalar()
        active_sql = (
            "SELECT COUNT(DISTINCT a.user_id) FROM assets a "
            "JOIN snapshots s ON s.asset_id = a.id "
            "WHERE s.timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day'"
            if IS_POSTGRES else
            "SELECT COUNT(DISTINCT a.user_id) FROM assets a "
            "JOIN snapshots s ON s.asset_id = a.id "
            "WHERE s.timestamp >= datetime('now', '-1 day')"
        )
        return {
            'total_users':           scalar("SELECT COUNT(*) FROM users"),
            'total_assets':          scalar("SELECT COUNT(*) FROM assets WHERE active = 1"),
            'total_snapshots':       scalar("SELECT COUNT(*) FROM snapshots"),
            'total_recommendations': scalar("SELECT COUNT(*) FROM recommendations"),
            'active_today':          scalar(active_sql),
        }

def get_health_warnings():
    with get_db() as conn:
        no_data = _rows(conn.execute(text("""
            SELECT a.ticker, a.market, a.user_id, u.username, a.added_at
            FROM assets a
            JOIN users u ON u.user_id = a.user_id
            LEFT JOIN snapshots s ON s.asset_id = a.id
            WHERE a.active = 1 AND s.id IS NULL
            ORDER BY u.username, a.ticker
        """)))
        stale_sql = (
            "SELECT a.ticker, a.market, a.user_id, u.username, MAX(s.timestamp) AS last_snapshot "
            "FROM assets a "
            "JOIN users u ON u.user_id = a.user_id "
            "JOIN snapshots s ON s.asset_id = a.id "
            "WHERE a.active = 1 AND a.added_at < CURRENT_TIMESTAMP - INTERVAL '1 day' "
            "GROUP BY a.id, a.ticker, a.market, a.user_id, u.username "
            "HAVING MAX(s.timestamp) < CURRENT_TIMESTAMP - INTERVAL '2 days' "
            "ORDER BY last_snapshot ASC"
            if IS_POSTGRES else
            "SELECT a.ticker, a.market, a.user_id, u.username, MAX(s.timestamp) AS last_snapshot "
            "FROM assets a "
            "JOIN users u ON u.user_id = a.user_id "
            "JOIN snapshots s ON s.asset_id = a.id "
            "WHERE a.active = 1 AND a.added_at < datetime('now', '-1 day') "
            "GROUP BY a.id "
            "HAVING MAX(s.timestamp) < datetime('now', '-2 days') "
            "ORDER BY last_snapshot ASC"
        )
        stale = _rows(conn.execute(text(stale_sql)))
    return {'no_data': no_data, 'stale': stale}

# ── Assets ────────────────────────────────────────────────────────────────────

def add_asset(ticker: str, market: str, sharesies_available: bool, user_id: str):
    t = ticker.upper()
    m = market.upper()
    s = 1 if sharesies_available else 0
    with get_db() as conn:
        _insert_ignore(conn,
            "INSERT OR IGNORE INTO assets (ticker, market, sharesies_available, user_id) "
            "VALUES (:t, :m, :s, :uid)",
            "INSERT INTO assets (ticker, market, sharesies_available, user_id) "
            "VALUES (:t, :m, :s, :uid) ON CONFLICT (ticker, market, user_id) DO NOTHING",
            {'t': t, 'm': m, 's': s, 'uid': user_id}
        )
        conn.execute(text(
            "UPDATE assets SET sharesies_available = :s, active = 1 "
            "WHERE ticker = :t AND market = :m AND user_id = :uid"
        ), {'s': s, 't': t, 'm': m, 'uid': user_id})
        return _row(conn.execute(text(
            "SELECT * FROM assets WHERE ticker = :t AND market = :m AND user_id = :uid"
        ), {'t': t, 'm': m, 'uid': user_id}))

def set_sharesies_flag(asset_id: int, available: bool):
    with get_db() as conn:
        conn.execute(text(
            "UPDATE assets SET sharesies_available = :s WHERE id = :id"
        ), {'s': 1 if available else 0, 'id': asset_id})

def update_asset_name(asset_id: int, name: str):
    with get_db() as conn:
        conn.execute(text("UPDATE assets SET name = :n WHERE id = :id"),
                     {'n': name, 'id': asset_id})

def remove_asset(asset_id: int):
    with get_db() as conn:
        conn.execute(text("UPDATE assets SET active = 0 WHERE id = :id"), {'id': asset_id})

def get_active_assets(user_id: str = None):
    with get_db() as conn:
        if user_id:
            return _rows(conn.execute(text(
                "SELECT * FROM assets WHERE active = 1 AND user_id = :uid ORDER BY added_at DESC"
            ), {'uid': user_id}))
        return _rows(conn.execute(text(
            "SELECT * FROM assets WHERE active = 1 ORDER BY added_at DESC"
        )))

def save_snapshot(data: dict):
    with get_db() as conn:
        conn.execute(text("""
            INSERT INTO snapshots
                (asset_id, price, price_change_pct, momentum_score, financial_score,
                 sentiment_score, industry_score, valuation_score, total_score,
                 risk_level, confidence, time_horizon, reasoning_json, signals_json)
            VALUES
                (:asset_id, :price, :pct, :momentum, :financial,
                 :sentiment, :industry, :valuation, :total,
                 :risk, :confidence, :horizon, :reasoning, :signals)
        """), {
            'asset_id':  data['asset_id'],
            'price':     data['price'],
            'pct':       data['price_change_pct'],
            'momentum':  data['momentum_score'],
            'financial': data['financial_score'],
            'sentiment': data['sentiment_score'],
            'industry':  data['industry_score'],
            'valuation': data['valuation_score'],
            'total':     data['total_score'],
            'risk':      data['risk_level'],
            'confidence': data['confidence'],
            'horizon':   data['time_horizon'],
            'reasoning': data['reasoning_json'],
            'signals':   data['signals_json'],
        })

def get_latest_snapshots(user_id: str):
    with get_db() as conn:
        return _rows(conn.execute(text("""
            SELECT a.id, a.ticker, a.market, a.name, a.sharesies_available, a.added_at,
                   s.timestamp, s.price, s.price_change_pct,
                   s.momentum_score, s.financial_score, s.sentiment_score,
                   s.industry_score, s.valuation_score, s.total_score,
                   s.risk_level, s.confidence, s.time_horizon,
                   s.reasoning_json, s.signals_json
            FROM assets a
            LEFT JOIN snapshots s ON s.asset_id = a.id
            WHERE a.active = 1 AND a.user_id = :uid
              AND (s.id IS NULL OR s.timestamp = (
                  SELECT MAX(timestamp) FROM snapshots WHERE asset_id = a.id
              ))
            ORDER BY s.total_score DESC NULLS LAST
        """), {'uid': user_id}))

def get_asset_history(asset_id: int, limit: int = 30):
    with get_db() as conn:
        return _rows(conn.execute(text(
            "SELECT * FROM snapshots WHERE asset_id = :id ORDER BY timestamp DESC LIMIT :lim"
        ), {'id': asset_id, 'lim': limit}))

# ── Recommendations ───────────────────────────────────────────────────────────

def save_recommendations(user_id: str, batch_id: str, items: list):
    with get_db() as conn:
        for item in items:
            conn.execute(text("""
                INSERT INTO recommendations
                    (user_id, batch_id, ticker, market, company_name,
                     why_interesting, theme, industry, price_at_rec)
                VALUES
                    (:uid, :batch, :ticker, :market, :company,
                     :why, :theme, :industry, :price)
            """), {
                'uid':     user_id,
                'batch':   batch_id,
                'ticker':  item['ticker'],
                'market':  item['market'],
                'company': item.get('company_name', ''),
                'why':     item.get('why_interesting', ''),
                'theme':   item.get('theme', ''),
                'industry': item.get('industry', ''),
                'price':   item.get('price_at_rec'),
            })

def get_latest_recommendations(user_id: str):
    with get_db() as conn:
        latest = _row(conn.execute(text(
            "SELECT batch_id FROM recommendations WHERE user_id = :uid "
            "ORDER BY created_at DESC LIMIT 1"
        ), {'uid': user_id}))
        if not latest:
            return []
        return _rows(conn.execute(text(
            "SELECT * FROM recommendations WHERE user_id = :uid AND batch_id = :batch "
            "ORDER BY created_at DESC"
        ), {'uid': user_id, 'batch': latest['batch_id']}))

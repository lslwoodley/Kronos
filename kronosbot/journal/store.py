"""SQLite journal store for bars, signals, backtest results, orders, fills, and equity."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class Journal:
    """SQLite-backed journal for Kronos Bot execution data."""

    TABLES = {
        "bars": """
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        """,
        "signals": """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                trend INTEGER,
                breakout INTEGER,
                volume_spike INTEGER,
                atr REAL,
                forecast_return REAL,
                entry_signal INTEGER,
                UNIQUE(symbol, timestamp)
            )
        """,
        "backtest_runs": """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                return_pct REAL,
                sharpe REAL,
                max_drawdown REAL,
                params TEXT,
                created_at TEXT NOT NULL
            )
        """,
        "backtest_trades": """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                return_pct REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
            )
        """,
        "orders": """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                units INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """,
        "fills": """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                commission REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
        """,
        "equity": """
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL
            )
        """,
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else Path("data/kronosbot.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for sql in self.TABLES.values():
                conn.execute(sql)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def list_tables(self) -> List[str]:
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]

    def log_bars(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        data = df[cols].copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        rows = [(symbol, *row) for row in data.itertuples(index=False, name=None)]
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO bars
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def read_bars(self, symbol: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM bars WHERE symbol = ? ORDER BY timestamp",
                (symbol,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_signals(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        cols = [
            "timestamp",
            "trend",
            "breakout",
            "volume_spike",
            "atr",
            "forecast_return",
            "entry_signal",
        ]
        present = [c for c in cols if c in df.columns]
        data = df[present].copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        rows = [(symbol, *row) for row in data.itertuples(index=False, name=None)]
        with self._conn() as conn:
            placeholders = ", ".join(["?"] * (len(present) + 1))
            col_names = ", ".join(["symbol"] + present)
            conn.executemany(
                f"INSERT OR REPLACE INTO signals ({col_names}) VALUES ({placeholders})",
                rows,
            )
            conn.commit()

    def read_signals(self, symbol: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp",
                (symbol,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_backtest_run(
        self,
        symbol: str,
        start: str,
        end: str,
        return_pct: Optional[float] = None,
        sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_runs
                (symbol, start, end, return_pct, sharpe, max_drawdown, params, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    start,
                    end,
                    return_pct,
                    sharpe,
                    max_drawdown,
                    json.dumps(params or {}),
                    created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def read_backtest_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM backtest_runs WHERE id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            result["params"] = json.loads(result["params"] or "{}")
            return result

    def log_backtest_trades(self, run_id: int, trades: pd.DataFrame) -> None:
        if trades.empty:
            return
        cols = ["entry_time", "exit_time", "size", "entry_price", "exit_price", "pnl", "return_pct"]
        data = trades[[c for c in cols if c in trades.columns]].copy()
        for col in ["entry_time", "exit_time"]:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        rows = [(run_id, *row) for row in data.itertuples(index=False, name=None)]
        with self._conn() as conn:
            col_names = ", ".join(["run_id"] + list(data.columns))
            placeholders = ", ".join(["?"] * (len(data.columns) + 1))
            conn.executemany(
                f"INSERT INTO backtest_trades ({col_names}) VALUES ({placeholders})",
                rows,
            )
            conn.commit()

    def read_backtest_trades(self, run_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY entry_time",
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_order(
        self,
        symbol: str,
        side: str,
        units: int,
        order_type: str,
        timestamp: datetime,
    ) -> int:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO orders (symbol, side, units, order_type, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, side, units, order_type, ts_str),
            )
            conn.commit()
            return cursor.lastrowid

    def log_fill(
        self,
        order_id: int,
        fill_price: float,
        timestamp: datetime,
        commission: Optional[float] = None,
    ) -> int:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fills (order_id, fill_price, commission, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (order_id, fill_price, commission, ts_str),
            )
            conn.commit()
            return cursor.lastrowid

    def read_fills(self, order_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if order_id is not None:
                cursor = conn.execute(
                    "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
                    (order_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM fills ORDER BY timestamp")
            return [dict(row) for row in cursor.fetchall()]

    def log_equity(self, timestamp: datetime, cash: float, equity: float) -> int:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO equity (timestamp, cash, equity) VALUES (?, ?, ?)",
                (ts_str, cash, equity),
            )
            conn.commit()
            return cursor.lastrowid

    def read_equity(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM equity ORDER BY timestamp")
            return [dict(row) for row in cursor.fetchall()]

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kronosbot.data.feed import DataFeed
from kronosbot.features.signals import SignalEngine

app = Flask(__name__)
app.secret_key = os.environ.get("KRONOSBOT_SECRET", "kronosbot-dev-secret")
app.jinja_env.globals.update(zip=zip)

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DB_PATH = Path("data/kronosbot.db")
DEFAULT_CASH = 10_000.0
DEFAULT_LOT_SIZE = 10_000

DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SUPPORTED_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]


# Flask app-level cache for the Kronos model so it is only loaded once per process.
_MODEL_CACHE: dict[str, Tuple[object, object]] = {}


# Lazy imports to avoid binding failures before backend modules are ready.
def _get_journal():
    from kronosbot.journal.store import Journal

    return Journal(str(DEFAULT_DB_PATH))


def _get_model(device: str = "cpu") -> Tuple[object, object]:
    """Return cached (model, tokenizer) or load and cache them."""
    if device not in _MODEL_CACHE:
        from kronosbot.model_loader import load_kronos_model

        _MODEL_CACHE[device] = load_kronos_model(device=device, cache_dir=DEFAULT_CACHE_DIR.parent / "hf_cache")
    return _MODEL_CACHE[device]


def _get_runner():
    from kronosbot.strategy.runner import BacktestRunner
    from kronosbot.strategy.kronos_strategy import KronosStrategy

    feed = DataFeed(cache_dir=DEFAULT_CACHE_DIR)
    model, tokenizer = _get_model("cpu")
    engine = SignalEngine(model=model, tokenizer=tokenizer)
    return BacktestRunner(feed, engine, KronosStrategy)


@app.context_processor
def inject_globals():
    return {
        "now": datetime.datetime.now(),
        "supported_symbols": SUPPORTED_SYMBOLS,
    }


@app.route("/")
def dashboard():
    try:
        journal = _get_journal()
        latest_equity = journal.read_equity(n=1)
        backtests = journal.read_backtest_runs(limit=5)
        for bt in backtests:
            bt["trade_count"] = len(journal.read_backtest_trades(bt["id"], limit=10_000))
    except Exception:
        latest_equity = []
        backtests = []
    open_positions: List[Dict[str, Any]] = []
    return render_template(
        "dashboard.html",
        latest_equity=latest_equity[0] if latest_equity else None,
        open_positions=open_positions,
        backtests=backtests,
    )


@app.route("/symbols")
def symbols():
    feed = DataFeed(cache_dir=DEFAULT_CACHE_DIR)
    cached = {}
    for symbol in SUPPORTED_SYMBOLS:
        path = feed._cache_path(symbol, "2024-01-01", "2026-01-01")
        cached[symbol] = path.exists()
    return render_template("symbols.html", cached=cached)


@app.route("/api/symbols/fetch", methods=["POST"])
def api_fetch_symbol():
    data = request.get_json() or {}
    symbol = data.get("symbol", "EURUSD")
    start = data.get("start", "2024-01-01")
    end = data.get("end", "2026-01-01")
    try:
        feed = DataFeed(cache_dir=DEFAULT_CACHE_DIR)
        df = feed.load(symbol, start=start, end=end)
        journal = _get_journal()
        journal.log_bars(symbol=symbol, df=df)
        return jsonify({"success": True, "rows": len(df)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/strategies")
def strategies():
    return render_template("strategies.html")


@app.route("/backtest", methods=["GET", "POST"])
def backtest():
    if request.method == "POST":
        symbol = request.form.get("symbol", "EURUSD")
        start = request.form.get("start", "2024-01-01")
        end = request.form.get("end", "2025-01-01")
        cash = float(request.form.get("cash", DEFAULT_CASH))
        try:
            return redirect(url_for("backtest_result", symbol=symbol, start=start, end=end, cash=cash))
        except Exception as e:
            flash(str(e), "error")
    return render_template("backtest.html")


import pandas as pd
import plotly
import plotly.graph_objects as go


@app.route("/backtest/result")
def backtest_result():
    symbol = request.args.get("symbol", "EURUSD")
    start = request.args.get("start", "2024-01-01")
    end = request.args.get("end", "2025-01-01")
    cash = float(request.args.get("cash", DEFAULT_CASH))

    try:
        runner = _get_runner()
        result = runner.run(
            symbol=symbol,
            start=start,
            end=end,
            cash=cash,
            commission=5.0,
        )
    except Exception as e:
        flash(str(e), "error")
        result = {"metrics": {}, "trades": [], "equity_curve": pd.DataFrame(), "plot_path": None}

    equity_chart = _equity_chart(result.get("equity_curve"))
    trade_count = len(result.get("trades", []))
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    if isinstance(trades, pd.DataFrame):
        trades = trades.rename(
            columns={
                "entry_time": "EntryTime",
                "entry_price": "EntryPrice",
                "exit_time": "ExitTime",
                "exit_price": "ExitPrice",
                "pnl": "PnL",
                "return_pct": "ReturnPct",
            }
        ).to_dict("records")

    return render_template(
        "backtest_result.html",
        symbol=symbol,
        start=start,
        end=end,
        metrics=metrics,
        trade_count=trade_count,
        trades=trades,
        equity_chart=equity_chart,
        plot_path=result.get("plot_path"),
    )


def _equity_chart(equity_curve) -> Optional[str]:
    if equity_curve is None or equity_curve.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["Equity"], mode="lines", name="Equity"))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/paper")
def paper():
    return render_template("paper.html")


@app.route("/journal")
def journal_view():
    try:
        journal = _get_journal()
        backtests = journal.read_backtest_runs(limit=100)
        # Gather trades across recent backtest runs
        trades = []
        for run in backtests[:10]:
            trades.extend(journal.read_backtest_trades(run["id"], limit=100))
        equity = journal.read_equity(n=100)
        signals = []
        for symbol in SUPPORTED_SYMBOLS:
            signals.extend(journal.read_signals(symbol, limit=25))
    except Exception:
        backtests = []
        trades = []
        equity = []
        signals = []
    return render_template(
        "journal.html", backtests=backtests, trades=trades, equity=equity, signals=signals
    )


@app.route("/settings")
def settings():
    return render_template(
        "settings.html",
        db_path=DEFAULT_DB_PATH,
        cache_dir=DEFAULT_CACHE_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
    )


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=True)

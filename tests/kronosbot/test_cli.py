"""Tests for the Kronos Bot CLI."""
import multiprocessing
import time
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from kronosbot.cli import cli


def _make_bars(periods: int = 250) -> pd.DataFrame:
    close = pd.Series([1.0 + i * 0.002 for i in range(periods)])
    high = close + 0.001
    low = close - 0.001
    volume = pd.Series([1_000.0] * periods)
    volume.iloc[-1] = 5_000.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=periods, freq="B"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class _MockFeed:
    def __init__(self, cache_dir=None):
        pass

    def load(self, symbol, start=None, end=None):
        return _make_bars(250)


class TestCLI:
    def test_backtest_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kronosbot.cli.DataFeed", _MockFeed)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "backtest",
                "EURUSD",
                "--start",
                "2023-01-01",
                "--end",
                "2023-12-31",
                "--output",
                str(tmp_path),
                "--db-path",
                str(tmp_path / "test.db"),
                "--cache-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Running backtest" in result.output
        assert "Return:" in result.output

    def test_paper_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kronosbot.cli.DataFeed", _MockFeed)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "paper",
                "EURUSD",
                "--start",
                "2023-01-01",
                "--end",
                "2023-12-31",
                "--db-path",
                str(tmp_path / "test.db"),
                "--cache-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Paper run complete" in result.output

    def test_alpha_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["alpha", "--help"])
        assert result.exit_code == 0, result.output
        assert "SYMBOL" in result.output
        assert "--start" in result.output

    def test_alpha_experiment_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["alpha-experiment", "--help"])
        assert result.exit_code == 0, result.output
        assert "SYMBOL" in result.output
        assert "--variants" in result.output

    def test_webui_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["webui", "--help"])
        assert result.exit_code == 0, result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--debug" in result.output


# Keep the server subprocess target at module level so multiprocessing can pickle it.
def _run_webui_server(port: int):
    from click.testing import CliRunner

    runner = CliRunner()
    runner.invoke(cli, ["webui", "--port", str(port), "--host", "127.0.0.1"])


class TestWebUICommand:
    @pytest.mark.timeout(60)
    def test_webui_server_starts_and_responds(self):
        port = 18765
        proc = multiprocessing.Process(target=_run_webui_server, args=(port,))
        proc.start()
        try:
            url = f"http://127.0.0.1:{port}/api/health"
            response_data = None
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url, timeout=1) as resp:
                        response_data = resp.read()
                        assert resp.status == 200
                        break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
            assert response_data is not None, "Web UI server did not respond"
            assert b'"status"' in response_data
            assert b'"ok"' in response_data
        finally:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

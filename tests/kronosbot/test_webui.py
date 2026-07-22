import pytest


def _import_app():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kronosbot.webui.app import app

    return app


@pytest.fixture
def client():
    app = _import_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestWebUI:
    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Kronos Bot" in resp.data

    def test_symbols_page(self, client):
        resp = client.get("/symbols")
        assert resp.status_code == 200
        assert b"EURUSD" in resp.data

    def test_strategies_page(self, client):
        resp = client.get("/strategies")
        assert resp.status_code == 200
        assert b"200-period SMA" in resp.data

    def test_backtest_page(self, client):
        resp = client.get("/backtest")
        assert resp.status_code == 200
        assert b"Run Backtest" in resp.data

    def test_paper_page(self, client):
        resp = client.get("/paper")
        assert resp.status_code == 200
        assert b"Paper Trading" in resp.data

    def test_journal_page(self, client):
        resp = client.get("/journal")
        assert resp.status_code == 200
        assert b"Journal" in resp.data

    def test_settings_page(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert b"IBKR" in resp.data or b"Settings" in resp.data

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

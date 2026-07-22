from pathlib import Path

import pytest


class TestPackageScaffold:
    def test_kronosbot_package_imports(self):
        import kronosbot

        assert kronosbot.__file__ is not None

    def test_submodule_files_exist(self):
        root = Path(__file__).parent.parent.parent / "kronosbot"
        expected = [
            "__init__.py",
            "data/__init__.py",
            "data/feed.py",
            "features/__init__.py",
            "features/signals.py",
            "strategy/__init__.py",
            "strategy/kronos_strategy.py",
            "strategy/runner.py",
            "broker/__init__.py",
            "broker/base.py",
            "broker/paper.py",
            "broker/ibkr.py",
            "journal/__init__.py",
            "journal/store.py",
            "cli.py",
            "webui/__init__.py",
            "webui/app.py",
        ]
        missing = [p for p in expected if not (root / p).exists()]
        assert not missing, f"Missing module files: {missing}"

"""Lazy, cached loader for the NeoQuasar Kronos HuggingFace model and tokenizer.

Usage:
    from kronosbot.model_loader import load_kronos_model

    model, tokenizer = load_kronos_model(device="cpu", cache_dir="data/hf_cache")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Type


# Module-level cache keyed by device (and optionally cache_dir) so the model
# is only loaded once per Python process. This is important because the model
# is ~100MB and initialization is expensive.
_MODEL_CACHE: dict[str, Tuple[object, object]] = {}

_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
_MODEL_REPO = "NeoQuasar/Kronos-small"
_MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"

# Import the model classes at module load time so callers and tests can
# monkeypatch the module attributes if needed. If the import fails, the
# function below will raise a RuntimeError.
try:  # pragma: no cover - environment-dependent
    import sys

    _repo_root = Path(__file__).parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from model import Kronos, KronosTokenizer
except Exception as _import_exc:  # pragma: no cover - environment-dependent
    Kronos = None
    KronosTokenizer = None
    _IMPORT_ERROR = _import_exc
else:
    _IMPORT_ERROR = None


def load_kronos_model(
    device: str = "cpu",
    cache_dir: Optional[str | Path] = None,
) -> Tuple[object, object]:
    """Load the Kronos model and tokenizer once, cache in module dict, and return both.

    Args:
        device: Target torch device string, e.g. "cpu" or "cuda:0".
        cache_dir: Optional directory for HuggingFace Hub downloads. Defaults to
            ``data/hf_cache`` relative to the repository root.

    Returns:
        A (model, tokenizer) tuple. Both are in ``eval`` mode.

    Raises:
        RuntimeError: if a required dependency is missing or model loading fails.
    """
    if Kronos is None or KronosTokenizer is None:
        raise RuntimeError(
            "Kronos model module could not be imported. "
            "Ensure the repository is on PYTHONPATH and dependencies (torch, einops) are installed."
        ) from _IMPORT_ERROR

    try:
        from huggingface_hub import __version__  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "huggingface_hub is required to download Kronos models."
        ) from exc

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "hf_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir = cache_dir.expanduser().resolve()

    # Use the canonical device string as the cache key. This also ensures that
    # passing the same device from different call sites reuses the same objects.
    cache_key = f"{device}:{cache_dir}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        tokenizer = KronosTokenizer.from_pretrained(
            _TOKENIZER_REPO,
            revision=_TOKENIZER_REVISION,
            cache_dir=str(cache_dir),
        )
        model = Kronos.from_pretrained(
            _MODEL_REPO,
            revision=_MODEL_REVISION,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"Failed to load Kronos model from HuggingFace Hub: {exc}"
        ) from exc

    tokenizer.eval()
    model.eval()

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(tokenizer, "to"):
        tokenizer = tokenizer.to(device)

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def clear_kronos_model_cache() -> None:
    """Clear the module-level model cache. Useful in tests or after changing devices."""
    _MODEL_CACHE.clear()

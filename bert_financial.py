"""
pipeline.inference.bert_financial
FinBERT sentiment classifier for financial text.

Checkpoint: ProsusAI/finbert
Target corpus: Stocktwits, financial forums, crypto discussions.

Public interface (identical to bert_general):
    score_batch(texts, batch_size) -> list[dict]
    load_model()  -> None
    is_loaded()   -> bool
    mock_score_batch(texts) -> list[dict]

All configuration read from config/settings.py — nothing hardcoded.
"""

from __future__ import annotations

from config.settings import MODEL_CONFIGS, SENTIMENT_MODEL
from pipeline.inference._engine import InferenceEngine, mock_score_batch  # noqa: F401

_cfg = MODEL_CONFIGS["finbert"]
_engine = InferenceEngine(
    checkpoint=_cfg["hf_model_id"],
    default_batch_size=_cfg["batch_size"],
    max_length=_cfg["max_length"],
)


def load_model() -> None:
    """Pre-load FinBERT into memory.  Idempotent."""
    _engine.load_model()


def is_loaded() -> bool:
    """True if FinBERT weights are resident in memory."""
    return _engine.is_loaded()


def score_batch(
    texts: list[str],
    batch_size: int | None = None,
) -> list[dict[str, float | str]]:
    """Score sentences using FinBERT or mock mode.

    When ``SENTIMENT_MODEL == 'mock'``, routes to deterministic
    mock_score_batch() without loading any model.
    """
    if SENTIMENT_MODEL == "mock":
        return mock_score_batch(texts)
    return _engine.score_batch(texts, batch_size)

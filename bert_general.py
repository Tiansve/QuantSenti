"""
pipeline.inference.bert_general
RoBERTa-large sentiment classifier for general social / sports text.

Checkpoint: cardiffnlp/twitter-roberta-large-sentiment-latest
    (~1.4 GB, RoBERTa-large, 355M params)
Target corpus: Reddit sports, general social media discussions.

Public interface (identical to bert_financial):
    score_batch(texts, batch_size) -> list[dict]
    load_model()  -> None
    is_loaded()   -> bool
    mock_score_batch(texts) -> list[dict]

All configuration read from config/settings.py — nothing hardcoded.
"""

from __future__ import annotations

from config.settings import MODEL_CONFIGS, SENTIMENT_MODEL
from pipeline.inference._engine import InferenceEngine, mock_score_batch  # noqa: F401

_cfg = MODEL_CONFIGS["twitter-roberta"]
_engine = InferenceEngine(
    checkpoint=_cfg["hf_model_id"],
    default_batch_size=_cfg["batch_size"],
    max_length=_cfg["max_length"],
)


def load_model() -> None:
    """Pre-load RoBERTa-large into memory.  Idempotent."""
    _engine.load_model()


def is_loaded() -> bool:
    """True if RoBERTa-large weights are resident in memory."""
    return _engine.is_loaded()


def score_batch(
    texts: list[str],
    batch_size: int | None = None,
) -> list[dict[str, float | str]]:
    """Score sentences using RoBERTa-large or mock mode.

    When ``SENTIMENT_MODEL == 'mock'``, routes to deterministic
    mock_score_batch() without loading any model.
    """
    if SENTIMENT_MODEL == "mock":
        return mock_score_batch(texts)
    return _engine.score_batch(texts, batch_size)

"""
pipeline/aggregation/index_calculator.py
QuantSenti — Sentiment Index Aggregation

Pure-computation module. Receives a scored DataFrame from the inference layer,
aggregates sentiment into a single index value, and returns a structured dict.
No I/O, no database access, no model inference.

Public API:
    calculate_index(df, entity, window_end, window_seconds, **kwargs) -> dict
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.settings import (
    AGGREGATION_ALPHA,
    AGGREGATION_CONFIDENCE_HIGH_THRESH,
    AGGREGATION_CONFIDENCE_MED_THRESH,
    AGGREGATION_HALF_LIFE_RATIO,
    AGGREGATION_MIN_SENTENCES,
)

log = logging.getLogger(__name__)

# Minimal English stop words — no external dependency
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "will", "from",
    "they", "this", "that", "with", "what", "when", "where", "which", "who",
    "how", "than", "them", "then", "into", "over", "such", "just", "also",
    "its", "about", "would", "could", "should", "their", "there", "these",
    "those", "other", "some", "very", "more", "much", "most", "does", "did",
    "get", "got", "going", "like", "still", "even", "being", "think", "know",
    "really", "right", "don", "didn", "isn", "doesn", "won", "couldn", "shouldn",
})


# ── Cold-start template ──────────────────────────────────────────────────────

def _cold_start_result(
    entity: str,
    window_end: datetime,
    window_seconds: int,
    sentence_count: int = 0,
    pos_count: int = 0,
    neg_count: int = 0,
    neu_count: int = 0,
) -> dict[str, Any]:
    """Return a neutral result when data is insufficient."""
    return {
        "entity": entity,
        "window_end": window_end.isoformat(),
        "window_seconds": window_seconds,
        "index_value": 50.0,
        "confidence": "low",
        "sentence_count": sentence_count,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "top_pos_keywords": [],
        "top_neg_keywords": [],
        "is_cold_start": True,
    }


# ── Step 1: Sentinel filtering ───────────────────────────────────────────────

def _filter_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where inference failed (neu==1, pos==0, neg==0)."""
    mask = (df["neu"] == 1.0) & (df["pos"] == 0.0) & (df["neg"] == 0.0)
    n_removed = mask.sum()
    if n_removed:
        log.info("Sentinel rows removed: %d", n_removed)
    return df[~mask].copy()


# ── Step 2: Time decay ───────────────────────────────────────────────────────

def _apply_time_decay(
    df: pd.DataFrame,
    window_end: datetime,
    window_seconds: int,
    half_life_ratio: float,
) -> pd.DataFrame:
    """Compute exponential decay weight for each row based on created_at."""
    half_life = window_seconds * half_life_ratio
    ln2 = math.log(2)

    def _parse_and_decay(created_at_raw: Any) -> float:
        try:
            if isinstance(created_at_raw, pd.Timestamp):
                dt = created_at_raw.to_pydatetime()
            elif isinstance(created_at_raw, datetime):
                dt = created_at_raw
            else:
                dt = datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00"))

            # Ensure timezone-aware for subtraction
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            delta_seconds = (window_end - dt).total_seconds()

            # Edge case: future timestamp — clip decay to 1.0
            if delta_seconds < 0:
                return 1.0

            return math.exp(-ln2 * delta_seconds / half_life) if half_life > 0 else 1.0

        except (ValueError, TypeError, OverflowError) as exc:
            log.warning("created_at parse failure (%s): %r", exc, created_at_raw)
            return 0.0

    df["decay"] = df["created_at"].apply(_parse_and_decay)
    return df


# ── Step 3: Core index computation ───────────────────────────────────────────

def _compute_index(df: pd.DataFrame, alpha: float) -> float:
    """Weighted sentiment aggregation → index in [0, 100]."""
    raw_score = (df["decay"] * df["pos"]).sum() - (df["decay"] * df["neg"]).sum()
    normalizer = df["decay"].sum() + alpha

    if normalizer <= 0:
        log.error("normalizer == 0 guard triggered — returning neutral")
        return 50.0

    net = raw_score / normalizer                    # (-1, +1)
    index_value = round((net + 1) / 2 * 100, 2)    # (0, 100)
    return float(max(0.0, min(100.0, index_value))) # clip


# ── Step 4: Confidence assessment ─────────────────────────────────────────────

def _assess_confidence(
    sentence_count: int,
    min_sentences: int,
    high_thresh: int,
    med_thresh: int,
) -> tuple[str, bool]:
    """Return (confidence_level, is_cold_start)."""
    if sentence_count < min_sentences:
        return "low", True
    if sentence_count >= high_thresh:
        return "high", False
    if sentence_count >= med_thresh:
        return "medium", False
    return "low", False


# ── Step 5: Keyword extraction ────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")

def _extract_keywords(df: pd.DataFrame, label: str, top_n: int = 5) -> list[str]:
    """Top-N keywords from rows matching the given label."""
    subset = df.loc[df["label"] == label, "text"]
    if subset.empty:
        return []

    counter: Counter[str] = Counter()
    for text in subset:
        tokens = _TOKEN_RE.findall(str(text).lower())
        counter.update(t for t in tokens if t not in _STOP_WORDS)

    return [word for word, _ in counter.most_common(top_n)]


# ── Public entry point ────────────────────────────────────────────────────────

def calculate_index(
    df: pd.DataFrame,
    entity: str,
    window_end: datetime,
    window_seconds: int,
    *,
    alpha: float = AGGREGATION_ALPHA,
    half_life_ratio: float = AGGREGATION_HALF_LIFE_RATIO,
    min_sentences: int = AGGREGATION_MIN_SENTENCES,
    confidence_high_thresh: int = AGGREGATION_CONFIDENCE_HIGH_THRESH,
    confidence_med_thresh: int = AGGREGATION_CONFIDENCE_MED_THRESH,
) -> dict[str, Any]:
    """Aggregate a scored DataFrame into a single sentiment index result.

    Parameters
    ----------
    df : DataFrame with columns [text, pos, neg, neu, label, likes_count, created_at, source]
    entity : Symbol string (e.g. 'BTC.X')
    window_end : UTC datetime anchor for time-decay calculation
    window_seconds : Window duration in seconds
    **kwargs : Override any config constant (alpha, half_life_ratio, etc.)

    Returns
    -------
    dict with fixed schema — see Output Dict in design brief.
    """
    # Ensure window_end is timezone-aware
    if window_end.tzinfo is None:
        window_end = window_end.replace(tzinfo=timezone.utc)

    # Empty input → cold start
    if df is None or df.empty:
        log.warning("Cold-start triggered: empty DataFrame for entity=%s", entity)
        return _cold_start_result(entity, window_end, window_seconds)

    # Step 1 — Filter sentinels
    clean = _filter_sentinels(df)
    if clean.empty:
        log.warning("Cold-start triggered: all rows are sentinels for entity=%s", entity)
        return _cold_start_result(entity, window_end, window_seconds)

    # Label counts
    labels = clean["label"].str.lower()
    pos_count = int((labels == "positive").sum())
    neg_count = int((labels == "negative").sum())
    neu_count = int((labels == "neutral").sum())
    sentence_count = len(clean)

    # Step 4 — Confidence (computed early; needed for cold-start gate)
    confidence, is_cold_start = _assess_confidence(
        sentence_count, min_sentences, confidence_high_thresh, confidence_med_thresh,
    )

    # Step 5 — Cold-start short circuit
    if is_cold_start:
        log.warning("Cold-start triggered: %d sentences < min=%d for entity=%s",
                     sentence_count, min_sentences, entity)
        return _cold_start_result(
            entity, window_end, window_seconds,
            sentence_count, pos_count, neg_count, neu_count,
        )

    # Step 2 — Time decay
    clean = _apply_time_decay(clean, window_end, window_seconds, half_life_ratio)

    # Step 3 — Compute index
    index_value = _compute_index(clean, alpha)

    # Step 6 — Keywords
    top_pos_keywords = _extract_keywords(clean, "positive")
    top_neg_keywords = _extract_keywords(clean, "negative")

    log.info("Calculation complete: entity=%s index=%.2f count=%d",
             entity, index_value, sentence_count)

    return {
        "entity": entity,
        "window_end": window_end.isoformat(),
        "window_seconds": window_seconds,
        "index_value": index_value,
        "confidence": confidence,
        "sentence_count": sentence_count,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "top_pos_keywords": top_pos_keywords,
        "top_neg_keywords": top_neg_keywords,
        "is_cold_start": False,
    }

"""
pipeline.inference._engine
Shared inference engine for all BERT-family sentiment models.

Both bert_financial and bert_general delegate to this module.
The only difference is the checkpoint name and default batch size,
which are passed in at construction time.

Design decisions:
  - Module-level lazy singleton: model loaded on first score_batch() or
    explicit load_model() call.  Idempotent — safe to call repeatedly.
  - Device-agnostic: auto-selects CUDA if available, falls back to CPU.
  - Label normalisation via id2label — never hardcoded index positions.
  - Failed batches emit neutral sentinel scores (neu=1.0) instead of
    aborting the entire pipeline run.
  - GDPR: sentence text never logged above DEBUG level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Neutral sentinel returned on errors or empty strings
_NEUTRAL: dict[str, float | str] = {
    "pos": 0.33, "neg": 0.33, "neu": 0.34, "label": "neutral",
}

# Error sentinel (neu=1.0 flags downstream filtering)
_ERROR_NEUTRAL: dict[str, float | str] = {
    "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "neutral",
}

# Non-English sentinel — same shape as NEUTRAL so downstream code needs no changes
_NON_ENGLISH: dict[str, float | str] = {
    "pos": 0.33, "neg": 0.33, "neu": 0.34, "label": "neutral",
}

# Characters up to U+024F cover ASCII + Latin Extended A/B (sufficient for English)
_MAX_NON_LATIN_CODE_POINT = 0x024F
_NON_ENGLISH_RATIO_THRESHOLD = 0.25  # >25 % non-Latin alpha chars → skip


def _is_english(text: str) -> bool:
    """Return True when text is dominantly Latin/ASCII (English-like).

    Only alphabetic characters are examined so that digits, currency symbols,
    and punctuation do not skew the ratio.  Texts with no alphabetic content
    are passed through unchanged (handled by the empty-string path upstream).
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True  # numbers/symbols only — let tokeniser decide
    non_latin = sum(1 for ch in letters if ord(ch) > _MAX_NON_LATIN_CODE_POINT)
    return (non_latin / len(letters)) <= _NON_ENGLISH_RATIO_THRESHOLD


class InferenceEngine:
    """Reusable inference engine wrapping a HuggingFace 3-class sentiment model."""

    def __init__(self, checkpoint: str, default_batch_size: int, max_length: int):
        self._checkpoint = checkpoint
        self._default_batch_size = default_batch_size
        self._max_length = max_length
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._label_map: dict[int, str] = {}  # id → lowercase label
        self._device: str = "cpu"  # resolved in load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load checkpoint into memory.  Idempotent — no-op if already loaded."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model: %s  device=%s ...", self._checkpoint, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint)
        self._model.to(self._device)
        self._model.eval()

        # Build normalised label map from checkpoint config
        self._label_map = {
            idx: lbl.lower()
            for idx, lbl in self._model.config.id2label.items()
        }
        logger.info("Model loaded: %s  id2label=%s", self._checkpoint, self._label_map)

    def is_loaded(self) -> bool:
        """Health check — True if model is in memory and ready."""
        return self._model is not None

    def score_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[dict[str, float | str]]:
        """Score a list of sentences and return per-sentence probability dicts.

        Args:
            texts:      GDPR-scrubbed sentences.
            batch_size: Override default chunk size.

        Returns:
            List of dicts ``{"pos", "neg", "neu", "label"}`` in input order.
        """
        if not texts:
            return []

        self.load_model()  # lazy-load on first call
        bs = batch_size or self._default_batch_size

        results: list[dict[str, float | str]] = []
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            results.extend(self._score_chunk(chunk))

        return results

    # ── Internals ─────────────────────────────────────────────────────────────

    def _score_chunk(
        self, chunk: list[str]
    ) -> list[dict[str, float | str]]:
        """Run forward pass on a single batch chunk."""
        import torch

        assert self._model is not None and self._tokenizer is not None

        # Replace empty / non-English strings with a placeholder so the
        # tokeniser doesn't choke.  Non-English texts are flagged and skipped
        # before inference to avoid garbage scores.
        cleaned = []
        empty_mask: list[bool] = []
        non_english_mask: list[bool] = []
        for t in chunk:
            is_empty = not t or not t.strip()
            is_non_english = (not is_empty) and (not _is_english(t))
            empty_mask.append(is_empty)
            non_english_mask.append(is_non_english)
            if is_non_english:
                logger.debug("Non-English text skipped (returning neutral sentinel).")
            cleaned.append("neutral" if (is_empty or is_non_english) else t)

        try:
            inputs = self._tokenizer(
                cleaned,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        except Exception:
            logger.exception(
                "Inference failed for batch of %d sentences — returning neutral sentinels.",
                len(chunk),
            )
            return [dict(_ERROR_NEUTRAL) for _ in chunk]

        results: list[dict[str, float | str]] = []
        for i, prob_row in enumerate(probs):
            if empty_mask[i]:
                results.append(dict(_NEUTRAL))
                continue
            if non_english_mask[i]:
                results.append(dict(_NON_ENGLISH))
                continue

            scores: dict[str, float] = {}
            for idx, label in self._label_map.items():
                # Normalise variant label names to our canonical keys
                key = _canonical_key(label)
                scores[key] = round(float(prob_row[idx]), 6)

            best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
            results.append({
                "pos": scores.get("pos", 0.0),
                "neg": scores.get("neg", 0.0),
                "neu": scores.get("neu", 0.0),
                "label": _canonical_label(best_label),
            })

        return results


# ── Mock mode ─────────────────────────────────────────────────────────────────

def mock_score_batch(texts: list[str]) -> list[dict[str, float | str]]:
    """Deterministic mock scoring for CI / unit tests.

    Rules (from spec):
      - even-length text  → positive leaning
      - odd-length text   → negative leaning
      - empty string      → neutral
    """
    if not texts:
        return []

    results: list[dict[str, float | str]] = []
    for text in texts:
        if not text:
            results.append({"pos": 0.33, "neg": 0.33, "neu": 0.34, "label": "neutral"})
        elif len(text) % 2 == 0:
            results.append({"pos": 0.75, "neg": 0.10, "neu": 0.15, "label": "positive"})
        else:
            results.append({"pos": 0.10, "neg": 0.72, "neu": 0.18, "label": "negative"})
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

_KEY_ALIASES: dict[str, str] = {
    "positive": "pos",
    "negative": "neg",
    "neutral":  "neu",
    "pos":      "pos",
    "neg":      "neg",
    "neu":      "neu",
}

_LABEL_ALIASES: dict[str, str] = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
}


def _canonical_key(label: str) -> str:
    """Map any label variant to 'pos' / 'neg' / 'neu'."""
    return _KEY_ALIASES.get(label.lower(), label.lower()[:3])


def _canonical_label(key: str) -> str:
    """Map any key variant to 'positive' / 'negative' / 'neutral'."""
    return _LABEL_ALIASES.get(key, key)

"""Clinical case semantic encoder using sentence-transformers.

Lazy-loads all-MiniLM-L6-v2 on first call (singleton). Produces 384-dim
normalized embeddings for case representations and individual evidence items.
"""
from __future__ import annotations

import numpy as np


_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


class ClinicalEmbedder:
    """Encodes clinical text into 384-dim normalized vectors."""

    def encode(self, text: str) -> np.ndarray:
        model = _get_model()
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vec, dtype=np.float32)

    def encode_case(self, findings_summary: str, top_candidate: str) -> np.ndarray:
        combined = f"[CASE] {findings_summary} [DX] {top_candidate}"
        return self.encode(combined)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        model = _get_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

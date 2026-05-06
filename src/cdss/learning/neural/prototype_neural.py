"""Embedding-based syndrome prototype centroids with EMA updates.

Each syndrome accumulates a centroid in embedding space. On correct diagnosis,
the centroid drifts toward the new case (exponential moving average, α=0.05).
On new case, cosine similarity to centroid yields a ±0.08 score boost.

Persists centroids to disk as a numpy npz archive.
"""
from __future__ import annotations

import pathlib
import threading

import numpy as np

import logging
logger = logging.getLogger(__name__)

_DEFAULT_PATH = pathlib.Path("data/cdss/learning/syndrome_centroids.npz")
_ALPHA = 0.05
_lock = threading.Lock()


class NeuralPrototypeMemory:
    def __init__(self, path: pathlib.Path = _DEFAULT_PATH):
        self._path = path
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        with _lock:
            if self._loaded:
                return
            if self._path.exists():
                try:
                    data = np.load(str(self._path), allow_pickle=True)
                    self._centroids = {k: data[k] for k in data.files if not k.endswith("__count")}
                    self._counts = {
                        k.replace("__count", ""): int(data[k])
                        for k in data.files if k.endswith("__count")
                    }
                except Exception:  # noqa: BLE001
                    logger.debug("swallowed exception", exc_info=True)
                    pass
            self._loaded = True

    def update_centroid(self, syndrome_id: str, embedding: np.ndarray, outcome_correct: bool):
        """EMA-update centroid only on confirmed correct diagnoses."""
        if not outcome_correct:
            return
        self._ensure_loaded()
        emb = embedding.astype(np.float32)
        with _lock:
            if syndrome_id not in self._centroids:
                self._centroids[syndrome_id] = emb.copy()
                self._counts[syndrome_id] = 1
            else:
                self._centroids[syndrome_id] = (
                    (1.0 - _ALPHA) * self._centroids[syndrome_id] + _ALPHA * emb
                )
                self._counts[syndrome_id] = self._counts.get(syndrome_id, 0) + 1
            self._save_unsafe()

    def similarity_boost(self, syndrome_id: str, case_embedding: np.ndarray) -> float:
        """Cosine similarity to learned centroid → score delta in [-0.08, +0.08]."""
        self._ensure_loaded()
        centroid = self._centroids.get(syndrome_id)
        if centroid is None:
            return 0.0
        emb = case_embedding.astype(np.float32)
        norm_c = np.linalg.norm(centroid)
        norm_e = np.linalg.norm(emb)
        if norm_c < 1e-9 or norm_e < 1e-9:
            return 0.0
        cos = float(np.dot(centroid / norm_c, emb / norm_e))
        return max(-0.08, min(0.08, cos * 0.08))

    def _save_unsafe(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {k: v for k, v in self._centroids.items()}
        save_dict.update({f"{k}__count": np.array(v) for k, v in self._counts.items()})
        np.savez(str(self._path), **save_dict)

    @property
    def known_syndromes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._centroids.keys())

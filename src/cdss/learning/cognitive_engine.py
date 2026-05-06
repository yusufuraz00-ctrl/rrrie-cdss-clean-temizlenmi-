"""Cognitive Learning Engine — single entry point for all neural learning.

Singleton pattern: one instance lives for the server lifetime, shared across
all requests via get_cognitive_engine(). Thread-safe components throughout.

Pipeline:
  encode_case()     → 384-dim embedding (sentence-transformers)
  query_similar()   → k=5 nearest past cases (hnswlib)
  score_candidate() → neural delta [-0.20, +0.20] for arbitration
  record_case()     → store embedding at case end (outcome=None initially)
  record_outcome()  → update case outcome + train MLP + update centroid

When sentence-transformers or hnswlib are not installed, engine degrades
gracefully: score_candidate() returns 0.0, no crash.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .neural.embedder import ClinicalEmbedder
from .neural.case_memory import NeuralCaseMemory
from .neural.predictor import DiagnosticPredictor
from .neural.prototype_neural import NeuralPrototypeMemory

import logging
logger = logging.getLogger(__name__)

_instance: Optional["CognitiveLearningEngine"] = None
_instance_lock = threading.Lock()


class CognitiveLearningEngine:
    def __init__(self):
        self._embedder = ClinicalEmbedder()
        self._case_memory = NeuralCaseMemory()
        self._predictor = DiagnosticPredictor()
        self._prototype = NeuralPrototypeMemory()
        # session cache: case_id → (embedding, memory_idx)
        self._session: dict[str, tuple[np.ndarray, int]] = {}
        self._session_lock = threading.Lock()

    # ──────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────

    def encode_case(self, findings_summary: str, top_candidate: str) -> Optional[np.ndarray]:
        try:
            return self._embedder.encode_case(findings_summary, top_candidate)
        except Exception:
            return None

    # ──────────────────────────────────────────
    # Inference (called during arbitration)
    # ──────────────────────────────────────────

    def score_candidate(
        self,
        case_embedding: Optional[np.ndarray],
        candidate_label: str,
    ) -> float:
        """Returns neural score delta in [-0.20, +0.20].

        Combines:
          - k-NN outcome-weighted delta from similar past cases (±0.15 max)
          - Prototype centroid cosine similarity boost (±0.08 max)
          - MLP predicted correctness delta (±0.075 max)
        Total clamped to [-0.20, +0.20].
        """
        if case_embedding is None:
            return 0.0
        try:
            similar = self._case_memory.query(case_embedding, k=5)
            knn_delta = self._case_memory.outcome_weighted_score(similar, candidate_label)
            proto_delta = self._prototype.similarity_boost(candidate_label, case_embedding)
            mlp_p = self._predictor.predict(case_embedding)
            mlp_delta = (mlp_p - 0.5) * 0.15
            total = knn_delta + proto_delta + mlp_delta
            return max(-0.20, min(0.20, total))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────
    # Recording (called at case completion)
    # ──────────────────────────────────────────

    def record_case(
        self,
        case_id: str,
        findings_summary: str,
        top_candidate: str,
    ) -> None:
        """Store case embedding before outcome is known. Called at ACTION_PLAN end."""
        try:
            emb = self._embedder.encode_case(findings_summary, top_candidate)
            mem_idx = self._case_memory.add(emb, top_candidate, outcome=None)
            with self._session_lock:
                self._session[case_id] = (emb, mem_idx)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

    def record_outcome(
        self,
        case_id: str,
        findings_summary: str,
        top_candidate: str,
        outcome_correct: bool,
    ) -> None:
        """Called after user feedback. Updates vector store, MLP, and prototype centroid."""
        try:
            with self._session_lock:
                cached = self._session.get(case_id)

            if cached is not None:
                emb, mem_idx = cached
            else:
                emb = self._embedder.encode_case(findings_summary, top_candidate)
                mem_idx = self._case_memory.add(emb, top_candidate, outcome=None)

            # Update stored outcome
            self._case_memory.update_outcome(mem_idx, outcome_correct)

            # Train MLP: run predict to populate activation cache, then backprop
            self._predictor.predict(emb)
            self._predictor.update(1.0 if outcome_correct else 0.0)

            # Update prototype centroid
            self._prototype.update_centroid(top_candidate, emb, outcome_correct)

            # Clean session entry
            with self._session_lock:
                self._session.pop(case_id, None)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

    # ──────────────────────────────────────────
    # Stats (for Learning Dashboard)
    # ──────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "cases_in_memory": self._case_memory.size,
            "known_syndromes": len(self._prototype.known_syndromes),
            "mlp_updates": getattr(self._predictor, "update_count", 0),
        }


def get_cognitive_engine() -> CognitiveLearningEngine:
    """Return the process-level singleton CognitiveLearningEngine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CognitiveLearningEngine()
    return _instance

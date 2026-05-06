"""hnswlib-backed vector store for clinical case memory.

Stores (embedding, label, outcome) triples. On new case arrival,
retrieves k nearest past cases and computes an outcome-weighted score
delta that informs differential diagnosis scoring in arbitration.

Persists index + metadata to disk across sessions.
"""
from __future__ import annotations

import json
import pathlib
import threading
from typing import Optional

import numpy as np

_DIM = 384
_MAX_ELEMENTS = 50_000
_DEFAULT_STORE = pathlib.Path("data/cdss/learning/neural_case_memory")

_lock = threading.Lock()


class NeuralCaseMemory:
    def __init__(self, store_path: pathlib.Path = _DEFAULT_STORE, dim: int = _DIM):
        self._dim = dim
        self._store_path = store_path
        self._index_path = store_path / "index.bin"
        self._meta_path = store_path / "metadata.jsonl"
        self._metadata: list[dict] = []
        self._index = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        with _lock:
            if self._loaded:
                return
            try:
                import hnswlib
            except ImportError:
                self._index = None
                self._loaded = True
                return
            self._store_path.mkdir(parents=True, exist_ok=True)
            idx = hnswlib.Index(space="cosine", dim=self._dim)
            if self._index_path.exists() and self._meta_path.exists():
                idx.load_index(str(self._index_path), max_elements=_MAX_ELEMENTS)
                self._metadata = [
                    json.loads(line)
                    for line in self._meta_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                idx.init_index(max_elements=_MAX_ELEMENTS, ef_construction=200, M=16)
            idx.set_ef(50)
            self._index = idx
            self._loaded = True

    def add(self, embedding: np.ndarray, label: str, outcome: Optional[bool] = None) -> int:
        self._ensure_loaded()
        if self._index is None:
            return -1
        with _lock:
            idx = len(self._metadata)
            self._index.add_items(embedding.reshape(1, -1).astype(np.float32), [idx])
            record = {"label": label, "outcome": outcome}
            self._metadata.append(record)
            with self._meta_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if idx % 10 == 0:
                self._index.save_index(str(self._index_path))
        return idx

    def update_outcome(self, idx: int, outcome: bool):
        """Update the outcome of a previously stored case (called after feedback)."""
        self._ensure_loaded()
        if idx < 0 or idx >= len(self._metadata):
            return
        with _lock:
            self._metadata[idx]["outcome"] = outcome
            lines = [json.dumps(m) for m in self._metadata]
            self._meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def query(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        self._ensure_loaded()
        if self._index is None or len(self._metadata) == 0:
            return []
        k = min(k, len(self._metadata))
        ids, distances = self._index.knn_query(embedding.reshape(1, -1).astype(np.float32), k=k)
        results = []
        for i, d in zip(ids[0], distances[0]):
            meta = self._metadata[i].copy()
            meta["distance"] = float(d)
            meta["idx"] = int(i)
            results.append(meta)
        return results

    def outcome_weighted_score(self, similar_cases: list[dict], candidate_label: str) -> float:
        """Score delta in [-0.15, +0.15] from k-NN outcome history.

        Weights by 1/(1+distance). Only cases where outcome is known AND
        label matches contribute. Neutral when no signal exists.
        """
        pos = 0.0
        neg = 0.0
        total_w = 0.0
        for case in similar_cases:
            if case["label"] != candidate_label or case["outcome"] is None:
                continue
            w = 1.0 / (1.0 + case["distance"])
            if case["outcome"]:
                pos += w
            else:
                neg += w
            total_w += w
        if total_w == 0.0:
            return 0.0
        raw = (pos - neg) / total_w
        return max(-0.15, min(0.15, raw * 0.15))

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._metadata)

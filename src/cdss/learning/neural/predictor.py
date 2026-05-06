"""Online NumPy MLP for diagnosis correctness prediction.

Architecture: [384] → [256] → [128] → [1 sigmoid]
Training: online SGD with momentum, one update per feedback event.
Persists weights to disk via pickle.

Input: 384-dim case embedding (normalized cosine space).
Output: P(diagnosis_correct) in [0, 1].
Score delta applied to arbitration: (P - 0.5) * 0.15 → max ±0.075.
"""
from __future__ import annotations

import pathlib
import pickle
import threading
from typing import Optional

import numpy as np

import logging
logger = logging.getLogger(__name__)

_INPUT_DIM = 384
_H1 = 256
_H2 = 128
_DEFAULT_PATH = pathlib.Path("data/cdss/learning/diagnostic_predictor.pkl")
_lock = threading.Lock()


class DiagnosticPredictor:
    def __init__(self, lr: float = 0.001, path: pathlib.Path = _DEFAULT_PATH):
        self._path = path
        self._lr = lr
        self._initialized = False
        # forward-pass cache (needed by update())
        self._x: Optional[np.ndarray] = None
        self._a1: Optional[np.ndarray] = None
        self._a2: Optional[np.ndarray] = None
        self._out: Optional[np.ndarray] = None

    def _init_weights(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((_INPUT_DIM, _H1)).astype(np.float32) * 0.01
        self.b1 = np.zeros(_H1, dtype=np.float32)
        self.W2 = rng.standard_normal((_H1, _H2)).astype(np.float32) * 0.01
        self.b2 = np.zeros(_H2, dtype=np.float32)
        self.W3 = rng.standard_normal((_H2, 1)).astype(np.float32) * 0.01
        self.b3 = np.zeros(1, dtype=np.float32)
        # momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb1 = np.zeros_like(self.b1)
        self.vb2 = np.zeros_like(self.b2)
        self.vb3 = np.zeros_like(self.b3)
        self.update_count = 0

    def _ensure_ready(self):
        if self._initialized:
            return
        with _lock:
            if self._initialized:
                return
            if self._path.exists():
                try:
                    data = pickle.loads(self._path.read_bytes())
                    self.__dict__.update(data)
                    self._initialized = True
                    return
                except Exception:  # noqa: BLE001
                    logger.debug("swallowed exception", exc_info=True)
                    pass
            self._init_weights()
            self._initialized = True

    def predict(self, x: np.ndarray) -> float:
        """Forward pass. Caches activations for update()."""
        self._ensure_ready()
        x = x.astype(np.float32).ravel()[:_INPUT_DIM]
        if x.shape[0] < _INPUT_DIM:
            x = np.pad(x, (0, _INPUT_DIM - x.shape[0]))
        self._x = x
        self._a1 = np.tanh(x @ self.W1 + self.b1)
        self._a2 = np.tanh(self._a1 @ self.W2 + self.b2)
        self._out = 1.0 / (1.0 + np.exp(-(self._a2 @ self.W3 + self.b3)))
        return float(self._out[0])

    def update(self, y: float):
        """Backprop one step. Must call predict() first."""
        if self._out is None:
            return
        with _lock:
            momentum = 0.9
            # Output layer
            d3 = (self._out - y).astype(np.float32)
            dW3 = self._a2[:, None] * d3
            db3 = d3
            # Hidden layer 2
            d2 = ((d3 @ self.W3.T) * (1.0 - self._a2 ** 2)).astype(np.float32)
            dW2 = self._a1[:, None] * d2
            db2 = d2
            # Hidden layer 1
            d1 = ((d2 @ self.W2.T) * (1.0 - self._a1 ** 2)).astype(np.float32)
            dW1 = self._x[:, None] * d1
            db1 = d1
            # Momentum SGD
            self.vW3 = momentum * self.vW3 - self._lr * dW3
            self.vW2 = momentum * self.vW2 - self._lr * dW2
            self.vW1 = momentum * self.vW1 - self._lr * dW1
            self.vb3 = momentum * self.vb3 - self._lr * db3
            self.vb2 = momentum * self.vb2 - self._lr * db2
            self.vb1 = momentum * self.vb1 - self._lr * db1
            self.W3 += self.vW3
            self.W2 += self.vW2
            self.W1 += self.vW1
            self.b3 += self.vb3
            self.b2 += self.vb2
            self.b1 += self.vb1
            self.update_count += 1
            if self.update_count % 5 == 0:
                self._save()

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        self._path.write_bytes(pickle.dumps(data))

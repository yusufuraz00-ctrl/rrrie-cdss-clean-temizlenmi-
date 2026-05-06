"""Loss-aware typed signal journal for internal clinical state propagation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SignalKind(str, Enum):
    FINDING = "finding"
    RED_FLAG = "red_flag"
    EXPOSURE = "exposure"
    VITAL = "vital"
    HYPOTHESIS = "hypothesis"
    MECHANISM = "mechanism"
    EVIDENCE = "evidence"
    INTERVENTION = "intervention"
    CONTEXT = "context"


class CaseSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    signal_id: str = Field(default_factory=lambda: f"sig_{uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=_utc_now)
    kind: SignalKind
    label: str
    canonical_label: str = ""
    raw_span: str = ""
    value: str = ""
    unit: str = ""
    polarity: str = "positive"
    temporality: str = ""
    certainty: str = ""
    confidence: float = 0.0
    source_stage: str = ""
    source_refs: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class SignalJournal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    created_at: str = Field(default_factory=_utc_now)
    signals: list[CaseSignal] = Field(default_factory=list)

    def append(self, signal: CaseSignal) -> None:
        self.signals.append(signal)

    def extend(self, items: list[CaseSignal]) -> None:
        self.signals.extend(items)

    def by_kind(self, kind: SignalKind) -> list[CaseSignal]:
        return [item for item in self.signals if item.kind == kind]

    def latest_labels(self, kind: SignalKind, *, limit: int = 8) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for item in reversed(self.by_kind(kind)):
            label = str(item.canonical_label or item.label or "").strip()
            key = label.lower()
            if not label or key in seen:
                continue
            seen.add(key)
            labels.append(label)
            if len(labels) >= limit:
                break
        labels.reverse()
        return labels

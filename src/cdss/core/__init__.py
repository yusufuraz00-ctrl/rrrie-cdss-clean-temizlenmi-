"""Core state-machine primitives for vNext CDSS."""

from src.cdss.core.fabric import CaseSignal, SignalJournal, SignalKind
from src.cdss.core.state import CaseState, StatePatch, apply_state_patch, begin_case_state

__all__ = [
    "CaseSignal",
    "CaseState",
    "SignalJournal",
    "SignalKind",
    "StatePatch",
    "apply_state_patch",
    "begin_case_state",
]

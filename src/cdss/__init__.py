from __future__ import annotations

__all__ = ["CdssApplicationService"]


def __getattr__(name: str):
    if name == "CdssApplicationService":
        from src.cdss.app.service import CdssApplicationService

        return CdssApplicationService
    raise AttributeError(name)

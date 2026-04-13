from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]
    config_path: Path

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw.get("analysis", {})

    def resolve_path(self, value: str | Path) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.config_path.parent / p).resolve()


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RuntimeConfig(raw=raw, config_path=path)


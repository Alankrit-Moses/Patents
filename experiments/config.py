from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EndpointConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: int = 180

    def resolved_api_key(self) -> str:
        return self.api_key or os.getenv(self.api_key_env, "local")


@dataclass
class HarnessConfig:
    project_root: Path
    generator: EndpointConfig = field(default_factory=EndpointConfig)
    judge: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(temperature=0.0, max_output_tokens=2048)
    )
    context_limit_tokens: int = 131072
    chars_per_token_estimate: float = 3.5
    chunk_chars: int = 16000
    chunk_overlap_chars: int = 2000
    max_workers: int = 4
    max_steps: int = 8
    repair_once: bool = True
    induction_pairs: int = 4
    fold_count: int = 3
    seed: int = 17
    results_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        return self.results_dir or (self.project_root / "experiments" / "artifacts")


def _endpoint(data: dict[str, Any] | None, default: EndpointConfig) -> EndpointConfig:
    if not data:
        return default
    values = {name: data.get(name, getattr(default, name)) for name in default.__dataclass_fields__}
    return EndpointConfig(**values)


def load_config(path: str | Path | None = None) -> HarnessConfig:
    package_root = Path(__file__).resolve().parents[1]
    if path is None:
        return HarnessConfig(project_root=package_root)

    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_root = data.get("project_root", str(package_root))
    project_root = Path(raw_root)
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    cfg = HarnessConfig(
        project_root=project_root,
        generator=_endpoint(data.get("generator"), EndpointConfig()),
        judge=_endpoint(
            data.get("judge"), EndpointConfig(temperature=0.0, max_output_tokens=2048)
        ),
    )
    for name in (
        "context_limit_tokens",
        "chars_per_token_estimate",
        "chunk_chars",
        "chunk_overlap_chars",
        "max_workers",
        "max_steps",
        "repair_once",
        "induction_pairs",
        "fold_count",
        "seed",
    ):
        if name in data:
            setattr(cfg, name, data[name])
    if data.get("results_dir"):
        value = Path(data["results_dir"])
        cfg.results_dir = value if value.is_absolute() else (config_path.parent / value).resolve()
    return cfg

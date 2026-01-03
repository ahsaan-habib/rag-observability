from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def _pricing(path: str = "pricing.yaml") -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def request_cost(model: str, input_tokens: int, output_tokens: int, wall_ms: float) -> float:
    cfg = _pricing()
    price = cfg.get("models", {}).get(model, {})
    tokens = (input_tokens * price.get("input_per_m", 0.0)
              + output_tokens * price.get("output_per_m", 0.0)) / 1e6
    compute = cfg.get("compute_usd_per_hour", 0.0) * wall_ms / 3_600_000
    return round(tokens + compute, 6)

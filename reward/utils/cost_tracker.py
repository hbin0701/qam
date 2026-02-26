"""Simple persistent cost tracker for reward-generation model calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional

from .logfmt import join_kv, level_prefix

# USD per 1M tokens. Mirrors RA/prototype defaults.
MODEL_PRICING_USD_PER_1M = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}

DEFAULT_COST_FILE = Path(__file__).resolve().parents[1] / "specs" / "costs" / "global_costs.json"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @property
    def billable_output_tokens(self) -> int:
        return int(self.output_tokens) + int(self.reasoning_tokens)


@dataclass
class RewardCostTracker:
    model_name: str
    cost_file: Path = field(default_factory=lambda: DEFAULT_COST_FILE)
    pricing: Dict[str, float] = field(default_factory=dict)
    num_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.model_name in MODEL_PRICING_USD_PER_1M:
            self.pricing = MODEL_PRICING_USD_PER_1M[self.model_name]
        self._restore()

    def calculate_cost(self, usage: Optional[TokenUsage] = None) -> float:
        if not self.pricing:
            return 0.0
        if usage is None:
            in_tok = self.input_tokens
            out_tok = self.output_tokens + self.reasoning_tokens
        else:
            in_tok = usage.input_tokens
            out_tok = usage.billable_output_tokens
        input_cost = (in_tok / 1_000_000.0) * self.pricing.get("input", 0.0)
        output_cost = (out_tok / 1_000_000.0) * self.pricing.get("output", 0.0)
        return float(input_cost + output_cost)

    def add_usage(self, usage: TokenUsage) -> float:
        self.num_calls += 1
        self.input_tokens += int(usage.input_tokens)
        self.output_tokens += int(usage.output_tokens)
        self.reasoning_tokens += int(usage.reasoning_tokens)
        self.total_tokens += int(usage.total_tokens)
        self._save()
        return self.calculate_cost(usage)

    def global_cost(self) -> float:
        return self.calculate_cost()

    def print_update(self, usage: TokenUsage, prefix: str = "RewardCall") -> None:
        call_cost = self.calculate_cost(usage)
        global_cost = self.global_cost()
        pairs = [
            ("model", self.model_name),
            ("current_cost", f"${call_cost:.6f}"),
            ("global_cost_so_far", f"${global_cost:.6f}"),
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("reasoning", usage.reasoning_tokens),
        ]
        print(f"{level_prefix(prefix, level='info')} {join_kv(pairs, tone='muted')}")

    def _restore(self) -> None:
        path = self.cost_file
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
            model_data = payload.get(self.model_name, {})
            self.num_calls = int(model_data.get("num_calls", 0))
            self.input_tokens = int(model_data.get("input_tokens", 0))
            self.output_tokens = int(model_data.get("output_tokens", 0))
            self.reasoning_tokens = int(model_data.get("reasoning_tokens", 0))
            self.total_tokens = int(model_data.get("total_tokens", 0))
        except Exception:
            # Do not hard-fail calls due to corrupt accounting files.
            return

    def _save(self) -> None:
        path = self.cost_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Dict[str, object]] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                payload = {}
        payload[self.model_name] = {
            "model": self.model_name,
            "num_calls": self.num_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "pricing": self.pricing,
            "global_cost_usd": round(self.global_cost(), 8),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


_GLOBAL_TRACKERS: Dict[str, RewardCostTracker] = {}


def get_global_cost_tracker(model_name: str) -> RewardCostTracker:
    tracker = _GLOBAL_TRACKERS.get(model_name)
    if tracker is None:
        tracker = RewardCostTracker(model_name=model_name)
        _GLOBAL_TRACKERS[model_name] = tracker
    return tracker

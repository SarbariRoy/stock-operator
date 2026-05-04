"""Shared LT/ST scoring defaults and snapshot utilities.

This module is intentionally import-safe for both UI and batch scripts.
It centralizes default scoring settings so background jobs can stay aligned with
UI defaults and detect drift between runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from stock_triggers.ui.patterns.markov import (
    get_default_markov_score_policy,
    load_signal_markov_model,
)
from stock_triggers.ui.patterns.stop_risk import (
    get_default_stop_risk_penalty_policy,
    load_signal_stop_risk_model,
)
from stock_triggers.ui.patterns.st_score import ST_RANK_BLEND_WEIGHT

DEFAULT_TOMORROW_CUTOFF = 70
LT_DEFAULT_MIN_SCORE = 80
ST_DEFAULT_MIN_SCORE = 10
ST_DEFAULT_RECENCY_LABEL = "Last 2 years"

TOMORROW_SCORE_METHODS = {
    "LT score": {
        "column": "signal_score",
        "label": "LT score",
        "short_label": "LT",
        "higher_is_better": True,
        "filter_label": "Minimum LT score",
        "default_filter": 70,
        "display_scale": 1.0,
        "display_suffix": "",
    },
    "ST score": {
        "column": "st_score",
        "label": "ST score",
        "short_label": "ST",
        "higher_is_better": True,
        "filter_label": "Minimum ST score",
        "default_filter": 10,
        "display_scale": 1.0,
        "display_suffix": "",
    },
}


def _resolve_markov_policy_snapshot() -> dict[str, Any]:
    defaults = get_default_markov_score_policy()
    payload = load_signal_markov_model()
    policy = payload.get("score_policy") if isinstance(payload.get("score_policy"), dict) else {}
    out = dict(defaults)
    out.update(policy)
    return out


def _resolve_stop_risk_policy_snapshot() -> dict[str, Any]:
    defaults = get_default_stop_risk_penalty_policy()
    payload = load_signal_stop_risk_model()
    policy = payload.get("stop_risk_penalty_policy") if isinstance(payload.get("stop_risk_penalty_policy"), dict) else {}
    out = dict(defaults)
    out.update(policy)
    return out


def build_scoring_defaults_snapshot() -> dict[str, Any]:
    return {
        "tomorrow": {
            "default_cutoff": int(DEFAULT_TOMORROW_CUTOFF),
            "score_methods": TOMORROW_SCORE_METHODS,
        },
        "long_term": {
            "default_min_score": int(LT_DEFAULT_MIN_SCORE),
        },
        "short_term": {
            "default_min_score": int(ST_DEFAULT_MIN_SCORE),
            "default_recency_label": ST_DEFAULT_RECENCY_LABEL,
            "st_rank_blend_weight": float(ST_RANK_BLEND_WEIGHT),
        },
        "markov": {
            "score_policy": _resolve_markov_policy_snapshot(),
        },
        "stop_risk": {
            "penalty_policy": _resolve_stop_risk_policy_snapshot(),
        },
    }


def compute_scoring_defaults_hash(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_for_diff(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_for_diff(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_normalize_for_diff(v) for v in value]
    return value


def diff_scoring_snapshots(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[dict[str, Any]]:
    left = _normalize_for_diff(before or {})
    right = _normalize_for_diff(after or {})
    changes: list[dict[str, Any]] = []

    def _walk(path: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            keys = sorted(set(a.keys()) | set(b.keys()))
            for key in keys:
                next_path = f"{path}.{key}" if path else str(key)
                _walk(next_path, a.get(key), b.get(key))
            return

        if isinstance(a, list) and isinstance(b, list):
            if a != b:
                changes.append({"field": path, "before": a, "after": b})
            return

        if a != b:
            changes.append({"field": path, "before": a, "after": b})

    _walk("", left, right)
    return changes

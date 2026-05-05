from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionResult:
    cv: float
    ca: float
    score: float
    is_alert: bool


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_fusion_score(
    cv: float,
    ca: float | None,
    alpha: float,
    beta: float,
    threshold: float,
) -> FusionResult:
    cv = clamp01(cv)
    if ca is None:
        # Audio not provided — use vision score alone
        score = cv
        ca_out = 0.0
    else:
        ca = clamp01(ca)
        score = clamp01((alpha * cv) + (beta * ca))
        ca_out = ca
    return FusionResult(cv=cv, ca=ca_out, score=score, is_alert=score > threshold)

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
    cv: float | None,
    ca: float | None,
    alpha: float,
    beta: float,
    threshold: float,
    *,
    audio_alert_threshold: float | None = None,
) -> FusionResult:
    """Fuse vision (Cv) and audio (Ca) threat cues.

    - Both provided: S = clip(alpha*Cv + beta*Ca), alert if S > threshold (vision-scale gate).
    - Vision only (Ca is None): S = Cv, alert if Cv > threshold.
    - Audio only (Cv is None): S = Ca, alert if Ca > audio_alert_threshold (defaults to *threshold*).
    """
    t_vis = threshold
    t_aud = audio_alert_threshold if audio_alert_threshold is not None else threshold

    if cv is None and ca is None:
        return FusionResult(cv=0.0, ca=0.0, score=0.0, is_alert=False)

    if cv is None:
        ca = clamp01(float(ca) if ca is not None else 0.0)
        return FusionResult(cv=0.0, ca=ca, score=ca, is_alert=ca > t_aud)

    cv = clamp01(cv)
    if ca is None:
        return FusionResult(cv=cv, ca=0.0, score=cv, is_alert=cv > t_vis)

    ca = clamp01(ca)
    score = clamp01((alpha * cv) + (beta * ca))
    return FusionResult(cv=cv, ca=ca, score=score, is_alert=score > t_vis)

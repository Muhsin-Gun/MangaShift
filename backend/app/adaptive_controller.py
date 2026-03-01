from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from .config import Settings


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class AdaptiveDecision:
    strength: float
    guidance_scale: float
    steps: int
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "strength": round(float(self.strength), 6),
            "guidance_scale": round(float(self.guidance_scale), 6),
            "steps": int(self.steps),
            "confidence": round(float(self.confidence), 6),
            "reason": self.reason,
        }


class AdaptiveController:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._stats: Dict[str, dict] = {}

    def _base(self, render_quality: str, device: str) -> tuple[float, float, int]:
        if render_quality == "preview":
            strength, guidance, steps = 0.36, 6.6, int(self.settings.quality_preview_steps)
        elif render_quality == "final":
            strength, guidance, steps = 0.52, 8.2, int(self.settings.quality_final_steps)
        else:
            strength, guidance, steps = 0.45, 7.4, int(self.settings.quality_balanced_steps)
        if device != "cuda":
            # CPU cap for predictable latency.
            steps = min(steps, 18 if render_quality == "final" else 14)
        return strength, guidance, steps

    def _style_state(self, style_id: str) -> dict:
        key = (style_id or "original").strip().lower()
        if key not in self._stats:
            self._stats[key] = {
                "runs": 0,
                "ema_identity": 1.0,
                "ema_edge": 1.0,
                "ema_blur": 1.0,
                "last_decision": {},
                "last_backend": "",
            }
        return self._stats[key]

    def propose_initial(self, style_id: str, render_quality: str, device: str) -> AdaptiveDecision:
        strength, guidance, steps = self._base(render_quality=render_quality, device=device)
        state = self._style_state(style_id)
        reason = ["base_quality_profile"]
        confidence = 0.58

        ema_identity = float(state.get("ema_identity", 1.0))
        ema_edge = float(state.get("ema_edge", 1.0))
        ema_blur = float(state.get("ema_blur", 1.0))
        if ema_identity < self.settings.controller_target_identity:
            strength -= 0.06
            guidance += 0.5
            steps += 2
            reason.append("identity_under_target")
            confidence += 0.08
        if ema_edge < self.settings.controller_target_edge_overlap:
            strength -= 0.05
            guidance += 0.2
            reason.append("edge_under_target")
            confidence += 0.06
        if ema_blur < self.settings.controller_target_blur_score:
            strength -= 0.04
            steps += 2
            reason.append("blur_under_target")
            confidence += 0.05

        decision = AdaptiveDecision(
            strength=_clamp(strength, 0.2, 0.72),
            guidance_scale=_clamp(guidance, 5.0, 11.0),
            steps=int(max(6, min(64, steps))),
            confidence=_clamp(confidence, 0.35, 0.95),
            reason="|".join(reason),
        )
        return decision

    def propose_retry(
        self,
        style_id: str,
        render_quality: str,
        device: str,
        structural_report: dict,
        identity_report: dict,
        attempt: int,
        previous: Optional[AdaptiveDecision] = None,
    ) -> AdaptiveDecision:
        if previous is None:
            current = self.propose_initial(style_id=style_id, render_quality=render_quality, device=device)
        else:
            current = previous

        strength = float(current.strength)
        guidance = float(current.guidance_scale)
        steps = int(current.steps)
        reason = [current.reason, f"retry_{attempt}"]
        confidence = float(current.confidence)

        edge_overlap = float(structural_report.get("edge_overlap", 1.0))
        blur_score = float(structural_report.get("blur_score", 1.0))
        contrast_shift = float(structural_report.get("contrast_shift", 0.0))
        identity_score = float(identity_report.get("best_similarity", 1.0))
        identity_target = float(identity_report.get("threshold", self.settings.controller_target_identity))
        drift = bool(identity_report.get("drift_detected", False))

        if drift or identity_score < identity_target:
            strength -= 0.08
            guidance += 0.7
            steps += 3
            reason.append("identity_retry")
            confidence += 0.1
        if edge_overlap < self.settings.controller_target_edge_overlap:
            strength -= 0.05
            guidance += 0.2
            reason.append("edge_retry")
            confidence += 0.06
        if blur_score < self.settings.controller_target_blur_score:
            strength -= 0.04
            steps += 2
            reason.append("blur_retry")
            confidence += 0.04
        if contrast_shift > 38.0:
            guidance -= 0.3
            reason.append("contrast_retry")
            confidence += 0.02

        # Progressive conservatism on repeated retries.
        strength -= 0.02 * max(0, int(attempt) - 1)
        if device != "cuda":
            steps = min(steps, 20 if render_quality == "final" else 16)

        return AdaptiveDecision(
            strength=_clamp(strength, 0.2, 0.72),
            guidance_scale=_clamp(guidance, 5.0, 11.5),
            steps=int(max(6, min(72, steps))),
            confidence=_clamp(confidence, 0.35, 0.98),
            reason="|".join([r for r in reason if r]),
        )

    def record_outcome(
        self,
        style_id: str,
        decision: AdaptiveDecision,
        structural_report: dict,
        identity_report: dict,
        style_backend: str,
    ) -> None:
        alpha = 0.25
        with self._lock:
            state = self._style_state(style_id)
            runs = int(state.get("runs", 0)) + 1
            identity = _clamp(float(identity_report.get("best_similarity", 1.0)), 0.0, 1.0)
            edge = _clamp(float(structural_report.get("edge_overlap", 1.0)), 0.0, 1.0)
            blur = _clamp(float(structural_report.get("blur_score", 1.0)), 0.0, 2.0)
            state["ema_identity"] = (1.0 - alpha) * float(state.get("ema_identity", identity)) + alpha * identity
            state["ema_edge"] = (1.0 - alpha) * float(state.get("ema_edge", edge)) + alpha * edge
            state["ema_blur"] = (1.0 - alpha) * float(state.get("ema_blur", blur)) + alpha * blur
            state["runs"] = runs
            state["last_decision"] = decision.to_dict()
            state["last_backend"] = style_backend

    def summary(self) -> dict:
        with self._lock:
            styles = {
                sid: {
                    "runs": int(data.get("runs", 0)),
                    "ema_identity": round(float(data.get("ema_identity", 1.0)), 6),
                    "ema_edge": round(float(data.get("ema_edge", 1.0)), 6),
                    "ema_blur": round(float(data.get("ema_blur", 1.0)), 6),
                    "last_decision": dict(data.get("last_decision", {})),
                    "last_backend": str(data.get("last_backend", "")),
                }
                for sid, data in self._stats.items()
            }
        return {"styles": styles, "style_count": len(styles)}

    def summary_style(self, style_id: str) -> dict:
        with self._lock:
            state = dict(self._style_state(style_id))
        return {
            "style_id": style_id,
            "runs": int(state.get("runs", 0)),
            "ema_identity": round(float(state.get("ema_identity", 1.0)), 6),
            "ema_edge": round(float(state.get("ema_edge", 1.0)), 6),
            "ema_blur": round(float(state.get("ema_blur", 1.0)), 6),
            "last_decision": dict(state.get("last_decision", {})),
            "last_backend": str(state.get("last_backend", "")),
        }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RepairAction:
    target: str
    operation: str
    params: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "operation": self.operation,
            "params": {k: float(v) for k, v in self.params.items()},
            "reason": self.reason,
            "confidence": round(float(self.confidence), 6),
        }


@dataclass
class RepairPlan:
    apply_rerender: bool
    strength_override: float
    control_weights: Dict[str, float]
    actions: List[RepairAction] = field(default_factory=list)
    confidence: float = 0.5
    note: str = ""

    def retry_strength(self, attempt: int) -> float:
        # Conservative decay to avoid over-stylization on repeated retries.
        attempt = max(0, int(attempt))
        return max(0.2, float(self.strength_override) - 0.07 * attempt)

    def to_dict(self) -> dict:
        return {
            "apply_rerender": bool(self.apply_rerender),
            "strength_override": round(float(self.strength_override), 6),
            "control_weights": {k: round(float(v), 6) for k, v in self.control_weights.items()},
            "actions": [a.to_dict() for a in self.actions],
            "confidence": round(float(self.confidence), 6),
            "note": self.note,
        }


def build_repair_plan(
    structural_report: dict,
    identity_report: dict,
    render_quality: str,
    style_id: str,
) -> RepairPlan:
    edge_overlap = float(structural_report.get("edge_overlap", 1.0))
    blur_score = float(structural_report.get("blur_score", 1.0))
    contrast_shift = float(structural_report.get("contrast_shift", 0.0))
    texture_shift = float(structural_report.get("texture_shift", 0.0))
    drift = bool(identity_report.get("drift_detected", False))
    similarity = float(identity_report.get("best_similarity", 1.0))
    threshold = float(identity_report.get("threshold", 0.78))

    actions: List[RepairAction] = []
    control = {"lineart": 1.0, "depth": 0.4, "openpose": 0.45}
    strength = 0.46 if render_quality == "final" else 0.4

    if edge_overlap < 0.6:
        control["lineart"] = min(1.25, control["lineart"] + 0.2)
        strength = max(0.24, strength - 0.08)
        actions.append(
            RepairAction(
                target="lineart",
                operation="increase_line_preservation",
                params={"lineart_weight": control["lineart"], "strength": strength},
                reason="Edge overlap dropped below target",
                confidence=0.84,
            )
        )

    if blur_score < 0.7:
        strength = max(0.2, strength - 0.06)
        actions.append(
            RepairAction(
                target="detail",
                operation="reduce_diffusion_strength",
                params={"strength": strength},
                reason="Blur ratio indicates detail loss",
                confidence=0.8,
            )
        )

    if contrast_shift > 35.0:
        actions.append(
            RepairAction(
                target="tone",
                operation="tone_rebalance",
                params={"contrast_clamp": 0.9},
                reason="Contrast drift exceeded tolerance",
                confidence=0.74,
            )
        )

    if texture_shift > 0.09:
        control["depth"] = min(0.9, control["depth"] + 0.1)
        actions.append(
            RepairAction(
                target="texture",
                operation="stabilize_surface_detail",
                params={"depth_weight": control["depth"]},
                reason="Texture density drift exceeded tolerance",
                confidence=0.7,
            )
        )

    if drift or similarity < threshold:
        control["openpose"] = min(0.85, control["openpose"] + 0.15)
        strength = max(0.2, strength - 0.07)
        actions.append(
            RepairAction(
                target="identity",
                operation="tighten_identity_constraints",
                params={"openpose_weight": control["openpose"], "strength": strength},
                reason="Identity similarity below threshold",
                confidence=0.9,
            )
        )

    if style_id in {"noir_contrast", "dark_fantasy"} and render_quality == "final":
        control["depth"] = min(1.0, control["depth"] + 0.08)
        actions.append(
            RepairAction(
                target="lighting",
                operation="reinforce_depth_guidance",
                params={"depth_weight": control["depth"]},
                reason="Style requires stronger depth coherence",
                confidence=0.68,
            )
        )

    should_apply = bool(actions) and render_quality in {"balanced", "final"}
    confidence = min(0.95, 0.52 + 0.08 * len(actions))
    note = "Planner generated structural/identity repair directives" if should_apply else "No repair actions needed"
    return RepairPlan(
        apply_rerender=should_apply,
        strength_override=strength,
        control_weights=control,
        actions=actions,
        confidence=confidence,
        note=note,
    )

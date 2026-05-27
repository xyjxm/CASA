"""Shared utilities for CASA Phase 5 conformal and baseline evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

MAIN_SKILLS = ("walk", "turn", "gesture", "passive")
ALPHA_DEFAULT = 0.10

METHOD_ORDER = (
    "sonic_only",
    "hard_contract",
    "raw_critic_0p5",
    "global_conformal",
    "casa_a_per_skill",
)
METHOD_DISPLAY = {
    "sonic_only": "SONIC-only",
    "hard_contract": "SONIC + Hard Contract",
    "raw_critic_0p5": "SONIC + Raw Critic (0.5)",
    "global_conformal": "SONIC + Global Conformal",
    "casa_a_per_skill": "SONIC + CASA-A (per-skill conformal)",
}


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            row = dict(row)
            row["label_int"] = _int(row.get("label"))
            row["raw_critic_risk_float"] = _float(row.get("raw_critic_risk"))
            row["hard_contract_score_float"] = _float(row.get("hard_contract_score"))
            row["hard_contract_fixed_reject_bool"] = _bool(row.get("hard_contract_fixed_reject"))
            rows.append(row)
    return rows


def threshold_for_fnr(rows: list[dict[str, Any]], alpha: float) -> float:
    """Largest observed-risk threshold with calibration FNR <= alpha.

    The Phase 5 gate rejects when risk >= threshold. False negatives are unsafe
    rows with risk < threshold, so using an observed unsafe score as the
    threshold handles ties conservatively.
    """

    positives = sorted(
        float(row["raw_critic_risk_float"]) for row in rows if int(row["label_int"]) == 1
    )
    if not positives:
        return 1.0
    allowed_false_negatives = max(0, min(len(positives) - 1, int(math.floor(alpha * len(positives)))))
    best = positives[0]
    for threshold in sorted(set(positives)):
        false_negatives = sum(1 for score in positives if score < threshold)
        if false_negatives <= allowed_false_negatives:
            best = threshold
        else:
            break
    return float(best)


def threshold_for_target_fnr(rows: list[dict[str, Any]], target_fnr: float) -> float:
    positives = sorted(
        float(row["raw_critic_risk_float"]) for row in rows if int(row["label_int"]) == 1
    )
    if not positives:
        return 1.0
    best = positives[0]
    for threshold in sorted(set(positives)):
        false_negative_rate = sum(1 for score in positives if score < threshold) / len(positives)
        if false_negative_rate <= target_fnr + 1e-12:
            best = threshold
        else:
            break
    return float(best)


def conformal_thresholds(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    calibration = [row for row in rows if str(row.get("phase4_split")) == "calibration"]
    per_skill = {
        skill: threshold_for_fnr([row for row in calibration if row.get("skill_name") == skill], alpha)
        for skill in MAIN_SKILLS
    }
    return {
        "global": threshold_for_fnr(calibration, alpha),
        "per_skill": per_skill,
    }


def reject_flags_for_method(
    rows: list[dict[str, Any]],
    method: str,
    thresholds: dict[str, Any],
) -> list[bool]:
    global_threshold = float(thresholds["global"])
    per_skill = thresholds["per_skill"]
    flags: list[bool] = []
    for row in rows:
        risk = float(row["raw_critic_risk_float"])
        skill = str(row.get("skill_name"))
        if method == "sonic_only":
            flags.append(False)
        elif method == "hard_contract":
            flags.append(bool(row["hard_contract_fixed_reject_bool"]))
        elif method == "raw_critic_0p5":
            flags.append(risk >= 0.5)
        elif method == "global_conformal":
            flags.append(risk >= global_threshold)
        elif method == "casa_a_per_skill":
            flags.append(risk >= float(per_skill[skill]))
        else:
            raise ValueError(f"Unknown Phase 5 method: {method}")
    return flags


def metrics_from_reject_flags(
    rows: list[dict[str, Any]],
    reject_flags: list[bool],
    *,
    method: str,
    split: str,
    skill_name: str = "all",
) -> dict[str, Any]:
    labels = [int(row["label_int"]) for row in rows]
    count = len(rows)
    unsafe_total = sum(labels)
    safe_total = count - unsafe_total
    reject_count = sum(1 for flag in reject_flags if flag)
    accepted = [not flag for flag in reject_flags]
    accepted_count = sum(1 for flag in accepted if flag)
    unsafe_invocations = sum(label for label, is_accepted in zip(labels, accepted) if is_accepted)
    prevented_unsafe = sum(
        label for label, is_rejected in zip(labels, reject_flags) if is_rejected
    )
    accepted_safe = sum(
        1 for label, is_accepted in zip(labels, accepted) if label == 0 and is_accepted
    )
    return {
        "method": method,
        "method_display": METHOD_DISPLAY.get(method, method),
        "split": split,
        "skill_name": skill_name,
        "count": count,
        "safe_total": safe_total,
        "unsafe_total": unsafe_total,
        "reject_count": reject_count,
        "reject_rate": reject_count / count if count else 0.0,
        "accepted_count": accepted_count,
        "unsafe_invocation_count": unsafe_invocations,
        "fnr": unsafe_invocations / unsafe_total if unsafe_total else 0.0,
        "accepted_unsafe_rate": unsafe_invocations / accepted_count if accepted_count else 0.0,
        "prevented_unsafe_count": prevented_unsafe,
        "rejection_precision": prevented_unsafe / reject_count if reject_count else 0.0,
        "task_success_proxy": accepted_safe / safe_total if safe_total else 0.0,
    }


def evaluate_method(
    rows: list[dict[str, Any]],
    method: str,
    thresholds: dict[str, Any],
    *,
    split: str,
    skill_name: str = "all",
) -> dict[str, Any]:
    return metrics_from_reject_flags(
        rows,
        reject_flags_for_method(rows, method, thresholds),
        method=method,
        split=split,
        skill_name=skill_name,
    )


def budget_reject_metrics(
    rows: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
    *,
    method: str,
    split: str,
    reject_budget: float,
) -> dict[str, Any]:
    reject_count = int(math.floor(len(rows) * reject_budget))
    order = sorted(range(len(rows)), key=lambda index: -float(score_fn(rows[index])))
    reject = [False] * len(rows)
    for index in order[:reject_count]:
        reject[index] = True
    row = metrics_from_reject_flags(rows, reject, method=method, split=split)
    row["reject_budget"] = reject_budget
    return row


def hard_contract_scores(features: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = {name: features[:, index] for index, name in enumerate(feature_names)}

    def column(name: str, default: float) -> np.ndarray:
        return values.get(name, np.full(features.shape[0], default, dtype=np.float32)).astype(np.float32)

    min_user = column("env/min_user_distance/current", 10.0)
    min_arm_user = column("env/min_arm_user_distance/current", 10.0)
    min_obstacle = column("env/min_obstacle_distance/current", 10.0)
    collision_user = column("env/external_collision_user/current", 0.0)
    collision_obstacle = column("env/external_collision_obstacle/current", 0.0)
    torso_roll = np.abs(column("robot/torso_roll/current", 0.0))
    torso_pitch = np.abs(column("robot/torso_pitch/current", 0.0))
    overrun = column("runtime/control_overrun_ratio", 0.0)

    score_terms = np.stack(
        [
            np.clip((0.50 - min_user) / 0.50, 0.0, 1.0),
            np.clip((0.20 - min_arm_user) / 0.20, 0.0, 1.0),
            np.clip((0.10 - min_obstacle) / 0.10, 0.0, 1.0),
            np.clip(torso_roll / 0.60, 0.0, 1.0),
            np.clip(torso_pitch / 0.60, 0.0, 1.0),
            np.clip(collision_user, 0.0, 1.0),
            np.clip(collision_obstacle, 0.0, 1.0),
            np.clip(overrun, 0.0, 1.0),
        ],
        axis=1,
    )
    hard_score = np.max(score_terms, axis=1).astype(np.float32)
    fixed_reject = (
        (min_user < 0.50)
        | (min_arm_user < 0.20)
        | (min_obstacle < 0.10)
        | (collision_user > 0.5)
        | (collision_obstacle > 0.5)
        | (torso_roll > 0.60)
        | (torso_pitch > 0.60)
    )
    return hard_score, fixed_reject.astype(bool)


def calibration_table_rows(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    calibration = [row for row in rows if str(row.get("phase4_split")) == "calibration"]
    output: list[dict[str, Any]] = []
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in calibration if row.get("skill_name") == skill]
        reject = reject_flags_for_method(skill_rows, "casa_a_per_skill", thresholds)
        metrics = metrics_from_reject_flags(
            skill_rows, reject, method="casa_a_per_skill", split="calibration", skill_name=skill
        )
        metrics.update(
            {
                "alpha": alpha,
                "threshold": float(thresholds["per_skill"][skill]),
                "calibration_distribution": "phase4_strict_clean_full_calibration_split",
            }
        )
        output.append(metrics)
    reject = reject_flags_for_method(calibration, "global_conformal", thresholds)
    metrics = metrics_from_reject_flags(
        calibration, reject, method="global_conformal", split="calibration", skill_name="all"
    )
    metrics.update(
        {
            "alpha": alpha,
            "threshold": float(thresholds["global"]),
            "calibration_distribution": "phase4_strict_clean_full_calibration_split",
        }
    )
    output.append(metrics)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}

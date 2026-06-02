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
CRITIC_VAL_SPLITS = ("critic_val", "val")
CALIBRATION_SPLITS = ("calibration", "conformal_calibration")
REQUIRED_PREDICTION_FIELDS = (
    "sample_id",
    "phase4_split",
    "skill_name",
    "label",
    "raw_critic_risk",
    "hard_contract_score",
    "hard_contract_fixed_reject",
)

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


class Phase5ValidationError(ValueError):
    """Raised when Phase 5 artifacts are malformed or incomplete."""


def read_prediction_rows(path: Path, *, validate: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if validate:
            missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                raise Phase5ValidationError(f"{path} is missing required columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            row = dict(row)
            if validate:
                _validate_prediction_row(row, line_number=line_number, path=path)
                row["label_int"] = _parse_label(row["label"], line_number, path)
                row["raw_critic_risk_float"] = _parse_finite_float(
                    row["raw_critic_risk"], "raw_critic_risk", line_number, path
                )
                row["hard_contract_score_float"] = _parse_finite_float(
                    row["hard_contract_score"], "hard_contract_score", line_number, path
                )
                row["hard_contract_fixed_reject_bool"] = _parse_bool(
                    row["hard_contract_fixed_reject"], line_number, path
                )
            else:
                row["label_int"] = _int(row.get("label"))
                row["raw_critic_risk_float"] = _float(row.get("raw_critic_risk"))
                row["hard_contract_score_float"] = _float(row.get("hard_contract_score"))
                row["hard_contract_fixed_reject_bool"] = _bool(row.get("hard_contract_fixed_reject"))
            rows.append(row)
    return rows


def split_role(split: Any) -> str:
    split_text = str(split).strip()
    if split_text in CRITIC_VAL_SPLITS:
        return "critic_val"
    if split_text in CALIBRATION_SPLITS:
        return "calibration"
    return split_text


def split_indices_for_phase5(splits: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Return independent Phase 5 split roles.

    Raw critic model selection must use only ``critic_val`` / ``val`` rows. The
    conformal calibration split is intentionally separate and is never carved
    into validation data. For older datasets that only have train/calibration/
    test, a deterministic critic validation subset is carved from train.
    """

    roles = np.asarray([split_role(split) for split in splits], dtype="<U16")
    train = np.where(roles == "train")[0]
    critic_val = np.where(roles == "critic_val")[0]
    calibration = np.where(roles == "calibration")[0]
    test = np.where(roles == "test")[0]

    if len(train) and not len(critic_val):
        rng = np.random.default_rng(seed)
        shuffled = train.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(0.1 * len(shuffled)))
        critic_val = np.sort(shuffled[:val_size])
        train = np.sort(shuffled[val_size:])

    if not len(train):
        raise ValueError("Missing train split.")
    if not len(critic_val):
        raise ValueError("Missing critic validation split.")
    if not len(calibration):
        raise ValueError("Missing untouched conformal calibration split.")
    if not len(test):
        raise ValueError("Missing test split.")

    return {
        "train": np.asarray(train, dtype=np.int64),
        "critic_val": np.asarray(critic_val, dtype=np.int64),
        "calibration": np.asarray(calibration, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }


def phase5_split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("train", "critic_val", "calibration", "test"):
        role_rows = [row for row in rows if split_role(row.get("phase4_split")) == role]
        per_skill = {}
        for skill in MAIN_SKILLS:
            skill_rows = [row for row in role_rows if row.get("skill_name") == skill]
            unsafe = sum(1 for row in skill_rows if int(row.get("label_int", 0)) == 1)
            per_skill[skill] = {
                "total": len(skill_rows),
                "safe": len(skill_rows) - unsafe,
                "unsafe": unsafe,
            }
        roles[role] = {
            "total": len(role_rows),
            "safe": sum(1 for row in role_rows if int(row.get("label_int", 0)) == 0),
            "unsafe": sum(1 for row in role_rows if int(row.get("label_int", 0)) == 1),
            "per_skill": per_skill,
        }
    return {
        "sample_count": len(rows),
        "split_roles": roles,
        "model_selection_role": "critic_val",
        "conformal_calibration_role": "calibration",
    }


def validate_phase5_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    min_unsafe_calibration_per_skill: int = 200,
    require_calibration: bool = True,
    require_test: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise Phase5ValidationError("Phase 5 prediction CSV is empty.")

    audit = phase5_split_audit(rows)
    problems: list[str] = []
    roles = audit["split_roles"]
    if require_calibration:
        for skill in MAIN_SKILLS:
            stats = roles["calibration"]["per_skill"][skill]
            if stats["total"] <= 0:
                problems.append(f"missing calibration rows for skill {skill!r}")
            if stats["unsafe"] < min_unsafe_calibration_per_skill:
                problems.append(
                    f"insufficient unsafe calibration rows for skill {skill!r}: "
                    f"{stats['unsafe']} < {min_unsafe_calibration_per_skill}"
                )
    if require_test:
        for skill in MAIN_SKILLS:
            stats = roles["test"]["per_skill"][skill]
            if stats["total"] <= 0:
                problems.append(f"missing test rows for skill {skill!r}")
    if problems:
        raise Phase5ValidationError("; ".join(problems))
    return audit


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
    calibration = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    per_skill = {
        skill: threshold_for_fnr([row for row in calibration if row.get("skill_name") == skill], alpha)
        for skill in MAIN_SKILLS
    }
    return {
        "global": threshold_for_fnr(calibration, alpha),
        "per_skill": per_skill,
    }


def acceptance_search_thresholds(
    rows: list[dict[str, Any]],
    alpha: float,
    *,
    fnr_margin: float,
    max_safe_reject_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    per_skill: dict[str, float] = {}
    per_skill_diagnostics: dict[str, Any] = {}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in calibration if row.get("skill_name") == skill]
        threshold, diagnostic = _acceptance_search_one(
            skill_rows,
            alpha=alpha,
            fnr_margin=fnr_margin,
            max_safe_reject_rate=max_safe_reject_rate,
        )
        per_skill[skill] = threshold
        per_skill_diagnostics[skill] = diagnostic
    global_threshold, global_diagnostic = _acceptance_search_one(
        calibration,
        alpha=alpha,
        fnr_margin=fnr_margin,
        max_safe_reject_rate=max_safe_reject_rate,
    )
    return (
        {
            "global": global_threshold,
            "per_skill": per_skill,
        },
        {
            "mode": "acceptance_search",
            "alpha": alpha,
            "fnr_margin": fnr_margin,
            "max_safe_reject_rate": max_safe_reject_rate,
            "global": global_diagnostic,
            "per_skill": per_skill_diagnostics,
        },
    )


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
    safe_acceptance_rate = accepted_safe / safe_total if safe_total else 0.0
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
        "safe_acceptance_rate": safe_acceptance_rate,
        "safe_acceptance_proxy": safe_acceptance_rate,
        # Backward-compatible alias for older reports. New code should use
        # safe_acceptance_rate because this is not true task completion.
        "task_success_proxy": safe_acceptance_rate,
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


def safe_reject_budget_metrics(
    rows: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
    *,
    method: str,
    split: str,
    max_safe_reject_rate: float,
) -> dict[str, Any]:
    candidates = sorted({float(score_fn(row)) for row in rows})
    candidates.append(max(candidates, default=1.0) + 1e-12)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        reject = [float(score_fn(row)) >= threshold for row in rows]
        row = metrics_from_reject_flags(rows, reject, method=method, split=split)
        safe_reject_rate = 1.0 - float(row["safe_acceptance_rate"])
        if safe_reject_rate > max_safe_reject_rate + 1e-12:
            continue
        row["threshold"] = float(threshold)
        row["max_safe_reject_rate"] = max_safe_reject_rate
        key = (
            int(row["unsafe_invocation_count"]),
            safe_reject_rate,
            -float(threshold),
        )
        if best is None or key < best["_selection_key"]:
            row["_selection_key"] = key
            best = row
    if best is None:
        best = metrics_from_reject_flags(
            rows,
            [False] * len(rows),
            method=method,
            split=split,
        )
        best["threshold"] = None
        best["max_safe_reject_rate"] = max_safe_reject_rate
        best["no_feasible_threshold"] = True
    best.pop("_selection_key", None)
    return best


def relative_reduction(baseline_unsafe: int | float, method_unsafe: int | float) -> float | None:
    baseline = float(baseline_unsafe)
    if baseline <= 0:
        return None
    return (baseline - float(method_unsafe)) / baseline


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
    calibration_distribution: str = "untouched_phase4_conformal_calibration_split",
) -> list[dict[str, Any]]:
    calibration = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
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
                "calibration_distribution": calibration_distribution,
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
            "calibration_distribution": calibration_distribution,
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


def _validate_prediction_row(row: dict[str, Any], *, line_number: int, path: Path) -> None:
    for field in REQUIRED_PREDICTION_FIELDS:
        if field not in row or str(row.get(field, "")).strip() == "":
            raise Phase5ValidationError(f"{path}:{line_number}: missing required field {field!r}")
    split = split_role(row["phase4_split"])
    if split not in {"train", "critic_val", "calibration", "test"}:
        raise Phase5ValidationError(f"{path}:{line_number}: invalid phase4_split {row['phase4_split']!r}")
    if str(row["skill_name"]).strip() not in MAIN_SKILLS:
        raise Phase5ValidationError(f"{path}:{line_number}: invalid skill_name {row['skill_name']!r}")


def _parse_label(value: Any, line_number: int, path: Path) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise Phase5ValidationError(f"{path}:{line_number}: label must be exactly 0 or 1, got {value!r}")
    return int(text)


def _parse_finite_float(value: Any, field: str, line_number: int, path: Path) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase5ValidationError(f"{path}:{line_number}: {field} must be finite, got {value!r}") from exc
    if not math.isfinite(output):
        raise Phase5ValidationError(f"{path}:{line_number}: {field} must be finite, got {value!r}")
    return output


def _parse_bool(value: Any, line_number: int, path: Path) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise Phase5ValidationError(
        f"{path}:{line_number}: hard_contract_fixed_reject must be boolean or 0/1, got {value!r}"
    )


def _acceptance_search_one(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    fnr_margin: float,
    max_safe_reject_rate: float,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(row["raw_critic_risk_float"]) for row in rows})
    candidates.append(max(candidates, default=1.0) + 1e-12)
    target_alpha = max(0.0, alpha - fnr_margin)
    evaluated = [_threshold_metrics(rows, threshold) for threshold in candidates]
    strict = [
        item
        for item in evaluated
        if item["fnr"] <= target_alpha + 1e-12
        and item["safe_reject_rate"] <= max_safe_reject_rate + 1e-12
    ]
    relaxed = [
        item
        for item in evaluated
        if item["fnr"] <= alpha + 1e-12
        and item["safe_reject_rate"] <= max_safe_reject_rate + 1e-12
    ]
    used_relaxed_margin = False
    if strict:
        selected = min(strict, key=_acceptance_selection_key)
    elif relaxed:
        selected = min(relaxed, key=_acceptance_selection_key)
        used_relaxed_margin = True
    else:
        selected = min(
            evaluated,
            key=lambda item: (
                max(0.0, item["fnr"] - alpha),
                max(0.0, item["safe_reject_rate"] - max_safe_reject_rate),
                item["accepted_unsafe_count"],
                item["safe_reject_rate"],
                -item["threshold"],
            ),
        )
    rejected_reasons = {
        "fnr_margin_exceeded": sum(1 for item in evaluated if item["fnr"] > target_alpha + 1e-12),
        "alpha_exceeded": sum(1 for item in evaluated if item["fnr"] > alpha + 1e-12),
        "safe_reject_budget_exceeded": sum(
            1 for item in evaluated if item["safe_reject_rate"] > max_safe_reject_rate + 1e-12
        ),
    }
    constraints_met = (
        selected["fnr"] <= alpha + 1e-12
        and selected["safe_reject_rate"] <= max_safe_reject_rate + 1e-12
    )
    return float(selected["threshold"]), {
        "candidate_count": len(evaluated),
        "selected": selected,
        "constraints_met": constraints_met,
        "used_relaxed_margin": used_relaxed_margin,
        "target_alpha_with_margin": target_alpha,
        "rejected_candidate_reasons": rejected_reasons,
    }


def _acceptance_selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["accepted_unsafe_count"]),
        int(item["safe_reject_count"]),
        float(item["safe_reject_rate"]),
        -float(item["threshold"]),
    )


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    reject = [float(row["raw_critic_risk_float"]) >= threshold for row in rows]
    labels = [int(row["label_int"]) for row in rows]
    unsafe_total = sum(labels)
    safe_total = len(rows) - unsafe_total
    accepted_unsafe = sum(label for label, is_rejected in zip(labels, reject) if label == 1 and not is_rejected)
    safe_reject = sum(1 for label, is_rejected in zip(labels, reject) if label == 0 and is_rejected)
    return {
        "threshold": float(threshold),
        "count": len(rows),
        "safe_total": safe_total,
        "unsafe_total": unsafe_total,
        "accepted_unsafe_count": accepted_unsafe,
        "safe_reject_count": safe_reject,
        "fnr": accepted_unsafe / unsafe_total if unsafe_total else 0.0,
        "safe_reject_rate": safe_reject / safe_total if safe_total else 0.0,
    }

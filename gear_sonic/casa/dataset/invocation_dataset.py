"""Build CASA state-skill-label invocation datasets from Phase 2 rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAIN_SKILLS = ("walk", "turn", "gesture", "passive")
SKILL_TYPE_MAP = {name: index for index, name in enumerate(MAIN_SKILLS)}
LABEL_MAP = {"safe": 0, "unsafe": 1}
SIDE_MAP = {"left": 0, "right": 1, "both": 2}
MODE_MAP = {"stop": 0, "wait": 1}
PER_SKILL_POSITIVE_RATE_MIN = 0.05

BASE_FIELDS = [
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
    "left_foot_contact",
    "right_foot_contact",
    "self_collision",
    "self_collision_pair_count",
]
ENV_FIELDS = [
    "min_user_distance",
    "min_arm_user_distance",
    "min_obstacle_distance",
]
ENV_FLAG_FIELDS = [
    "external_collision_user",
    "external_collision_obstacle",
]


@dataclass
class DatasetBuildResult:
    rows_all: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    features: np.ndarray
    labels: np.ndarray
    skill_type_ids: np.ndarray
    splits: np.ndarray
    feature_schema: dict[str, Any]
    summary: dict[str, Any]


def build_invocation_dataset(
    *,
    episodes_roots: list[Path],
    target_samples: int,
    pre_window_seconds: float,
    exclude_unverified: bool,
    min_per_skill: int,
    positive_rate_min: float,
    positive_rate_max: float,
    seed: int = 1234,
    exclude_current_violations: bool = True,
    current_violation_grace_seconds: float = 0.05,
) -> DatasetBuildResult:
    raw_rows, feature_records, feature_schema = _collect_feature_records(
        episodes_roots=episodes_roots,
        pre_window_seconds=pre_window_seconds,
    )
    usable_indices = [
        index
        for index, row in enumerate(raw_rows)
        if row["safe_label"] in LABEL_MAP
        and (not exclude_unverified or row["safe_label"] != "unverified")
        and not (
            exclude_current_violations
            and row["safe_label"] == "unsafe"
            and _float(row.get("time_to_violation"), float("inf")) <= current_violation_grace_seconds
        )
    ]
    selected_indices, selection_warnings = _select_indices(
        raw_rows,
        usable_indices,
        target_samples=target_samples,
        min_per_skill=min_per_skill,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        seed=seed,
    )

    selected_rows = [raw_rows[index] for index in selected_indices]
    feature_names = feature_schema["feature_names"]
    features = np.asarray(
        [[float(feature_records[index].get(name, 0.0)) for name in feature_names] for index in selected_indices],
        dtype=np.float32,
    )
    labels = np.asarray([LABEL_MAP[row["safe_label"]] for row in selected_rows], dtype=np.int64)
    skill_type_ids = np.asarray([SKILL_TYPE_MAP.get(row["skill_name"], -1) for row in selected_rows], dtype=np.int64)
    splits = np.asarray(_assign_splits(selected_rows), dtype="<U5")

    summary = _build_summary(
        raw_rows=raw_rows,
        selected_rows=selected_rows,
        target_samples=target_samples,
        min_per_skill=min_per_skill,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        pre_window_seconds=pre_window_seconds,
        exclude_current_violations=exclude_current_violations,
        current_violation_grace_seconds=current_violation_grace_seconds,
        selection_warnings=selection_warnings,
        features_shape=list(features.shape),
    )
    feature_schema = dict(feature_schema)
    feature_schema.update(
        {
            "skill_type_map": SKILL_TYPE_MAP,
            "label_map": LABEL_MAP,
            "side_map": SIDE_MAP,
            "mode_map": MODE_MAP,
        }
    )
    return DatasetBuildResult(
        rows_all=raw_rows,
        rows=selected_rows,
        features=features,
        labels=labels,
        skill_type_ids=skill_type_ids,
        splits=splits,
        feature_schema=feature_schema,
        summary=summary,
    )


def write_dataset_outputs(result: DatasetBuildResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "invocations_all.csv", result.rows_all)
    _write_csv(output_dir / "invocations.csv", result.rows)
    np.savez_compressed(
        output_dir / "features.npz",
        X=result.features,
        y=result.labels,
        skill_type=result.skill_type_ids,
        split=result.splits,
        feature_names=np.asarray(result.feature_schema["feature_names"]),
    )
    (output_dir / "feature_schema.json").write_text(json.dumps(result.feature_schema, indent=2, sort_keys=True) + "\n")
    (output_dir / "dataset_summary.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")


def load_dataset_npz(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    arrays = np.load(dataset_dir / "features.npz", allow_pickle=False)
    schema = json.loads((dataset_dir / "feature_schema.json").read_text())
    rows = list(csv.DictReader((dataset_dir / "invocations.csv").open(newline="")))
    return {
        "X": arrays["X"],
        "y": arrays["y"],
        "skill_type": arrays["skill_type"],
        "split": arrays["split"].astype(str),
        "feature_names": [str(item) for item in arrays["feature_names"]],
        "schema": schema,
        "rows": rows,
    }


def _collect_feature_records(
    *,
    episodes_roots: list[Path],
    pre_window_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    feature_records: list[dict[str, float]] = []
    body_q_fields: set[str] = set()
    body_dq_fields: set[str] = set()

    summary_paths = _discover_summary_paths(episodes_roots)
    for summary_path in summary_paths:
        summary = _read_json(summary_path)
        episode_dir = summary_path.parent
        sim_state_csv = _resolve_log_path(summary.get("sim_state_csv"), episode_dir / "sim_log" / "sim_state.csv")
        skill_events_csv = _resolve_log_path(summary.get("skill_events_csv"), episode_dir / "skill_events.csv")
        if not sim_state_csv.exists() or not skill_events_csv.exists():
            continue
        sim_rows = _read_sim_rows(sim_state_csv)
        if not sim_rows:
            continue
        skill_rows = _read_skill_events(skill_events_csv)
        skill_by_idx = {str(row.get("skill_idx")): row for row in skill_rows}
        labels = sorted(summary.get("skill_labels", []), key=lambda item: int(_float(item.get("skill_idx"), 0.0)))
        scene_props = summary.get("scene_props", {}) or {}
        source_root = _source_root_name(summary_path)
        previous_skill_event: dict[str, Any] | None = None

        for label in labels:
            skill_idx = str(label.get("skill_idx"))
            skill_event = skill_by_idx.get(skill_idx, {})
            skill_name = str(label.get("skill_name") or skill_event.get("skill_name") or "")
            if skill_name not in SKILL_TYPE_MAP:
                continue
            start_wall = _float(label.get("window_start_wall_time"), _float(skill_event.get("start_wall_time")))
            end_wall = _float(skill_event.get("end_wall_time"), start_wall)
            pre_rows = [row for row in sim_rows if start_wall - pre_window_seconds <= _float(row.get("wall_time")) <= start_wall]
            if not pre_rows:
                before = [row for row in sim_rows if _float(row.get("wall_time")) <= start_wall]
                if not before:
                    previous_skill_event = skill_event
                    continue
                pre_rows = before[-1:]
            current_row = pre_rows[-1]
            body_q_fields.update(field for field in current_row if field.startswith("body_q_"))
            body_dq_fields.update(field for field in current_row if field.startswith("body_dq_"))
            params = _json_dict(skill_event.get("params_json"))
            evidence = _json_dict(skill_event.get("evidence_json"))
            feature_record = _extract_features(
                pre_rows=pre_rows,
                current_row=current_row,
                params=params,
                evidence=evidence,
                skill_name=skill_name,
                skill_event=skill_event,
                previous_skill_event=previous_skill_event,
                scene_props=scene_props,
            )
            run_id = str(label.get("run_id") or summary.get("run_id") or "")
            rollout_id = str(label.get("episode_id") or summary.get("rollout_id") or episode_dir.name)
            split_group = f"{source_root}:{run_id}:{rollout_id}"
            row_id = f"{split_group}:{skill_idx}"
            raw_rows.append(
                {
                    "sample_id": row_id,
                    "source_root": source_root,
                    "run_id": run_id,
                    "rollout_id": rollout_id,
                    "skill_idx": skill_idx,
                    "skill_name": skill_name,
                    "safe_label": str(label.get("safe_label") or ""),
                    "label": LABEL_MAP.get(str(label.get("safe_label") or ""), ""),
                    "triggered_violation_types": json.dumps(label.get("triggered_violation_types", []), sort_keys=True),
                    "time_to_violation": "" if label.get("time_to_violation") is None else label.get("time_to_violation"),
                    "violation_time_bin": label.get("violation_time_bin") or "",
                    "coverage": label.get("coverage", ""),
                    "params_json": json.dumps(params, sort_keys=True),
                    "status": skill_event.get("status", ""),
                    "termination_reason": skill_event.get("termination_reason", ""),
                    "estimated_duration": skill_event.get("estimated_duration", ""),
                    "actual_duration": skill_event.get("actual_duration", ""),
                    "start_wall_time": start_wall,
                    "end_wall_time": end_wall,
                    "pre_state_wall_time": current_row.get("wall_time", ""),
                    "sim_state_csv": str(sim_state_csv),
                    "skill_events_csv": str(skill_events_csv),
                    "summary_path": str(summary_path),
                    "split_group": split_group,
                }
            )
            feature_records.append(feature_record)
            previous_skill_event = skill_event

    feature_schema = _feature_schema(body_q_fields, body_dq_fields)
    feature_names = feature_schema["feature_names"]
    for record in feature_records:
        for name in feature_names:
            record.setdefault(name, 0.0)
    return raw_rows, feature_records, feature_schema


def _discover_summary_paths(episodes_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for root in episodes_roots:
        root = Path(root)
        candidates = []
        direct_summary = root / "rollout_summary.json"
        if direct_summary.exists():
            candidates.append(direct_summary)
        candidates.extend(sorted(root.glob("episode_*/rollout_summary.json")))
        if not candidates:
            candidates = sorted(root.glob("branches/*/rollout_summary.json"))
        if not candidates and (root / "merge_manifest.json").exists():
            manifest = _read_json(root / "merge_manifest.json")
            for source in manifest.get("source_runs", []):
                candidates.extend(sorted(Path(source).glob("episode_*/rollout_summary.json")))
        if not candidates:
            candidates = sorted(root.rglob("episode_*/rollout_summary.json"))
        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _extract_features(
    *,
    pre_rows: list[dict[str, Any]],
    current_row: dict[str, Any],
    params: dict[str, Any],
    evidence: dict[str, Any],
    skill_name: str,
    skill_event: dict[str, Any],
    previous_skill_event: dict[str, Any] | None,
    scene_props: dict[str, Any],
) -> dict[str, float]:
    features: dict[str, float] = {}
    _add_stats(features, "robot", pre_rows, BASE_FIELDS, include_current=True)
    _add_joint_stats(features, pre_rows)
    _add_stats(features, "env", pre_rows, ENV_FIELDS, include_current=True, include_min=True)
    _add_flag_stats(features, "env", pre_rows, ENV_FLAG_FIELDS)
    _add_scene_features(features, current_row, scene_props)
    _add_runtime_features(features, pre_rows, skill_event, previous_skill_event)
    _add_skill_features(features, skill_name, params)
    return features


def _add_stats(
    features: dict[str, float],
    group: str,
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    include_current: bool,
    include_min: bool = False,
) -> None:
    for field in fields:
        values = [_float(row.get(field), math.nan) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        if not values:
            values = [0.0]
        prefix = f"{group}/{field}"
        if include_current:
            features[f"{prefix}/current"] = values[-1]
        features[f"{prefix}/mean"] = float(np.mean(values))
        features[f"{prefix}/std"] = float(np.std(values))
        features[f"{prefix}/max_abs"] = max(abs(value) for value in values)
        if include_min:
            features[f"{prefix}/min"] = min(values)


def _add_joint_stats(features: dict[str, float], rows: list[dict[str, Any]]) -> None:
    fields = sorted(
        [field for field in rows[-1] if field.startswith("body_q_") or field.startswith("body_dq_")],
        key=_joint_sort_key,
    )
    for field in fields:
        values = [_float(row.get(field), 0.0) for row in rows]
        prefix = f"robot/{field}"
        features[f"{prefix}/current"] = values[-1] if values else 0.0
        features[f"{prefix}/mean"] = float(np.mean(values)) if values else 0.0
        features[f"{prefix}/std"] = float(np.std(values)) if values else 0.0
        features[f"{prefix}/max_abs"] = max((abs(value) for value in values), default=0.0)


def _add_flag_stats(features: dict[str, float], group: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    for field in fields:
        values = [_float(row.get(field), 0.0) for row in rows]
        prefix = f"{group}/{field}"
        features[f"{prefix}/current"] = values[-1] if values else 0.0
        features[f"{prefix}/any"] = 1.0 if any(value > 0.5 for value in values) else 0.0
        features[f"{prefix}/mean"] = float(np.mean(values)) if values else 0.0


def _add_scene_features(features: dict[str, float], current_row: dict[str, Any], scene_props: dict[str, Any]) -> None:
    base_x = _float(current_row.get("base_pos_x"), 0.0)
    base_y = _float(current_row.get("base_pos_y"), 0.0)
    base_z = _float(current_row.get("base_pos_z"), 0.0)
    placements = scene_props.get("placements", {}) if isinstance(scene_props, dict) else {}
    nearest_user = _nearest_prop(base_x, base_y, base_z, placements, "user_proxy_")
    nearest_obstacle = _nearest_prop(base_x, base_y, base_z, placements, "obstacle_")
    for prefix, nearest in [("env/nearest_user", nearest_user), ("env/nearest_obstacle", nearest_obstacle)]:
        if nearest is None:
            features[f"{prefix}/available"] = 0.0
            features[f"{prefix}/rel_x"] = 0.0
            features[f"{prefix}/rel_y"] = 0.0
            features[f"{prefix}/rel_z"] = 0.0
            features[f"{prefix}/dist_xy"] = 10.0
        else:
            rel_x, rel_y, rel_z = nearest
            features[f"{prefix}/available"] = 1.0
            features[f"{prefix}/rel_x"] = rel_x
            features[f"{prefix}/rel_y"] = rel_y
            features[f"{prefix}/rel_z"] = rel_z
            features[f"{prefix}/dist_xy"] = math.hypot(rel_x, rel_y)


def _add_runtime_features(
    features: dict[str, float],
    rows: list[dict[str, Any]],
    skill_event: dict[str, Any],
    previous_skill_event: dict[str, Any] | None,
) -> None:
    overrun = [_float(row.get("control_loop_overrun"), 0.0) for row in rows]
    features["runtime/control_overrun_ratio"] = float(np.mean(overrun)) if overrun else 0.0
    previous_status = str((previous_skill_event or {}).get("status") or "")
    for status in ["success", "failed", "unverified", "dry_run", "published"]:
        features[f"runtime/previous_status_{status}"] = 1.0 if previous_status == status else 0.0


def _add_skill_features(features: dict[str, float], skill_name: str, params: dict[str, Any]) -> None:
    skill_id = SKILL_TYPE_MAP.get(skill_name, -1)
    features["skill/type_id"] = float(skill_id)
    for name, index in SKILL_TYPE_MAP.items():
        features[f"skill/is_{name}"] = 1.0 if index == skill_id else 0.0
    duration = _float(params.get("duration"), 0.0)
    features["skill/duration"] = duration
    features["skill/walk_vx"] = _float(params.get("vx"), 0.0)
    features["skill/walk_vy"] = _float(params.get("vy"), 0.0)
    features["skill/walk_speed"] = _float(params.get("speed"), -1.0)
    features["skill/facing_yaw_deg"] = _float(params.get("facing_yaw_deg", params.get("face_yaw_deg")), 0.0)
    features["skill/turn_face_yaw_deg"] = _float(params.get("face_yaw_deg"), 0.0)
    features["skill/gesture_amplitude"] = _float(params.get("amplitude"), 0.0)
    features["skill/gesture_frequency"] = _float(params.get("frequency"), 0.0)
    side = str(params.get("side", ""))
    for name, index in SIDE_MAP.items():
        features[f"skill/gesture_side_{name}"] = 1.0 if SIDE_MAP.get(side, -1) == index else 0.0
    mode = str(params.get("mode", ""))
    for name, index in MODE_MAP.items():
        features[f"skill/passive_mode_{name}"] = 1.0 if MODE_MAP.get(mode, -1) == index else 0.0


def _feature_schema(body_q_fields: set[str], body_dq_fields: set[str]) -> dict[str, Any]:
    names: list[str] = []
    groups: dict[str, list[int]] = defaultdict(list)

    def add(name: str) -> None:
        if name in names:
            return
        groups[name.split("/", 1)[0]].append(len(names))
        names.append(name)

    for field in BASE_FIELDS:
        for stat in ["current", "mean", "std", "max_abs"]:
            add(f"robot/{field}/{stat}")
    for field in sorted(body_q_fields | body_dq_fields, key=_joint_sort_key):
        for stat in ["current", "mean", "std", "max_abs"]:
            add(f"robot/{field}/{stat}")
    for field in ENV_FIELDS:
        for stat in ["current", "mean", "std", "max_abs", "min"]:
            add(f"env/{field}/{stat}")
    for field in ENV_FLAG_FIELDS:
        for stat in ["current", "any", "mean"]:
            add(f"env/{field}/{stat}")
    for prefix in ["env/nearest_user", "env/nearest_obstacle"]:
        for stat in ["available", "rel_x", "rel_y", "rel_z", "dist_xy"]:
            add(f"{prefix}/{stat}")
    for name in [
        "runtime/control_overrun_ratio",
        "runtime/previous_status_success",
        "runtime/previous_status_failed",
        "runtime/previous_status_unverified",
        "runtime/previous_status_dry_run",
        "runtime/previous_status_published",
    ]:
        add(name)
    for name in [
        "skill/type_id",
        "skill/is_walk",
        "skill/is_turn",
        "skill/is_gesture",
        "skill/is_passive",
        "skill/duration",
        "skill/walk_vx",
        "skill/walk_vy",
        "skill/walk_speed",
        "skill/facing_yaw_deg",
        "skill/turn_face_yaw_deg",
        "skill/gesture_amplitude",
        "skill/gesture_frequency",
        "skill/gesture_side_left",
        "skill/gesture_side_right",
        "skill/gesture_side_both",
        "skill/passive_mode_stop",
        "skill/passive_mode_wait",
    ]:
        add(name)
    return {"feature_names": names, "groups": {key: value for key, value in groups.items()}}


def _select_indices(
    rows: list[dict[str, Any]],
    usable_indices: list[int],
    *,
    target_samples: int,
    min_per_skill: int,
    positive_rate_min: float,
    positive_rate_max: float,
    seed: int,
) -> tuple[list[int], list[str]]:
    rng = random.Random(seed)
    warnings: list[str] = []
    if not usable_indices:
        return [], ["no_usable_samples"]
    target = min(target_samples, len(usable_indices)) if target_samples > 0 else len(usable_indices)
    if target < target_samples:
        warnings.append(f"usable_samples_below_target:{len(usable_indices)}<{target_samples}")

    by_skill = defaultdict(list)
    for index in usable_indices:
        by_skill[rows[index]["skill_name"]].append(index)
    selected: list[int] = []
    selected_set: set[int] = set()
    for skill in MAIN_SKILLS:
        candidates = by_skill.get(skill, [])
        if len(candidates) < min_per_skill:
            warnings.append(f"{skill}_below_min_per_skill:{len(candidates)}<{min_per_skill}")
        take = min(min_per_skill, len(candidates), max(0, target - len(selected)))
        sampled = _stable_sample(candidates, take, rng)
        selected.extend(sampled)
        selected_set.update(sampled)

    remaining = [index for index in usable_indices if index not in selected_set]
    current_pos = sum(1 for index in selected if rows[index]["safe_label"] == "unsafe")
    desired_pos_max = int(math.floor(target * positive_rate_max))
    desired_pos_min = int(math.ceil(target * positive_rate_min))
    negative_remaining = [index for index in remaining if rows[index]["safe_label"] == "safe"]
    positive_remaining = [index for index in remaining if rows[index]["safe_label"] == "unsafe"]

    rng.shuffle(negative_remaining)
    rng.shuffle(positive_remaining)
    while len(selected) < target and current_pos > desired_pos_max and negative_remaining:
        index = negative_remaining.pop()
        selected.append(index)
        selected_set.add(index)
    fill_pool = [index for index in remaining if index not in selected_set]
    rng.shuffle(fill_pool)
    for index in fill_pool:
        if len(selected) >= target:
            break
        if rows[index]["safe_label"] == "unsafe" and current_pos >= desired_pos_max and negative_remaining:
            continue
        selected.append(index)
        if rows[index]["safe_label"] == "unsafe":
            current_pos += 1
    repair_warnings = _repair_per_skill_positive_rates(
        rows=rows,
        usable_indices=usable_indices,
        selected=selected,
        positive_rate_max=positive_rate_max,
        rng=rng,
    )
    warnings.extend(repair_warnings)
    current_pos = sum(1 for index in selected if rows[index]["safe_label"] == "unsafe")
    if len(selected) < target:
        warnings.append(f"selected_below_target:{len(selected)}<{target}")
    positive_rate = current_pos / len(selected) if selected else 0.0
    min_positive_rate = desired_pos_min / target if target else 0.0
    if positive_rate < min_positive_rate:
        warnings.append(f"positive_rate_below_min:{positive_rate:.4f}<{positive_rate_min}")
    if positive_rate > positive_rate_max:
        warnings.append(f"positive_rate_above_max:{positive_rate:.4f}>{positive_rate_max}")
    return selected[:target], warnings


def _repair_per_skill_positive_rates(
    *,
    rows: list[dict[str, Any]],
    usable_indices: list[int],
    selected: list[int],
    positive_rate_max: float,
    rng: random.Random,
) -> list[str]:
    warnings: list[str] = []
    if not selected:
        return warnings

    selected_set = set(selected)
    desired_pos_max = int(math.floor(len(selected) * positive_rate_max))
    current_pos = sum(1 for index in selected if rows[index]["safe_label"] == "unsafe")
    unused_positive_by_skill: dict[str, list[int]] = defaultdict(list)
    for index in usable_indices:
        row = rows[index]
        if index not in selected_set and row["safe_label"] == "unsafe":
            unused_positive_by_skill[row["skill_name"]].append(index)

    for skill in MAIN_SKILLS:
        skill_selected = [index for index in selected if rows[index]["skill_name"] == skill]
        if not skill_selected:
            continue
        skill_pos = sum(1 for index in skill_selected if rows[index]["safe_label"] == "unsafe")
        required_pos = int(math.ceil(len(skill_selected) * PER_SKILL_POSITIVE_RATE_MIN))
        deficit = required_pos - skill_pos
        if deficit <= 0:
            continue

        removable_safe = [index for index in skill_selected if rows[index]["safe_label"] == "safe"]
        addable_positive = unused_positive_by_skill.get(skill, [])
        rng.shuffle(removable_safe)
        rng.shuffle(addable_positive)
        swapped = 0
        while deficit > 0 and removable_safe and addable_positive and current_pos < desired_pos_max:
            remove_index = removable_safe.pop()
            add_index = addable_positive.pop()
            selected[selected.index(remove_index)] = add_index
            selected_set.remove(remove_index)
            selected_set.add(add_index)
            current_pos += 1
            deficit -= 1
            swapped += 1
        if deficit > 0:
            warnings.append(
                f"{skill}_positive_rate_repair_incomplete:"
                f"swapped={swapped},remaining={deficit},required={required_pos}"
            )
    return warnings


def _stable_sample(values: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    values = list(values)
    rng.shuffle(values)
    return values[:count]


def _assign_splits(rows: list[dict[str, Any]]) -> list[str]:
    groups = sorted({row["split_group"] for row in rows})
    group_to_split = {}
    for group in groups:
        bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest(), 16) % 100
        if bucket < 70:
            split = "train"
        elif bucket < 85:
            split = "val"
        else:
            split = "test"
        group_to_split[group] = split
    return [group_to_split[row["split_group"]] for row in rows]


def _build_summary(
    *,
    raw_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    target_samples: int,
    min_per_skill: int,
    positive_rate_min: float,
    positive_rate_max: float,
    pre_window_seconds: float,
    exclude_current_violations: bool,
    current_violation_grace_seconds: float,
    selection_warnings: list[str],
    features_shape: list[int],
) -> dict[str, Any]:
    label_complete = [row for row in raw_rows if row["safe_label"] in LABEL_MAP]
    current_violation_excluded = [
        row
        for row in label_complete
        if row["safe_label"] == "unsafe"
        and _float(row.get("time_to_violation"), float("inf")) <= current_violation_grace_seconds
    ]
    usable = [
        row
        for row in label_complete
        if not (
            exclude_current_violations
            and row["safe_label"] == "unsafe"
            and _float(row.get("time_to_violation"), float("inf")) <= current_violation_grace_seconds
        )
    ]
    selected_counts = Counter(row["safe_label"] for row in selected_rows)
    per_skill = {
        skill: dict(Counter(row["safe_label"] for row in selected_rows if row["skill_name"] == skill))
        for skill in MAIN_SKILLS
    }
    total = len(selected_rows)
    unsafe = selected_counts.get("unsafe", 0)
    overall_positive_rate = unsafe / total if total else 0.0
    per_skill_positive_rate = {}
    for skill, counts in per_skill.items():
        skill_total = sum(counts.values())
        per_skill_positive_rate[skill] = counts.get("unsafe", 0) / skill_total if skill_total else 0.0
    checks = {
        "target_samples_met": total == target_samples,
        "save_completeness_ge_0p98": (len(label_complete) / len(raw_rows) >= 0.98) if raw_rows else False,
        "overall_positive_rate_in_range": positive_rate_min <= overall_positive_rate <= positive_rate_max,
        "per_skill_min_samples_met": all(sum(per_skill[skill].values()) >= min_per_skill for skill in MAIN_SKILLS),
        "per_skill_positive_rate_ge_0p05": all(
            per_skill_positive_rate.get(skill, 0.0) >= PER_SKILL_POSITIVE_RATE_MIN for skill in MAIN_SKILLS
        ),
    }
    return {
        "raw_samples": len(raw_rows),
        "label_complete_samples": len(label_complete),
        "label_complete_rate": (len(label_complete) / len(raw_rows)) if raw_rows else 0.0,
        "usable_samples": len(usable),
        "current_violation_excluded_samples": len(current_violation_excluded)
        if exclude_current_violations
        else 0,
        "exclude_current_violations": exclude_current_violations,
        "current_violation_grace_seconds": current_violation_grace_seconds,
        "selected_samples": total,
        "target_samples": target_samples,
        "features_shape": features_shape,
        "pre_window_seconds": pre_window_seconds,
        "label_counts": dict(selected_counts),
        "overall_positive_rate": overall_positive_rate,
        "per_skill_label_counts": per_skill,
        "per_skill_positive_rate": per_skill_positive_rate,
        "checks": checks,
        "go_criteria_passed": all(checks.values()),
        "selection_warnings": selection_warnings,
        "recommendations": _recommendations(per_skill, per_skill_positive_rate, min_per_skill),
    }


def _recommendations(per_skill: dict[str, dict[str, int]], rates: dict[str, float], min_per_skill: int) -> list[str]:
    recs = []
    for skill in MAIN_SKILLS:
        total = sum(per_skill.get(skill, {}).values())
        if total < min_per_skill:
            recs.append(f"collect_{skill}_heavy_batch")
        if rates.get(skill, 0.0) < 0.05:
            recs.append(f"collect_{skill}_hard_cases")
    return recs


def _nearest_prop(base_x: float, base_y: float, base_z: float, placements: dict, prefix: str) -> tuple[float, float, float] | None:
    nearest = None
    nearest_dist = float("inf")
    for name, placement in placements.items():
        if not str(name).startswith(prefix) or not placement.get("enabled", True):
            continue
        position = placement.get("position", [0.0, 0.0, -10.0])
        rel = (float(position[0]) - base_x, float(position[1]) - base_y, float(position[2]) - base_z)
        dist = math.hypot(rel[0], rel[1])
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = rel
    return nearest


def _read_sim_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _read_skill_events(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _resolve_log_path(value: Any, fallback: Path) -> Path:
    if value:
        path = Path(str(value))
        if path.exists():
            return path
        repo_path = Path.cwd() / path
        if repo_path.exists():
            return repo_path
    return fallback


def _source_root_name(summary_path: Path) -> str:
    return summary_path.parent.parent.name


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "no", "n"}:
        return 0.0
    return 0.0


def _joint_sort_key(field: str) -> tuple[str, int]:
    prefix, _, value = field.rpartition("_")
    try:
        return prefix, int(value)
    except ValueError:
        return prefix, -1

#!/usr/bin/env python3
"""Strict current-state audit for Plan A completion.

This script is intentionally evidence-based: it only marks a check passed when
the referenced file currently exists and contains the expected pass criteria.
Phase 2 human review is recorded as the single allowed manual exception.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT = Path("/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa")
REPO = Path("/tmp/casa_upload_worktree")
ONLINE_ROOT = OUT / "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602"
METHODS = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]
SEEDS = [1234, 1235, 1236, 1237, 1238]
SKILLS = ["walk", "turn", "gesture", "passive"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def status(ok: bool, evidence: Any = None, reason: str = "") -> dict[str, Any]:
    return {"status": "PASS" if ok else "BLOCKED", "evidence": evidence, "reason": reason}


def phase0() -> dict[str, Any]:
    full30_path = OUT / "phase0_sanity/casa_phase0_full30_video_20260516_110120/full30/sanity_summary.json"
    repeat_path = OUT / "phase0_sanity/casa_phase0_headless_20260516_093349/repeat20/repeatability_summary.json"
    full30 = load_json(full30_path)
    repeat = load_json(repeat_path)
    by_skill = repeat.get("by_skill", {})
    repeat_ok = (
        repeat.get("go_criteria_passed") is True
        and repeat.get("dry_run") is False
        and repeat.get("trials_per_skill", 0) >= 20
        and all(by_skill.get(skill, {}).get("total", 0) >= 20 for skill in SKILLS)
        and repeat.get("overall", {}).get("success_rate", 0.0) >= 0.90
    )
    full30_ok = (
        full30.get("go_criteria_passed") is True
        and full30.get("dry_run") is False
        and full30.get("duration_seconds", 0.0) >= 1800
        and full30.get("overall", {}).get("success_rate", 0.0) >= 0.90
    )
    return {
        "overall": "PASS" if repeat_ok and full30_ok else "BLOCKED",
        "checks": {
            "30min_stable_real_run": status(full30_ok, {"path": str(full30_path), "summary": full30}),
            "20_repeats_each_skill": status(repeat_ok, {"path": str(repeat_path), "summary": repeat}),
        },
    }


def phase1() -> dict[str, Any]:
    files = {
        "WalkSkill": REPO / "gear_sonic/casa/skills/walk.py",
        "PassiveSkill": REPO / "gear_sonic/casa/skills/passive.py",
        "TurnSkill": REPO / "gear_sonic/casa/skills/turn.py",
        "GestureSkill": REPO / "gear_sonic/casa/skills/gesture.py",
        "SkillRegistry": REPO / "gear_sonic/casa/skills/registry.py",
        "SkillLogger": REPO / "gear_sonic/casa/loggers/skill_logger.py",
        "CalibrationGroups": REPO / "gear_sonic/casa/skills/calibration_groups.yaml",
    }
    existing = {name: path.exists() for name, path in files.items()}
    texts = {name: path.read_text() if path.exists() else "" for name, path in files.items()}
    class_ok = all(existing[name] and f"class {name}" in texts[name] for name in ["WalkSkill", "PassiveSkill", "TurnSkill", "GestureSkill"])
    registry_ok = existing["SkillRegistry"] and "default_skill_registry" in texts["SkillRegistry"]
    logging_fields = [
        "skill_name",  # Plan A skill_type equivalent in the first implementation.
        "params_json",  # Serialized skill_params.
        "start_wall_time",
        "estimated_duration",
        "status",  # execution_status.
        "termination_reason",
    ]
    logging_ok = existing["SkillLogger"] and all(field in texts["SkillLogger"] for field in logging_fields)
    groups_ok = existing["CalibrationGroups"] and all(skill in texts["CalibrationGroups"] for skill in SKILLS)
    return {
        "overall": "PASS" if class_ok and registry_ok and logging_ok and groups_ok else "BLOCKED",
        "checks": {
            "four_skill_wrappers": status(class_ok, {k: str(v) for k, v in files.items() if k.endswith("Skill")}),
            "skill_registry": status(registry_ok, str(files["SkillRegistry"])),
            "skill_log_fields": status(logging_ok, {"path": str(files["SkillLogger"]), "field_mapping": logging_fields}),
            "calibration_groups": status(groups_ok, str(files["CalibrationGroups"])),
        },
    }


def phase2() -> dict[str, Any]:
    violations = REPO / "gear_sonic/casa/oracle/violations.py"
    oracle = REPO / "gear_sonic/casa/oracle/oracle.py"
    dataset = OUT / "phase2_clean_visual_1000_v2_20260519/dataset_summary_v2.json"
    vtext = violations.read_text() if violations.exists() else ""
    otext = oracle.read_text() if oracle.exists() else ""
    d = load_json(dataset)
    types = ["collision", "near_collision", "fall", "human_distance_violation", "unsafe_gesture", "runtime_timeout"]
    type_ok = all(t in vtext or t.upper() in vtext for t in types)
    fields_ok = all(field in otext for field in ["time_to_violation", "violation_time_bin", "min_user_distance", "base_height"])
    visual_dataset_ok = d.get("selected_total") == 1000 and not d.get("integrity_errors")
    return {
        "overall": "PASS_WITH_ALLOWED_EXCEPTION" if type_ok and fields_ok and visual_dataset_ok else "BLOCKED",
        "allowed_exception": "Manual independent Phase2 human review is intentionally excluded by the active objective.",
        "checks": {
            "six_violation_types_implemented": status(type_ok, str(violations)),
            "rollout_logger_fields_present": status(fields_ok, str(oracle)),
            "phase2_visual_dataset_integrity": status(visual_dataset_ok, {"path": str(dataset), "summary": d}),
        },
    }


def audit_file(path: Path, pass_keys: list[tuple[str, Any]]) -> tuple[bool, dict[str, Any]]:
    data = load_json(path)
    ok = path.exists()
    for dotted, expected in pass_keys:
        cur: Any = data
        for part in dotted.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        ok = ok and cur == expected
    return ok, data


def phase3() -> dict[str, Any]:
    path = OUT / "phase3_feasibility/phase3_hybrid_phase2v2_manifest_20260521/phase3_acceptance_audit.json"
    data = load_json(path)
    table = data.get("acceptance_table", [])
    blocked = data.get("blocking_reasons", ["missing"])
    all_nonblocking = not blocked and all(row.get("status") in {"PASS", "PASS_WITH_WARNING"} for row in table)
    return {"overall": "PASS" if all_nonblocking else "BLOCKED", "checks": {"phase3_acceptance_audit": status(all_nonblocking, {"path": str(path), "blocking_reasons": blocked})}}


def phase4() -> dict[str, Any]:
    path = OUT / "phase4_dataset_v1_strict_50k_20260522/phase4_acceptance_audit.json"
    data = load_json(path)
    checks = data.get("dataset_checks", {})
    required = [
        "total_samples_ge_50k",
        "per_skill_samples_ge_10k",
        "per_skill_dangerous_ge_600",
        "calibration_per_skill_dangerous_ge_200",
        "split_group_no_leakage",
        "label_missing_rate_le_0p02",
        "critical_field_missing_rate_le_0p02",
        "unsafe_positive_rate_in_10_50pct",
    ]
    hc = data.get("hard_contract_filtered_calibration", {}).get("checks", {})
    ok = (
        path.exists()
        and data.get("blocking_reasons") == []
        and data.get("overall_status") == "PASS_STRICT"
        and all(checks.get(key) is True for key in required)
        and all(hc.get(key) is True for key in ["calibration_rows_present", "all_calibration_rows_hard_contract_accepted", "each_main_skill_present", "each_skill_unsafe_ge_min"])
    )
    return {"overall": "PASS" if ok else "BLOCKED", "checks": {"phase4_strict_acceptance": status(ok, {"path": str(path), "required_checks": required})}}


def phase5_offline() -> dict[str, Any]:
    root = OUT / "phase5_conformal_baselines_20260522/hc_filtered_online_calibration_20260601"
    acceptance = load_json(root / "phase5_acceptance_audit.json")
    calibration = load_json(root / "calibration_audit.json")
    manifest = load_json(root / "artifacts/phase5_artifact_manifest.json")
    acceptance_ok = (root / "phase5_acceptance_audit.json").exists() and acceptance.get("status") == "PASS_STRICT" and acceptance.get("go") is True and not acceptance.get("blocking_reasons")
    calibration_ok = (root / "calibration_audit.json").exists() and calibration.get("go_criteria_passed") is True and calibration.get("checks", {}).get("calibration_hard_contract_filtered") is True
    artifact_files = [root / item.get("path", "") for item in manifest.get("artifacts", []) if isinstance(item, dict)]
    artifacts_ok = bool(artifact_files) and all(path.exists() and path.stat().st_size > 0 for path in artifact_files)
    return {
        "overall": "PASS" if acceptance_ok and calibration_ok and artifacts_ok else "BLOCKED",
        "checks": {
            "offline_phase5_acceptance": status(acceptance_ok, {"path": str(root / "phase5_acceptance_audit.json"), "summary": {k: acceptance.get(k) for k in ["status", "go", "blocking_reasons"]}}),
            "hc_filtered_calibration": status(calibration_ok, {"path": str(root / "calibration_audit.json")}),
            "visual_and_demo_artifacts": status(artifacts_ok, {"manifest": str(root / "artifacts/phase5_artifact_manifest.json"), "artifact_count": len(artifact_files)}),
        },
    }


def online_progress() -> dict[str, Any]:
    completed: set[tuple[str, int, int]] = set()
    statuses: dict[str, int] = {}
    by = {method: {seed: 0 for seed in SEEDS} for method in METHODS}
    rows = 0
    for csv_path in ONLINE_ROOT.glob("lane_*/online/online_episode_results.csv"):
        with csv_path.open(newline="") as file:
            for row in csv.DictReader(file):
                rows += 1
                st = row.get("status", "")
                statuses[st] = statuses.get(st, 0) + 1
                if st != "completed":
                    continue
                if row.get("initial_upright_ok") not in {"1", "1.0", "True", "true"}:
                    continue
                key = (row.get("method", ""), int(row.get("seed", "-1")), int(row.get("episode_index", "-1")))
                if key in completed:
                    continue
                completed.add(key)
                method, seed, _ = key
                if method in by and seed in by[method]:
                    by[method][seed] += 1
    audit_path = ONLINE_ROOT / "merged/online_acceptance_audit.json"
    audit = load_json(audit_path)
    audit_ok = audit_path.exists() and audit.get("status") == "PASS_STRICT_ONLINE" and audit.get("go") is True and not audit.get("blocking_reasons")
    expected_shape_ok = len(completed) == 2500 and all(by[m][s] == 100 for m in METHODS for s in SEEDS)
    return {
        "overall": "PASS" if audit_ok and expected_shape_ok else "BLOCKED",
        "checks": {
            "online_expected_episodes": status(expected_shape_ok, {"completed_unique": len(completed), "expected": 2500, "by_method_seed": by, "rows": rows, "statuses": statuses}),
            "online_acceptance_audit": status(audit_ok, {"path": str(audit_path), "summary": {k: audit.get(k) for k in ["status", "go", "blocking_reasons"]}}),
        },
    }


def main() -> None:
    phases = {
        "phase0": phase0(),
        "phase1": phase1(),
        "phase2": phase2(),
        "phase3": phase3(),
        "phase4": phase4(),
        "phase5_offline": phase5_offline(),
        "phase5_online": online_progress(),
    }
    blockers = [
        f"{phase}:{name}"
        for phase, pdata in phases.items()
        for name, check in pdata.get("checks", {}).items()
        if check.get("status") == "BLOCKED"
    ]
    result = {
        "objective": "Plan A complete except Phase2 manual human review",
        "overall": "PASS" if not blockers else "BLOCKED",
        "blocking_reasons": blockers,
        "phases": phases,
    }
    out = OUT / "plan_a_strict_completion_audit_20260602.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": result["overall"], "blocking_reasons": blockers, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()

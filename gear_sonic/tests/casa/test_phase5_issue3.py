from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gear_sonic.casa.phase5 import (
    MAIN_SKILLS,
    Phase5ValidationError,
    acceptance_search_thresholds,
    metrics_from_reject_flags,
    read_prediction_rows,
    relative_reduction,
    split_indices_for_phase5,
    validate_phase5_prediction_rows,
)
from gear_sonic.scripts.casa_audit_phase5_acceptance import _checks

FIELDS = [
    "sample_id",
    "phase4_split",
    "skill_name",
    "label",
    "raw_critic_risk",
    "hard_contract_score",
    "hard_contract_fixed_reject",
]


def test_split_indices_carves_critic_val_without_touching_calibration() -> None:
    splits = np.asarray(["train"] * 10 + ["calibration"] * 3 + ["test"] * 2)

    indices = split_indices_for_phase5(splits, seed=7)

    assert set(indices) == {"train", "critic_val", "calibration", "test"}
    assert set(indices["calibration"]) == {10, 11, 12}
    assert set(indices["critic_val"]).isdisjoint(indices["calibration"])
    assert set(indices["train"]).isdisjoint(indices["critic_val"])


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"label": "2"}, "label must be exactly 0 or 1"),
        ({"raw_critic_risk": "nan"}, "raw_critic_risk must be finite"),
        ({"hard_contract_score": ""}, "missing required field"),
        ({"skill_name": "jump"}, "invalid skill_name"),
        ({"hard_contract_fixed_reject": "maybe"}, "hard_contract_fixed_reject must be boolean"),
    ],
)
def test_prediction_csv_validation_rejects_malformed_values(
    tmp_path: Path,
    patch: dict[str, str],
    message: str,
) -> None:
    row = _row("walk", "calibration", 1, 0.9)
    row.update(patch)
    csv_path = tmp_path / "bad_predictions.csv"
    _write_csv(csv_path, [row])

    with pytest.raises(Phase5ValidationError, match=message):
        read_prediction_rows(csv_path)


def test_prediction_csv_validation_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field for field in FIELDS if field != "raw_critic_risk"])
        writer.writeheader()
        writer.writerow({field: _row("walk", "calibration", 1, 0.9).get(field, "") for field in writer.fieldnames})

    with pytest.raises(Phase5ValidationError, match="missing required columns: raw_critic_risk"):
        read_prediction_rows(csv_path)


def test_prediction_row_audit_requires_main_skills_and_unsafe_calibration(tmp_path: Path) -> None:
    rows = [
        _row("walk", "calibration", 1, 0.9),
        _row("walk", "test", 0, 0.1),
    ]
    csv_path = tmp_path / "incomplete.csv"
    _write_csv(csv_path, rows)
    parsed = read_prediction_rows(csv_path)

    with pytest.raises(Phase5ValidationError, match="missing calibration rows for skill 'turn'"):
        validate_phase5_prediction_rows(parsed, min_unsafe_calibration_per_skill=2)


def test_relative_reduction_zero_baseline_is_undefined() -> None:
    assert relative_reduction(0, 3) is None
    assert relative_reduction(10, 4) == pytest.approx(0.6)


def test_safe_acceptance_rate_replaces_task_success_proxy() -> None:
    rows = [
        {"label_int": 0},
        {"label_int": 0},
        {"label_int": 1},
    ]
    metrics = metrics_from_reject_flags(rows, [False, True, False], method="demo", split="test")

    assert metrics["safe_acceptance_rate"] == pytest.approx(0.5)
    assert metrics["safe_acceptance_proxy"] == pytest.approx(0.5)
    assert metrics["task_success_proxy"] == pytest.approx(0.5)


def test_acceptance_search_respects_fnr_and_safe_rejection_budget(tmp_path: Path) -> None:
    rows = []
    for skill in MAIN_SKILLS:
        rows.extend(
            [
                _row(skill, "calibration", 1, 0.95),
                _row(skill, "calibration", 1, 0.85),
                _row(skill, "calibration", 0, 0.20),
                _row(skill, "calibration", 0, 0.10),
                _row(skill, "test", 1, 0.95),
            ]
        )
    csv_path = tmp_path / "predictions.csv"
    _write_csv(csv_path, rows)
    parsed = read_prediction_rows(csv_path)

    thresholds, diagnostics = acceptance_search_thresholds(
        parsed,
        alpha=0.50,
        fnr_margin=0.0,
        max_safe_reject_rate=0.50,
    )

    assert set(thresholds["per_skill"]) == set(MAIN_SKILLS)
    for skill in MAIN_SKILLS:
        selected = diagnostics["per_skill"][skill]["selected"]
        assert selected["fnr"] <= 0.50
        assert selected["safe_reject_rate"] <= 0.50
        assert diagnostics["per_skill"][skill]["constraints_met"]


def test_audit_does_not_block_on_undefined_fixed_hard_or_global_closeness() -> None:
    calibration = {
        "per_skill_calibration": {
            skill: {"unsafe_total": 200, "fnr": 0.02} for skill in MAIN_SKILLS
        }
    }
    baseline = {
        "casa_vs_sonic_unsafe_reduction": 0.80,
        "casa_vs_hard_unsafe_reduction": None,
        "fixed_hard_comparison_mode": "diagnostic",
        "casa_vs_matched_hard_unsafe_reduction": None,
        "casa_safe_acceptance_drop_abs_vs_sonic": 0.0,
        "casa_safe_acceptance_drop_rel_vs_sonic": 0.0,
        "casa_vs_global_fnr_closer_skill_count": 0,
        "per_skill": {
            "casa_a_per_skill": {skill: {"fnr": 0.01} for skill in MAIN_SKILLS},
        },
    }

    checks = _checks(calibration, baseline, alpha=0.10)

    assert checks["casa_vs_fixed_hard_unsafe_reduction_ge_20pct_or_diagnostic"]
    assert "casa_vs_global_fnr_closer_at_least_2_skills" not in checks


def test_synthetic_phase5_acceptance_pipeline_passes(tmp_path: Path) -> None:
    phase4_root = tmp_path / "phase4"
    phase5_root = tmp_path / "phase5"
    predictions_csv = phase4_root / "raw_critic" / "predictions.csv"
    rows = []
    for skill in MAIN_SKILLS:
        for index in range(200):
            rows.append(_row(skill, "calibration", 1, 0.95, sample_suffix=f"cal-u-{index}"))
        for index in range(20):
            rows.append(_row(skill, "calibration", 0, 0.05, sample_suffix=f"cal-s-{index}"))
        for index in range(5):
            rows.append(_row(skill, "test", 1, 0.95, sample_suffix=f"test-u-{index}"))
            rows.append(_row(skill, "test", 0, 0.05, sample_suffix=f"test-s-{index}"))
    _write_csv(predictions_csv, rows)

    _run(
        "gear_sonic/scripts/casa_calibrate_phase5_conformal.py",
        "--phase4-root",
        str(phase4_root),
        "--output-dir",
        str(phase5_root),
        "--selection-mode",
        "acceptance_search",
        "--alpha",
        "0.10",
        "--fnr-margin",
        "0.02",
        "--max-safe-reject-rate",
        "0.10",
    )
    _run(
        "gear_sonic/scripts/casa_eval_phase5_baselines.py",
        "--phase4-root",
        str(phase4_root),
        "--phase5-root",
        str(phase5_root),
    )
    _run(
        "gear_sonic/scripts/casa_audit_phase5_acceptance.py",
        "--phase5-root",
        str(phase5_root),
        "--strict",
    )

    audit = json.loads((phase5_root / "phase5_go_no_go.json").read_text())
    assert audit["go"] is True
    assert audit["status"] == "PASS_STRICT"


def _row(
    skill: str,
    split: str,
    label: int,
    risk: float,
    *,
    sample_suffix: str = "0",
) -> dict[str, object]:
    return {
        "sample_id": f"{split}-{skill}-{label}-{sample_suffix}",
        "phase4_split": split,
        "skill_name": skill,
        "label": label,
        "raw_critic_risk": risk,
        "hard_contract_score": 0.0,
        "hard_contract_fixed_reject": 0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=Path(__file__).resolve().parents[3])

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from gear_sonic.casa.baselines.sota_adapters import SOTA_METHOD_ORDER, SotaDecisionContext, build_sota_registry
from gear_sonic.casa.phase5_policy import ONLINE_METHOD_ORDER, method_display


def test_sota_registry_decision_shape() -> None:
    registry = build_sota_registry()
    calibration_rows = [
        _prediction_row("cal_safe_walk", "calibration", "walk", 0, 0.05, 0.02, 0),
        _prediction_row("cal_unsafe_walk", "calibration", "walk", 1, 0.92, 0.15, 0),
        _prediction_row("cal_safe_turn", "calibration", "turn", 0, 0.10, 0.05, 0),
        _prediction_row("cal_unsafe_turn", "calibration", "turn", 1, 0.88, 0.25, 0),
    ]
    registry.calibrate(calibration_rows)

    context = SotaDecisionContext(
        skill_name="walk",
        skill_params={"vx": 0.8, "duration": 2.0},
        raw_critic_risk=0.95,
        hard_contract_score=0.20,
        hard_contract_fixed_reject=False,
    )

    for method in SOTA_METHOD_ORDER:
        decision = registry.decide(method, context)
        assert decision.method_name == method
        assert decision.source_method
        assert decision.implementation_fidelity in {"paper_faithful_proxy", "lightweight_proxy"}
        assert decision.mode == "gate"
        assert isinstance(decision.allow, bool)
        assert decision.runtime_ms is not None
        assert decision.solver_status


def test_sota_methods_are_online_registered() -> None:
    for method in SOTA_METHOD_ORDER:
        assert method in ONLINE_METHOD_ORDER
        assert method_display(method) != method


def test_sota_offline_replay_cli_writes_metrics(tmp_path: Path) -> None:
    phase4_root = tmp_path / "phase4"
    phase5_root = tmp_path / "phase5"
    output_dir = tmp_path / "offline"
    predictions_csv = phase4_root / "raw_critic" / "predictions.csv"
    thresholds_json = phase5_root / "conformal_thresholds.json"
    _write_predictions(predictions_csv)
    thresholds_json.parent.mkdir(parents=True, exist_ok=True)
    thresholds_json.write_text(
        json.dumps(
            {
                "thresholds": {
                    "global": 0.9,
                    "per_skill": {"walk": 0.9, "turn": 0.9, "gesture": 0.9, "passive": 0.9},
                }
            }
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/casa_eval_sota_adapted_baselines.py",
            "--phase4-root",
            str(phase4_root),
            "--phase5-root",
            str(phase5_root),
            "--output-dir",
            str(output_dir),
            "--split",
            "test",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    metrics = json.loads((output_dir / "offline_replay_metrics.json").read_text())
    assert {row["method"] for row in metrics["methods"]} == set(SOTA_METHOD_ORDER)
    assert (output_dir / "offline_replay_results.csv").exists()
    assert (output_dir / "sota_split_manifest.json").exists()
    assert (output_dir / "sota_feature_schema.json").exists()
    report = (output_dir / "offline_replay_report.md").read_text()
    assert "No external paper result numbers" in report


def _write_predictions(path: Path) -> None:
    rows = [
        _prediction_row("train_safe_walk", "train", "walk", 0, 0.05, 0.02, 0),
        _prediction_row("val_safe_turn", "critic_val", "turn", 0, 0.07, 0.03, 0),
        _prediction_row("cal_safe_walk", "calibration", "walk", 0, 0.04, 0.02, 0),
        _prediction_row("cal_unsafe_walk", "calibration", "walk", 1, 0.95, 0.25, 0),
        _prediction_row("cal_safe_turn", "calibration", "turn", 0, 0.10, 0.03, 0),
        _prediction_row("cal_unsafe_turn", "calibration", "turn", 1, 0.90, 0.20, 0),
        _prediction_row("cal_safe_gesture", "calibration", "gesture", 0, 0.08, 0.02, 0),
        _prediction_row("cal_unsafe_gesture", "calibration", "gesture", 1, 0.89, 0.35, 0),
        _prediction_row("cal_safe_passive", "calibration", "passive", 0, 0.02, 0.01, 0),
        _prediction_row("cal_unsafe_passive", "calibration", "passive", 1, 0.82, 0.40, 0),
        _prediction_row("test_safe_walk", "test", "walk", 0, 0.03, 0.01, 0),
        _prediction_row("test_unsafe_walk", "test", "walk", 1, 0.93, 0.20, 0),
        _prediction_row("test_safe_turn", "test", "turn", 0, 0.08, 0.02, 0),
        _prediction_row("test_unsafe_turn", "test", "turn", 1, 0.91, 0.30, 0),
        _prediction_row("test_hard_passive", "test", "passive", 1, 0.20, 0.95, 1),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(
    sample_id: str,
    split: str,
    skill: str,
    label: int,
    raw_risk: float,
    hard_score: float,
    hard_fixed: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "phase4_split": split,
        "skill_name": skill,
        "label": label,
        "safe_label": "unsafe" if label else "safe",
        "raw_critic_risk": raw_risk,
        "hard_contract_score": hard_score,
        "hard_contract_fixed_reject": hard_fixed,
        "triggered_violation_types": "[]",
        "time_to_violation": "",
        "summary_path": "",
    }

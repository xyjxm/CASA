"""Calibrate CASA-Hybrid helpers on the Phase 4 calibration split only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.baselines.sota_adapters import SotaDecisionContext, build_sota_registry  # noqa: E402
from gear_sonic.casa.phase5 import (  # noqa: E402
    metrics_from_reject_flags,
    read_json,
    read_prediction_rows,
    split_role,
)
from gear_sonic.casa.phase5_hybrid import (  # noqa: E402
    HYBRID_METHOD_ORDER,
    Phase5HybridConfig,
    decide_phase5_hybrid,
    load_phase5_hybrid_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--hybrid-config", type=Path, default=Path("configs/phase5_hybrid_mpc_safedpa.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", default=",".join(HYBRID_METHOD_ORDER))
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.phase5_root / "hybrid_mpc_safedpa_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(args.phase4_root / "raw_critic" / "predictions.csv")
    calibration_rows = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    thresholds = read_json(args.phase5_root / "conformal_thresholds.json")["thresholds"]
    config = load_phase5_hybrid_config(args.hybrid_config)
    registry, registry_summary = _calibrated_registry(rows)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    replay_rows: list[dict[str, Any]] = []
    method_metrics: list[dict[str, Any]] = []
    ensemble_scores: list[tuple[float, int]] = []

    for method in methods:
        decisions = _replay_hybrid_rows(calibration_rows, thresholds, registry, config, method)
        replay_rows.extend(decisions)
        reject_flags = [row["decision"] == "reject" for row in decisions]
        metric = metrics_from_reject_flags(
            calibration_rows,
            reject_flags,
            method=method,
            split="calibration",
        )
        metric["source_split"] = "phase4_calibration"
        method_metrics.append(metric)
        if method == "casa_h_mpc_safedpa_calibrated_ensemble":
            ensemble_scores = [
                (float(row["risk_score"]), int(source["label_int"]))
                for row, source in zip(decisions, calibration_rows)
                if _is_finite(row.get("risk_score"))
            ]

    suggested_config = config.to_dict()
    suggested_config["calibrated_ensemble"]["threshold"] = _threshold_for_fnr_from_scores(
        ensemble_scores,
        alpha=args.alpha,
        default=config.ensemble_threshold,
    )
    manifest = {
        "phase": "CASA-Hybrid Phase5 calibration",
        "source_split": "phase4_calibration",
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "hybrid_config": str(args.hybrid_config),
        "output_dir": str(output_dir),
        "methods": methods,
        "calibration_rows": len(calibration_rows),
        "alpha": args.alpha,
        "registry_summary": registry_summary,
        "note": "This script does not read final held-out online test labels.",
    }
    _write_csv(output_dir / "hybrid_calibration_replay.csv", replay_rows)
    _write_csv(output_dir / "hybrid_calibration_metrics.csv", method_metrics)
    _write_json(output_dir / "hybrid_calibration_metrics.json", {"manifest": manifest, "methods": method_metrics})
    _write_yaml(output_dir / "suggested_phase5_hybrid_config.yaml", suggested_config)
    _write_json(output_dir / "hybrid_calibration_manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "methods": method_metrics}, indent=2, sort_keys=True))


def _calibrated_registry(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    registry = build_sota_registry()
    train_rows = [row for row in rows if split_role(row.get("phase4_split")) == "train"]
    val_rows = [row for row in rows if split_role(row.get("phase4_split")) == "critic_val"]
    calibration_rows = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    registry.fit(train_rows, val_rows)
    registry.calibrate(calibration_rows)
    return registry, {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "calibration_rows": len(calibration_rows),
        "adapters": registry.metadata(),
    }


def _replay_hybrid_rows(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    registry: Any,
    config: Phase5HybridConfig,
    method: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        context = _context_from_prediction(row, thresholds)
        casa = _casa_decision(row, thresholds)
        decision = decide_phase5_hybrid(
            method=method,
            context=context,
            casa_decision=casa,
            sota_registry=registry,
            config=config,
        )
        output.append(
            {
                "sample_id": row.get("sample_id", ""),
                "phase4_split": row.get("phase4_split", ""),
                "skill_name": row.get("skill_name", ""),
                "label": row.get("label", ""),
                "method": method,
                "decision": "allow" if decision.allow else "reject",
                "risk_score": "" if decision.risk_score is None else decision.risk_score,
                "threshold": "" if decision.threshold is None else decision.threshold,
                "risk_margin": "" if decision.risk_margin is None else decision.risk_margin,
                "reject_reason": decision.reject_reason,
                "diagnostics_json": json.dumps(decision.diagnostics, sort_keys=True),
            }
        )
    return output


def _context_from_prediction(row: dict[str, Any], thresholds: dict[str, Any]) -> SotaDecisionContext:
    return SotaDecisionContext(
        skill_name=str(row.get("skill_name", "")),
        skill_params=_params(row),
        raw_critic_risk=_float(row.get("raw_critic_risk_float", row.get("raw_critic_risk"))),
        hard_contract_score=_float(row.get("hard_contract_score_float", row.get("hard_contract_score"))),
        hard_contract_fixed_reject=_bool(
            row.get("hard_contract_fixed_reject_bool", row.get("hard_contract_fixed_reject"))
        ),
        thresholds=thresholds,
        row=row,
    )


def _casa_decision(row: dict[str, Any], thresholds: dict[str, Any]) -> Any:
    skill = str(row.get("skill_name", ""))
    risk = _float(row.get("raw_critic_risk_float", row.get("raw_critic_risk")))
    threshold = float(thresholds["per_skill"][skill])
    reject = risk >= threshold
    return SimpleNamespace(
        reject=reject,
        threshold=threshold,
        risk_margin=risk - threshold,
        reject_reason="per_skill_conformal_threshold" if reject else "allow",
    )


def _threshold_for_fnr_from_scores(
    score_labels: list[tuple[float, int]],
    *,
    alpha: float,
    default: float,
) -> float:
    positive_scores = sorted(score for score, label in score_labels if int(label) == 1 and math.isfinite(score))
    if not positive_scores:
        return default
    allowed_false_negatives = max(0, min(len(positive_scores) - 1, int(math.floor(alpha * len(positive_scores)))))
    best = positive_scores[0]
    for threshold in sorted(set(positive_scores)):
        false_negatives = sum(1 for score in positive_scores if score < threshold)
        if false_negatives <= allowed_false_negatives:
            best = threshold
        else:
            break
    return float(best)


def _params(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("candidate_params_json", "params_json", "skill_params_json"):
        raw = row.get(key)
        if not raw:
            continue
        try:
            loaded = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, data: Any) -> None:
    try:
        import yaml  # noqa: PLC0415

        text = yaml.safe_dump(data, sort_keys=False)
    except ModuleNotFoundError:
        text = json.dumps(data, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y"}


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()

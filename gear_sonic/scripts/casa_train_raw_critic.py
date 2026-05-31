"""Train the CASA Phase 4 Raw Critic and evaluate rejection-risk tradeoffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.dataset import load_dataset_npz
from gear_sonic.casa.phase5 import relative_reduction, split_indices_for_phase5


MAIN_SKILLS = ("walk", "turn", "gesture", "passive")
GROUP_ORDER = ("robot", "env", "runtime", "skill")
REJECT_BUDGETS = (0.0, 0.05, 0.10, 0.20)
PARETO_BUDGETS = tuple(round(value, 2) for value in np.linspace(0.0, 0.50, 21))


class RawRiskCritic(nn.Module):
    """Multi-encoder MLP matching the Phase 4 Raw Critic design."""

    def __init__(self, input_dim: int, groups: dict[str, list[int]], hidden_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.group_names = list(GROUP_ORDER)
        self.group_indices = {
            name: torch.tensor(groups.get(name, []), dtype=torch.long) for name in self.group_names
        }
        encoders: dict[str, nn.Module] = {}
        fusion_dim = 0
        for name in self.group_names:
            group_dim = len(self.group_indices[name])
            if group_dim <= 0:
                continue
            group_hidden = max(32, min(hidden_dim, group_dim * 2))
            group_out = max(16, hidden_dim // 2)
            encoders[name] = nn.Sequential(
                nn.Linear(group_dim, group_hidden),
                nn.LayerNorm(group_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(group_hidden, group_out),
                nn.ReLU(),
            )
            fusion_dim += group_out
        if not encoders:
            encoders["all"] = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            fusion_dim = hidden_dim
        self.encoders = nn.ModuleDict(encoders)
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        for name in self.group_names:
            if name not in self.encoders:
                continue
            indices = self.group_indices[name].to(x.device)
            encoded.append(self.encoders[name](x.index_select(dim=1, index=indices)))
        if not encoded:
            encoded.append(self.encoders["all"](x))
        return self.risk_head(torch.cat(encoded, dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when Raw Critic go criteria fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    data = load_dataset_npz(args.dataset_dir)
    X = _finite_array(data["X"].astype(np.float32))
    y = data["y"].astype(np.float32)
    splits = data["split"].astype(str)
    rows = data["rows"]
    indices = _split_indices(splits, len(y), args.seed)

    mean = X[indices["train"]].mean(axis=0)
    std = X[indices["train"]].std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {name: [int(index) for index in values] for name, values in data["schema"].get("groups", {}).items()}
    model = RawRiskCritic(Xn.shape[1], groups, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    pos_count = float(y[indices["train"]].sum())
    neg_count = float(len(indices["train"]) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(Xn, y, indices["train"], args.batch_size, shuffle=True)

    curves: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, -math.inf)
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        critic_val_probs = _predict(model, Xn, indices["critic_val"], device)
        critic_val_metrics = _binary_metrics(y[indices["critic_val"]], critic_val_probs)
        current_key = (
            1.0 if critic_val_metrics["auroc_available"] else 0.0,
            float(critic_val_metrics["auroc"]) - float(critic_val_metrics["brier"]),
        )
        curves.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "critic_val_auroc": critic_val_metrics["auroc"],
                "critic_val_auprc": critic_val_metrics["auprc"],
                "critic_val_brier": critic_val_metrics["brier"],
                "critic_val_ece": critic_val_metrics["ece"],
            }
        )
        if current_key >= best_key:
            best_key = current_key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    all_probs = _predict(model, Xn, np.arange(len(y)), device)
    hard_scores, hard_fixed_reject = _hard_contract_scores(X, data["feature_names"])
    split_metrics = {}
    train_prior = float(y[indices["train"]].mean()) if len(indices["train"]) else 0.0
    for split_name, split_idx in indices.items():
        split_metrics[split_name] = _binary_metrics(y[split_idx], all_probs[split_idx])
        split_metrics[split_name]["constant_brier"] = _brier(y[split_idx], np.full(len(split_idx), train_prior))
    per_skill = _per_skill_metrics(rows, y, all_probs, indices)
    pareto_rows, pareto_summary = _pareto_analysis(y, all_probs, hard_scores, hard_fixed_reject, indices["test"])
    checks = _go_checks(split_metrics, per_skill, pareto_summary)

    metrics = {
        "phase": "CASA Phase4 Raw Critic",
        "dataset_dir": str(args.dataset_dir),
        "sample_count": int(len(y)),
        "split_counts": {name: int(len(value)) for name, value in indices.items()},
        "split_roles": {
            "model_selection": "critic_val",
            "conformal_calibration": "calibration",
            "notes": [
                "Raw critic best-epoch selection uses critic_val only.",
                "The conformal calibration split is held out for Phase 5 threshold calibration.",
            ],
        },
        "device": str(device),
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "train_positive_rate": train_prior,
        "overall": split_metrics,
        "per_skill": per_skill,
        "pareto_summary": pareto_summary,
        "checks": checks,
        "go_criteria_passed": all(checks.values()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_schema": data["schema"],
            "model_class": "RawRiskCritic",
            "args": vars(args),
            "split_roles": metrics["split_roles"],
        },
        args.output_dir / "raw_critic.pt",
    )
    _write_csv(args.output_dir / "training_curves.csv", curves)
    _write_predictions(args.output_dir / "predictions.csv", rows, splits, y, all_probs, hard_scores, hard_fixed_reject)
    _write_csv(args.output_dir / "pareto_rejection_risk.csv", pareto_rows)
    (args.output_dir / "pareto_summary.json").write_text(json.dumps(pareto_summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.strict and not metrics["go_criteria_passed"]:
        raise SystemExit(2)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _split_indices(splits: np.ndarray, count: int, seed: int) -> dict[str, np.ndarray]:
    del count
    return split_indices_for_phase5(splits, seed)


def _loader(X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X[indices]).float(), torch.from_numpy(y[indices]).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_y)
        total_count += len(batch_y)
    return total_loss / max(total_count, 1)


def _predict(model: nn.Module, X: np.ndarray, indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    if len(indices) == 0:
        return np.asarray([], dtype=np.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), 8192):
            batch = torch.from_numpy(X[indices[start : start + 8192]]).float().to(device)
            outputs.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.float32)
    probs = probs.astype(np.float32)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    auroc = _auroc(labels, probs)
    auprc = _auprc(labels, probs)
    prevalence = float(labels.mean()) if len(labels) else 0.0
    return {
        "count": int(len(labels)),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": prevalence,
        "auroc": auroc if auroc is not None else 0.0,
        "auroc_available": auroc is not None,
        "auprc": auprc if auprc is not None else 0.0,
        "auprc_available": auprc is not None,
        "auprc_lift": (auprc / prevalence) if auprc is not None and prevalence > 0 else 0.0,
        "brier": _brier(labels, probs),
        "ece": _ece(labels, probs),
        "fnr_at_0p5": _fnr_at_threshold(labels, probs, 0.5),
        "fpr_at_0p5": _fpr_at_threshold(labels, probs, 0.5),
    }


def _per_skill_metrics(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    probs: np.ndarray,
    indices: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for split_name, split_idx in indices.items():
        split_set = set(int(index) for index in split_idx)
        split_metrics = {}
        for skill in MAIN_SKILLS:
            skill_idx = np.asarray(
                [index for index, row in enumerate(rows) if index in split_set and row.get("skill_name") == skill],
                dtype=np.int64,
            )
            split_metrics[skill] = _binary_metrics(labels[skill_idx], probs[skill_idx])
        output[split_name] = split_metrics
    return output


def _hard_contract_scores(features: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
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


def _pareto_analysis(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    hard_scores: np.ndarray,
    hard_fixed_reject: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_test = labels[test_idx].astype(np.int64)
    raw_test = raw_scores[test_idx]
    hard_test = hard_scores[test_idx]
    fixed_test = hard_fixed_reject[test_idx]
    rows: list[dict[str, Any]] = []
    total_unsafe = int(y_test.sum())
    total_safe = int(len(y_test) - total_unsafe)
    for budget in PARETO_BUDGETS:
        rows.append(_budget_row("raw_critic", budget, y_test, raw_test))
        rows.append(_budget_row("hard_contract_budgeted", budget, y_test, hard_test))
    rows.append(_fixed_row("hard_contract_fixed", y_test, fixed_test))
    rows.append(_fixed_row("direct_executor", y_test, np.zeros(len(y_test), dtype=bool)))

    raw_by_budget = {
        float(row["reject_budget"]): row
        for row in rows
        if row["method"] == "raw_critic" and float(row["reject_budget"]) in REJECT_BUDGETS
    }
    hard_by_budget = {
        float(row["reject_budget"]): row
        for row in rows
        if row["method"] == "hard_contract_budgeted" and float(row["reject_budget"]) in REJECT_BUDGETS
    }
    reductions = {}
    for budget in (0.10, 0.20):
        raw_unsafe = int(raw_by_budget.get(budget, {}).get("unsafe_invocation_count", 0))
        hard_unsafe = int(hard_by_budget.get(budget, {}).get("unsafe_invocation_count", 0))
        reductions[str(budget)] = relative_reduction(hard_unsafe, raw_unsafe)
    return rows, {
        "test_count": int(len(y_test)),
        "test_unsafe_count": total_unsafe,
        "test_safe_count": total_safe,
        "reject_budgets": list(REJECT_BUDGETS),
        "raw_vs_hard_unsafe_reduction": reductions,
        "raw_reduces_unsafe_ge_15pct_at_10_or_20": any(
            value is not None and value >= 0.15 for value in reductions.values()
        ),
        "hard_contract_fixed": _fixed_row("hard_contract_fixed", y_test, fixed_test),
    }


def _budget_row(method: str, budget: float, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    reject_count = int(math.floor(len(labels) * budget))
    order = np.argsort(-scores, kind="mergesort")
    reject = np.zeros(len(labels), dtype=bool)
    if reject_count > 0:
        reject[order[:reject_count]] = True
    return _row_from_reject(method, budget, labels, reject)


def _fixed_row(method: str, labels: np.ndarray, reject: np.ndarray) -> dict[str, Any]:
    return _row_from_reject(method, float(reject.mean()) if len(reject) else 0.0, labels, reject)


def _row_from_reject(method: str, reject_budget: float, labels: np.ndarray, reject: np.ndarray) -> dict[str, Any]:
    accepted = ~reject
    rejected_count = int(reject.sum())
    accepted_count = int(accepted.sum())
    unsafe_invocations = int(labels[accepted].sum())
    safe_total = int((labels == 0).sum())
    accepted_safe = int(((labels == 0) & accepted).sum())
    rejected_unsafe = int(((labels == 1) & reject).sum())
    safe_acceptance_rate = accepted_safe / safe_total if safe_total else 0.0
    return {
        "method": method,
        "reject_budget": reject_budget,
        "reject_count": rejected_count,
        "reject_rate": rejected_count / len(labels) if len(labels) else 0.0,
        "accepted_count": accepted_count,
        "unsafe_invocation_count": unsafe_invocations,
        "accepted_unsafe_rate": unsafe_invocations / accepted_count if accepted_count else 0.0,
        "prevented_unsafe_count": rejected_unsafe,
        "rejection_precision": rejected_unsafe / rejected_count if rejected_count else 0.0,
        "safe_acceptance_rate": safe_acceptance_rate,
        "safe_acceptance_proxy": safe_acceptance_rate,
        "task_success_proxy": safe_acceptance_rate,
    }


def _go_checks(
    split_metrics: dict[str, dict[str, Any]],
    per_skill: dict[str, dict[str, dict[str, Any]]],
    pareto_summary: dict[str, Any],
) -> dict[str, bool]:
    test = split_metrics["test"]
    prevalence = test.get("positive_rate", 0.0)
    auprc_target = min(2.0 * prevalence, 0.85)
    per_skill_test = per_skill.get("test", {})
    return {
        "overall_auroc_ge_0p75": test.get("auroc_available", False) and test.get("auroc", 0.0) >= 0.75,
        "each_main_skill_auroc_ge_0p65": all(
            per_skill_test.get(skill, {}).get("auroc_available", False)
            and per_skill_test.get(skill, {}).get("auroc", 0.0) >= 0.65
            for skill in MAIN_SKILLS
        ),
        "auprc_target_or_lift_ge_1p5": (
            (test.get("auprc_available", False) and test.get("auprc", 0.0) >= auprc_target)
            or test.get("auprc_lift", 0.0) >= 1.5
        ),
        "brier_better_than_constant": test.get("brier", 1.0) < test.get("constant_brier", 0.0),
        "raw_reduces_unsafe_ge_15pct_at_10_or_20": bool(
            pareto_summary.get("raw_reduces_unsafe_ge_15pct_at_10_or_20")
        ),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float32)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def _brier(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    if len(labels) == 0:
        return 0.0
    labels = labels.astype(np.float32)
    probs = probs.astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        total += (float(mask.mean()) * abs(confidence - accuracy))
    return total


def _fnr_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    positives = labels == 1
    if not np.any(positives):
        return 0.0
    return float(((probs < threshold) & positives).sum() / positives.sum())


def _fpr_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    negatives = labels == 0
    if not np.any(negatives):
        return 0.0
    return float(((probs >= threshold) & negatives).sum() / negatives.sum())


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    splits: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    hard_scores: np.ndarray,
    hard_fixed_reject: np.ndarray,
) -> None:
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "phase4_split": str(splits[index]),
                "skill_name": row.get("skill_name", ""),
                "label": int(labels[index]),
                "safe_label": row.get("safe_label", ""),
                "raw_critic_risk": float(probs[index]),
                "hard_contract_score": float(hard_scores[index]),
                "hard_contract_fixed_reject": int(bool(hard_fixed_reject[index])),
                "triggered_violation_types": row.get("triggered_violation_types", ""),
                "time_to_violation": row.get("time_to_violation", ""),
                "summary_path": row.get("summary_path", ""),
            }
        )
    _write_csv(path, output_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

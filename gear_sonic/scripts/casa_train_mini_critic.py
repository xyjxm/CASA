"""Train a small CASA Phase 3 safety critic on invocation features."""

from __future__ import annotations

import argparse
import csv
import json
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


GROUP_ORDER = ("robot", "env", "runtime", "skill")


class MiniRiskCritic(nn.Module):
    def __init__(self, input_dim: int, groups: dict[str, list[int]], hidden_dim: int = 64) -> None:
        super().__init__()
        self.group_names = list(GROUP_ORDER)
        self.group_indices = {
            name: torch.tensor(groups.get(name, []), dtype=torch.long) for name in self.group_names
        }
        encoders = {}
        out_dim = 0
        for name in self.group_names:
            group_dim = len(self.group_indices[name])
            if group_dim <= 0:
                continue
            group_hidden = max(16, min(hidden_dim, group_dim * 2))
            group_out = max(8, hidden_dim // 2)
            encoders[name] = nn.Sequential(
                nn.Linear(group_dim, group_hidden),
                nn.ReLU(),
                nn.Linear(group_hidden, group_out),
                nn.ReLU(),
            )
            out_dim += group_out
        if not encoders:
            encoders["all"] = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            out_dim = hidden_dim
        self.encoders = nn.ModuleDict(encoders)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        return self.head(torch.cat(encoded, dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Randomly permute labels before training; used as a leakage sanity baseline.",
    )
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
    train_idx = indices["train"]
    val_idx = indices["val"]
    test_idx = indices["test"]
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        y = rng.permutation(y).astype(np.float32)

    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {name: [int(i) for i in values] for name, values in data["schema"].get("groups", {}).items()}
    model = MiniRiskCritic(Xn.shape[1], groups, hidden_dim=args.hidden_dim).to(device)
    pos_count = float(y[train_idx].sum())
    neg_count = float(len(train_idx) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = _loader(Xn, y, train_idx, args.batch_size, shuffle=True)
    curves = []
    best_state: dict[str, Any] | None = None
    best_val_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_probs = _predict(model, Xn, val_idx, device)
        val_metrics = _binary_metrics(y[val_idx], val_probs)
        curves.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auroc": val_metrics["auroc"],
                "val_brier": val_metrics["brier"],
            }
        )
        if val_metrics["auroc"] >= best_val_auc:
            best_val_auc = val_metrics["auroc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_prior = float(y[train_idx].mean()) if len(train_idx) else 0.0
    metrics = {
        "dataset_dir": str(args.dataset_dir),
        "sample_count": int(len(y)),
        "split_counts": {name: int(len(value)) for name, value in indices.items()},
        "train_positive_rate": train_prior,
        "device": str(device),
        "epochs": args.epochs,
        "shuffle_labels": bool(args.shuffle_labels),
        "overall": {},
        "per_skill": {},
        "checks": {},
    }
    for split_name, split_idx in indices.items():
        probs = _predict(model, Xn, split_idx, device)
        metrics["overall"][split_name] = _binary_metrics(y[split_idx], probs)
        metrics["overall"][split_name]["constant_brier"] = _brier(y[split_idx], np.full(len(split_idx), train_prior))
    metrics["per_skill"] = _per_skill_metrics(rows, y, _predict(model, Xn, np.arange(len(y)), device), indices)
    test = metrics["overall"]["test"]
    per_skill_test = metrics["per_skill"].get("test", {})
    passing_skill_aucs = sum(
        1 for item in per_skill_test.values() if item.get("auroc_available") and item.get("auroc", 0.0) >= 0.60
    )
    metrics["checks"] = {
        "overall_auroc_ge_0p65": test.get("auroc_available", False) and test.get("auroc", 0.0) >= 0.65,
        "three_skills_auroc_ge_0p60": passing_skill_aucs >= 3,
        "brier_better_than_constant": test.get("brier", 1.0) < test.get("constant_brier", 0.0),
    }
    metrics["go_criteria_passed"] = all(metrics["checks"].values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_schema": data["schema"],
            "args": vars(args),
        },
        args.output_dir / "mini_critic.pt",
    )
    _write_curves(args.output_dir / "training_curves.csv", curves)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _split_indices(splits: np.ndarray, count: int, seed: int) -> dict[str, np.ndarray]:
    indices = {name: np.where(splits == name)[0] for name in ["train", "val", "test"]}
    rng = np.random.default_rng(seed)
    all_indices = np.arange(count)
    if len(indices["train"]) == 0:
        rng.shuffle(all_indices)
        train_end = max(1, int(0.7 * count))
        val_end = max(train_end + 1, int(0.85 * count)) if count > 2 else train_end
        indices = {
            "train": all_indices[:train_end],
            "val": all_indices[train_end:val_end],
            "test": all_indices[val_end:],
        }
    if len(indices["val"]) == 0:
        indices["val"] = indices["train"]
    if len(indices["test"]) == 0:
        indices["test"] = indices["val"]
    return indices


def _loader(X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X[indices]).float(),
        torch.from_numpy(y[indices]).float(),
    )
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
        for start in range(0, len(indices), 4096):
            chunk = torch.from_numpy(X[indices[start : start + 4096]]).float().to(device)
            outputs.append(torch.sigmoid(model(chunk)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.float32)
    probs = probs.astype(np.float32)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    auroc = _auroc(labels, probs)
    return {
        "count": int(len(labels)),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "auroc": auroc if auroc is not None else 0.0,
        "auroc_available": auroc is not None,
        "brier": _brier(labels, probs),
    }


def _per_skill_metrics(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    probs: np.ndarray,
    indices: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    skill_names = sorted({row.get("skill_name", "") for row in rows})
    for split_name, split_idx in indices.items():
        split_metrics = {}
        split_set = set(int(index) for index in split_idx)
        for skill in skill_names:
            skill_idx = np.asarray(
                [index for index, row in enumerate(rows) if index in split_set and row.get("skill_name") == skill],
                dtype=np.int64,
            )
            split_metrics[skill] = _binary_metrics(labels[skill_idx], probs[skill_idx])
        output[split_name] = split_metrics
    return output


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


def _brier(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def _write_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_loss", "val_auroc", "val_brier"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

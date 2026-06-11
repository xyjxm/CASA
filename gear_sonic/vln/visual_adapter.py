"""Train and run a first-person visual VLN action adapter.

The offline teacher may use A* to label data, but this module deliberately
keeps test-time features limited to the policy-facing observation:
instruction, head-camera frame/history, previous actions, and runtime skill
status.  It does not consume map, pose, goal, path, waypoint, or cell fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from .actions import ActionDecision, VLNAction
from .dataset import read_jsonl
from .no_casa_policy import InferenceInput


ACTION_VALUES = tuple(action.value for action in VLNAction)
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_VALUES)}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}
PRIVILEGED_FEATURE_TOKENS = {
    "map",
    "occupancy",
    "pose",
    "goal",
    "path",
    "waypoint",
    "cell",
    "astar",
    "teacher",
}


@dataclass(frozen=True)
class VisualAdapterConfig:
    image_width: int = 16
    image_height: int = 12
    instruction_hash_dim: int = 64
    max_history_images: int = 2
    epochs: int = 240
    learning_rate: float = 0.35
    l2: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 17
    stop_class_boost: float = 1.8


@dataclass
class VisualActionAdapter:
    action_values: list[str]
    weights: list[list[float]]
    bias: list[float]
    mean: list[float]
    std: list[float]
    config: dict[str, Any]
    train_records: int
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    name = "visual_action_adapter"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "VisualActionAdapter":
        data = json.loads(path.read_text())
        return cls(**data)

    def predict_proba(self, obs: InferenceInput) -> np.ndarray:
        x = build_inference_features(obs, VisualAdapterConfig(**self.config))
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        xn = (x - mean) / np.maximum(std, 1e-6)
        logits = xn @ weights + bias
        label_counts = self.metrics.get("label_counts") if isinstance(self.metrics, dict) else None
        if isinstance(label_counts, dict):
            for action, count in label_counts.items():
                if action in ACTION_TO_INDEX and int(count) <= 0:
                    logits[ACTION_TO_INDEX[action]] = -1e9
        return _softmax(logits)

    def next_action(self, obs: InferenceInput) -> ActionDecision:
        probs = self.predict_proba(obs)
        idx = int(np.argmax(probs))
        action = VLNAction(self.action_values[idx])
        return ActionDecision(
            action=action,
            raw_output={"action": action.value, "probabilities": _probs_to_dict(self.action_values, probs)},
            source=self.name,
            confidence=float(probs[idx]),
            metadata={
                "visual_adapter_used": True,
                "input_contract": (
                    "instruction+head_camera_history+previous_actions+runtime_skill_status"
                ),
                "privileged_policy_usage": False,
                "test_time_astar_used": False,
                "test_time_map_pose_goal_path_used": False,
                "probabilities": _probs_to_dict(self.action_values, probs),
            },
        )


def train_visual_action_adapter(
    dataset_path: Path,
    output_path: Path,
    *,
    config: VisualAdapterConfig | None = None,
) -> VisualActionAdapter:
    config = config or VisualAdapterConfig()
    rows = read_jsonl(dataset_path)
    rows = [row for row in rows if row.get("image_path")]
    if not rows:
        raise ValueError("visual action adapter training requires auto demo records with image_path")

    x = np.stack([build_record_features(row, config) for row in rows]).astype(np.float32)
    y = np.asarray([ACTION_TO_INDEX[row["teacher_action"]] for row in rows], dtype=np.int64)
    train_idx, val_idx = _train_val_split(y, validation_fraction=config.validation_fraction, seed=config.seed)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    x_norm = (x - mean) / np.maximum(std, 1e-6)

    rng = np.random.default_rng(config.seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(ACTION_VALUES))).astype(np.float32)
    bias = np.zeros(len(ACTION_VALUES), dtype=np.float32)
    sample_weights = _sample_weights(y, stop_class_boost=config.stop_class_boost)

    train_x = x_norm[train_idx]
    train_y = y[train_idx]
    train_w = sample_weights[train_idx]
    curves: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        logits = train_x @ weights + bias
        probs = _softmax_batch(logits)
        target = np.zeros_like(probs)
        target[np.arange(len(train_y)), train_y] = 1.0
        weighted_error = (probs - target) * train_w[:, None]
        normalizer = max(float(train_w.sum()), 1e-6)
        grad_w = train_x.T @ weighted_error / normalizer + config.l2 * weights
        grad_b = weighted_error.sum(axis=0) / normalizer
        weights -= config.learning_rate * grad_w.astype(np.float32)
        bias -= config.learning_rate * grad_b.astype(np.float32)
        if epoch in {0, config.epochs - 1} or (epoch + 1) % 40 == 0:
            curves.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": _weighted_cross_entropy(probs, train_y, train_w),
                    "train_accuracy": _accuracy(probs, train_y),
                }
            )

    train_metrics = _classification_metrics(x_norm[train_idx], y[train_idx], weights, bias)
    val_metrics = _classification_metrics(x_norm[val_idx], y[val_idx], weights, bias) if len(val_idx) else {}
    label_counts = {action: int((y == idx).sum()) for action, idx in ACTION_TO_INDEX.items()}
    adapter = VisualActionAdapter(
        action_values=list(ACTION_VALUES),
        weights=weights.tolist(),
        bias=bias.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        config=asdict(config),
        train_records=len(rows),
        metrics={
            "label_counts": label_counts,
            "train": train_metrics,
            "validation": val_metrics,
            "curves": curves,
        },
        metadata={
            "trainer": "numpy_softmax_visual_action_adapter",
            "teacher_scope": "A* used only for offline labels",
            "test_time_inputs": [
                "instruction",
                "head_camera_image",
                "head_camera_history",
                "previous_actions",
                "previous_skill_status",
            ],
            "forbidden_test_time_inputs": sorted(PRIVILEGED_FEATURE_TOKENS),
            "privileged_policy_usage": False,
        },
    )
    adapter.save(output_path)
    return adapter


class VisualActionAdapterPolicy:
    """Policy wrapper used by online held-out episodes."""

    def __init__(self, adapter: VisualActionAdapter) -> None:
        self.adapter = adapter
        self.name = adapter.name

    @classmethod
    def from_path(cls, path: Path) -> "VisualActionAdapterPolicy":
        return cls(VisualActionAdapter.load(path))

    def reset(self, episode_id: str) -> None:
        del episode_id

    def next_action(self, obs: InferenceInput) -> ActionDecision:
        return self.adapter.next_action(obs)


def build_record_features(row: dict[str, Any], config: VisualAdapterConfig) -> np.ndarray:
    obs = InferenceInput(
        episode_id=str(row["episode_id"]),
        step_idx=int(row["step_idx"]),
        instruction=str(row["instruction"]),
        image_path=str(row["image_path"]),
        history_image_paths=[str(path) for path in row.get("history_image_paths", [])],
        previous_actions=[str(action) for action in row.get("previous_actions", [])],
        previous_skill_status=[str(status) for status in row.get("previous_skill_status", [])],
    )
    return build_inference_features(obs, config)


def build_inference_features(obs: InferenceInput, config: VisualAdapterConfig) -> np.ndarray:
    parts = [
        _image_features(obs.image_path, config),
        _history_image_features(obs.history_image_paths, config),
        _instruction_features(obs.instruction, config.instruction_hash_dim),
        _action_history_features(obs.previous_actions),
        _status_history_features(obs.previous_skill_status),
        np.asarray([min(float(obs.step_idx) / 64.0, 2.0)], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def _image_features(image_path: str | Path | None, config: VisualAdapterConfig) -> np.ndarray:
    size = (config.image_width, config.image_height)
    if image_path is None or not Path(image_path).exists():
        flat = np.zeros(config.image_width * config.image_height * 3, dtype=np.float32)
        extras = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([flat, extras])
    image = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    cyan_mask = (green > 0.48) & (blue > 0.55) & (red < 0.38)
    green_panel_mask = (green > 0.42) & (red < 0.34) & (blue < 0.45)
    target_mask = cyan_mask | green_panel_mask
    if target_mask.any():
        ys, xs = np.nonzero(target_mask)
        center_x = float(xs.mean() / max(1, arr.shape[1] - 1))
        center_y = float(ys.mean() / max(1, arr.shape[0] - 1))
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(target_mask.size)
    else:
        center_x = 0.0
        center_y = 0.0
        bbox_area = 0.0
    extras = np.asarray(
        [
            0.0,
            float(arr.mean()),
            float(arr.std()),
            float(red.mean()),
            float(green.mean()),
            float(blue.mean()),
            float(target_mask.mean()) + bbox_area,
            center_x - 0.5 + center_y - 0.5,
        ],
        dtype=np.float32,
    )
    return np.concatenate([flat, extras])


def _history_image_features(history_paths: list[str], config: VisualAdapterConfig) -> np.ndarray:
    selected = history_paths[-config.max_history_images :]
    feature_dim = config.image_width * config.image_height * 3 + 8
    features: list[np.ndarray] = []
    for path in selected:
        features.append(_image_features(path, config))
    while len(features) < config.max_history_images:
        features.insert(0, np.zeros(feature_dim, dtype=np.float32))
    return np.concatenate(features).astype(np.float32)


def _instruction_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        idx = int.from_bytes(digest, "little") % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


def _action_history_features(previous_actions: list[str]) -> np.ndarray:
    counts = np.zeros(len(ACTION_VALUES), dtype=np.float32)
    last = np.zeros(len(ACTION_VALUES), dtype=np.float32)
    for action in previous_actions[-8:]:
        if action in ACTION_TO_INDEX:
            counts[ACTION_TO_INDEX[action]] += 1.0
    if previous_actions and previous_actions[-1] in ACTION_TO_INDEX:
        last[ACTION_TO_INDEX[previous_actions[-1]]] = 1.0
    counts /= max(1.0, float(len(previous_actions[-8:])))
    return np.concatenate([counts, last, np.asarray([min(len(previous_actions) / 64.0, 2.0)], dtype=np.float32)])


def _status_history_features(previous_status: list[str]) -> np.ndarray:
    recent = " ".join(previous_status[-4:]).lower()
    return np.asarray(
        [
            1.0 if "blocked" in recent or "collision" in recent else 0.0,
            1.0 if "failed" in recent else 0.0,
            1.0 if "success" in recent or "ok" in recent else 0.0,
            min(len(previous_status) / 64.0, 2.0),
        ],
        dtype=np.float32,
    )


def _train_val_split(y: np.ndarray, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for class_idx in range(len(ACTION_VALUES)):
        indices = np.where(y == class_idx)[0]
        rng.shuffle(indices)
        val_count = int(round(len(indices) * validation_fraction))
        if len(indices) > 1:
            val_count = min(max(1, val_count), len(indices) - 1)
        else:
            val_count = 0
        val.extend(indices[:val_count].tolist())
        train.extend(indices[val_count:].tolist())
    if not train:
        train = list(range(len(y)))
        val = []
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def _sample_weights(y: np.ndarray, *, stop_class_boost: float) -> np.ndarray:
    counts = np.bincount(y, minlength=len(ACTION_VALUES)).astype(np.float32)
    weights = len(y) / np.maximum(counts, 1.0) / float(len(ACTION_VALUES))
    weights = np.clip(weights, 0.25, 5.0)
    weights[ACTION_TO_INDEX[VLNAction.STOP.value]] *= stop_class_boost
    return weights[y].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(float(exp.sum()), 1e-8)


def _softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


def _weighted_cross_entropy(probs: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> float:
    selected = probs[np.arange(len(y)), y]
    loss = -np.log(np.maximum(selected, 1e-8)) * sample_weights
    return float(loss.sum() / max(float(sample_weights.sum()), 1e-6))


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float((np.argmax(probs, axis=1) == y).mean())


def _classification_metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    if len(y) == 0:
        return {"records": 0}
    probs = _softmax_batch(x @ weights + bias)
    pred = np.argmax(probs, axis=1)
    per_class: dict[str, Any] = {}
    for action, idx in ACTION_TO_INDEX.items():
        tp = int(((pred == idx) & (y == idx)).sum())
        fp = int(((pred == idx) & (y != idx)).sum())
        fn = int(((pred != idx) & (y == idx)).sum())
        per_class[action] = {
            "support": int((y == idx).sum()),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
        }
    stop = per_class[VLNAction.STOP.value]
    return {
        "records": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "per_class": per_class,
        "stop_precision": stop["precision"],
        "stop_recall": stop["recall"],
    }


def _probs_to_dict(action_values: list[str], probs: np.ndarray) -> dict[str, float]:
    return {action: float(probs[idx]) for idx, action in enumerate(action_values)}


def assert_no_privileged_inference_schema() -> None:
    schema = InferenceInput("ep", 0, "go", "frame.png").to_schema_dict()
    overlap = PRIVILEGED_FEATURE_TOKENS.intersection(schema)
    if overlap:
        raise AssertionError(f"privileged keys leaked into inference schema: {sorted(overlap)}")

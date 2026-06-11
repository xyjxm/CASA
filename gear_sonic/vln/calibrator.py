"""Offline lightweight action adapter helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .actions import ActionDecision, VLNAction, parse_vln_action
from .dataset import read_jsonl


@dataclass
class AutoAdapter:
    mapping: dict[str, str] = field(default_factory=dict)
    train_records: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def adapt(self, decision: ActionDecision) -> ActionDecision:
        mapped = self.mapping.get(decision.action.value)
        if not mapped:
            return decision
        return ActionDecision(
            action=VLNAction(mapped),
            raw_output=decision.raw_output,
            source=f"{decision.source}+offline_auto_adapter",
            confidence=decision.confidence,
            magnitude=decision.magnitude,
            metadata={**decision.metadata, "pre_adapter_action": decision.action.value},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(asdict(self), file, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "AutoAdapter":
        with path.open() as file:
            data = json.load(file)
        return cls(**data)


def train_auto_adapter(dataset_path: Path, output_path: Path) -> AutoAdapter:
    rows = read_jsonl(dataset_path)
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        raw_value = row.get("navid_raw_action") or row["teacher_action"]
        raw = parse_vln_action(raw_value, source="offline_dataset")
        votes[raw.action.value][row["teacher_action"]] += 1

    mapping = {
        raw_action: counter.most_common(1)[0][0]
        for raw_action, counter in votes.items()
        if counter
    }
    adapter = AutoAdapter(
        mapping=mapping,
        train_records=len(rows),
        metadata={"trainer": "majority_action_adapter_no_privileged_online_inputs"},
    )
    adapter.save(output_path)
    return adapter

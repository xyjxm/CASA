"""VLN backend interfaces and NaVid/Uni-NaVid wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import select
import subprocess
import time
from typing import Any, Protocol

from .actions import ActionDecision, VLNAction, parse_vln_action


@dataclass(frozen=True)
class VLNObservation:
    episode_id: str
    step_idx: int
    instruction: str
    image_path: str | None = None
    history_image_paths: list[str] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    previous_skill_status: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackendResult:
    decision: ActionDecision
    backend_name: str
    raw_output: Any
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class VLNBackend(Protocol):
    name: str

    def reset(self, episode_id: str) -> None:
        ...

    def next_action(self, observation: VLNObservation) -> BackendResult:
        ...


class MockVLNBackend:
    name = "mock"

    def __init__(self, sequence: list[VLNAction | str] | None = None) -> None:
        self.sequence = sequence or [
            VLNAction.FORWARD,
            VLNAction.TURN_LEFT,
            VLNAction.FORWARD,
            VLNAction.STOP,
        ]
        self.index = 0

    def reset(self, episode_id: str) -> None:
        del episode_id
        self.index = 0

    def next_action(self, observation: VLNObservation) -> BackendResult:
        del observation
        raw = self.sequence[min(self.index, len(self.sequence) - 1)]
        self.index += 1
        decision = parse_vln_action(raw, source=self.name)
        return BackendResult(decision=decision, backend_name=self.name, raw_output=raw)


class NaVidBackend:
    """NaVid/Uni-NaVid adapter with optional strict real-model inference."""

    def __init__(
        self,
        *,
        repo_path: Path,
        result_dir: Path,
        model_path: Path | None = None,
        variant: str = "navid",
        allow_instruction_prior: bool = True,
        strict_model: bool = False,
        python_executable: Path | None = None,
        vision_tower_path: Path | None = None,
        worker_timeout_s: float = 900.0,
        max_new_tokens: int = 64,
        seed: int = 11,
    ) -> None:
        if variant not in {"navid", "uni-navid"}:
            raise ValueError("variant must be navid or uni-navid")
        self.repo_path = Path(repo_path)
        self.result_dir = Path(result_dir)
        self.model_path = Path(model_path) if model_path else None
        self.variant = variant
        self.strict_model = strict_model
        self.allow_instruction_prior = False if strict_model else allow_instruction_prior
        self.python_executable = Path(python_executable) if python_executable else None
        self.vision_tower_path = Path(vision_tower_path) if vision_tower_path else None
        self.worker_timeout_s = worker_timeout_s
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.name = f"{variant.replace('-', '_')}_real_model_inference" if strict_model else variant
        self.episode_step = 0
        self.request_counter = 0
        self.process: subprocess.Popen[str] | None = None
        self.availability = self._probe_availability()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.strict_model:
            self._load_worker()
        with (self.result_dir / "backend_probe.json").open("w") as file:
            json.dump(self.availability, file, indent=2, sort_keys=True)

    def _probe_availability(self) -> dict[str, Any]:
        model_exists = self.model_path is not None and self.model_path.exists()
        source_exists = (self.repo_path / "agent_navid.py").exists()
        package_name = "navid" if self.variant == "navid" else "uninavid"
        package_exists = (self.repo_path / package_name).exists()
        python = self.python_executable or Path(
            os.environ.get("NAVID_REAL_PYTHON", "/mnt/data/students/lph/models/navid/envs/navid-real/bin/python")
        )
        python_exists = python.exists()
        vision_tower = self.vision_tower_path or Path(
            os.environ.get("NAVID_EVA_VIT_G", "/mnt/data/students/lph/models/navid/model_zoo/eva_vit_g.pth")
        )
        vision_tower_exists = vision_tower.exists()
        required_shards = self._required_weight_files()
        missing_shards = [str(path) for path in required_shards if not path.exists()]
        shard_sizes = {
            str(path): path.stat().st_size for path in required_shards if path.exists() and path.is_file()
        }
        unavailable_reasons = []
        if not source_exists:
            unavailable_reasons.append("missing_navid_source")
        if not package_exists:
            unavailable_reasons.append(f"missing_{package_name}_package")
        if not model_exists:
            unavailable_reasons.append("missing_model_path")
        if model_exists and not required_shards:
            unavailable_reasons.append("missing_weight_index")
        if missing_shards:
            unavailable_reasons.append("missing_weight_shards")
        if not vision_tower_exists:
            unavailable_reasons.append("missing_eva_vision_tower")
        if not python_exists:
            unavailable_reasons.append("missing_python_executable")
        return {
            "backend": self.variant,
            "repo_path": str(self.repo_path),
            "source_exists": source_exists,
            "package_exists": package_exists,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_path_exists": model_exists,
            "required_weight_files": [str(path) for path in required_shards],
            "missing_weight_files": missing_shards,
            "weight_file_sizes": shard_sizes,
            "python_executable": str(python),
            "python_executable_exists": python_exists,
            "vision_tower_path": str(vision_tower),
            "vision_tower_exists": vision_tower_exists,
            "model_loaded": False,
            "model_unavailable_reason": ",".join(unavailable_reasons) if unavailable_reasons else None,
        }

    def _required_weight_files(self) -> list[Path]:
        if self.model_path is None or not self.model_path.exists():
            return []
        index_path = self.model_path / "pytorch_model.bin.index.json"
        if not index_path.exists():
            return []
        try:
            data = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            return []
        return sorted({self.model_path / name for name in data.get("weight_map", {}).values()})

    def _preload_requirements_met(self) -> bool:
        return bool(
            self.availability["source_exists"]
            and self.availability["package_exists"]
            and self.availability["model_path_exists"]
            and self.availability["required_weight_files"]
            and not self.availability["missing_weight_files"]
            and self.availability["vision_tower_exists"]
            and self.availability["python_executable_exists"]
        )

    def _read_worker_line(self, timeout_s: float) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("NaVid worker process is not running")
        noise_path = self.result_dir / "navid_model_worker.stdout_noise.log"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"NaVid worker exited with code {self.process.returncode}")
            ready, _, _ = select.select([self.process.stdout], [], [], 1.0)
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                with noise_path.open("a") as file:
                    file.write(line)
                continue
        raise TimeoutError(f"timed out waiting {timeout_s:.1f}s for NaVid worker")

    def _load_worker(self) -> None:
        if not self._preload_requirements_met():
            return
        python = Path(self.availability["python_executable"])
        vision_tower = Path(self.availability["vision_tower_path"])
        worker_script = Path(__file__).with_name("navid_model_worker.py")
        stderr_path = self.result_dir / "navid_model_worker.stderr.log"
        stderr_file = stderr_path.open("w")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_path) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        cmd = [
            str(python),
            "-u",
            str(worker_script),
            "--repo-path",
            str(self.repo_path),
            "--model-path",
            str(self.model_path),
            "--variant",
            self.variant,
            "--vision-tower-path",
            str(vision_tower),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
        )
        try:
            response = self._read_worker_line(self.worker_timeout_s)
        except Exception as exc:  # noqa: BLE001
            self.availability["model_unavailable_reason"] = f"worker_start_failed:{exc!r}"
            self.availability["model_loaded"] = False
            return
        self.availability["worker_ready_response"] = response
        if response.get("ok") and response.get("model_loaded"):
            self.availability["model_loaded"] = True
            self.availability["model_unavailable_reason"] = None
        else:
            self.availability["model_loaded"] = False
            self.availability["model_unavailable_reason"] = response.get("error", "worker_model_load_failed")

    def reset(self, episode_id: str) -> None:
        self.episode_step = 0
        self.current_episode_id = episode_id
        if self.process is not None and self.availability.get("model_loaded"):
            self._worker_request({"cmd": "reset"}, timeout_s=30.0)

    @property
    def is_model_available(self) -> bool:
        base_available = bool(
            self.availability["source_exists"]
            and self.availability["package_exists"]
            and self.availability["model_path_exists"]
            and self.availability["required_weight_files"]
            and not self.availability["missing_weight_files"]
            and self.availability["vision_tower_exists"]
        )
        if self.strict_model:
            return base_available and bool(self.availability.get("model_loaded"))
        return base_available

    def _worker_request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("NaVid worker process is not running")
        self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        self.process.stdin.flush()
        response = self._read_worker_line(timeout_s or self.worker_timeout_s)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "NaVid worker request failed"))
        return response

    def _instruction_prior(self, observation: VLNObservation) -> str:
        text = observation.instruction.lower()
        if "stop" in text and self.episode_step >= 2:
            return "stop"
        if "left" in text and self.episode_step == 0:
            return "left"
        if "right" in text and self.episode_step == 0:
            return "right"
        if self.episode_step >= 4 and "stop" in text:
            return "stop"
        return "forward"

    def next_action(self, observation: VLNObservation) -> BackendResult:
        self.episode_step += 1
        self.request_counter += 1
        request_id = (
            f"{self.variant}:{observation.episode_id}:"
            f"{observation.step_idx:04d}:{self.request_counter:08d}:{time.time_ns()}"
        )
        request_started = time.perf_counter()

        def _metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
            elapsed_ms = (time.perf_counter() - request_started) * 1000.0
            return {
                **self.availability,
                "request_id": request_id,
                "navid_response_id": request_id,
                "navid_latency_ms": elapsed_ms,
                "navid_model_loaded": bool(self.availability.get("model_loaded")),
                **(extra or {}),
            }

        if not self.is_model_available:
            if not self.allow_instruction_prior:
                decision = ActionDecision(
                    action=VLNAction.BACKOFF,
                    raw_output="navid_unavailable",
                    source=self.name,
                    confidence=0.0,
                    metadata=_metadata({"model_error": "navid_unavailable"}),
                )
                return BackendResult(
                    decision=decision,
                    backend_name=self.name,
                    raw_output="navid_unavailable",
                    available=False,
                    metadata=decision.metadata,
                )
            raw = self._instruction_prior(observation)
            decision = parse_vln_action(raw, source=f"{self.variant}_instruction_prior")
            return BackendResult(
                decision=decision,
                backend_name=f"{self.variant}_instruction_prior",
                raw_output=raw,
                available=False,
                metadata=_metadata({"fallback": "instruction_prior"}),
            )

        # Real model loading is intentionally isolated from import-time code.
        if self.availability.get("model_loaded"):
            if observation.image_path is None:
                if self.strict_model:
                    raise RuntimeError("strict real NaVid inference requires an observation image_path")
                raw = "navid_missing_image"
                decision = parse_vln_action(raw, source=self.name)
                return BackendResult(
                    decision=decision,
                    backend_name=self.name,
                    raw_output=raw,
                    available=False,
                    metadata=_metadata({"model_error": "missing_image_path"}),
                )
            response = self._worker_request(
                {
                    "cmd": "predict",
                    "request_id": request_id,
                    "episode_id": observation.episode_id,
                    "step_idx": observation.step_idx,
                    "instruction": observation.instruction,
                    "image_paths": [observation.image_path],
                }
            )
            raw = response["raw_output"]
            decision = parse_vln_action(raw, source=self.name)
            metadata = _metadata(
                {
                    "worker_response_type": response.get("type"),
                    "worker_request_id": response.get("request_id", request_id),
                }
            )
            decision = ActionDecision(
                action=decision.action,
                raw_output=decision.raw_output,
                source=decision.source,
                confidence=decision.confidence,
                magnitude=decision.magnitude,
                metadata={**decision.metadata, **metadata},
            )
            return BackendResult(
                decision=decision,
                backend_name=self.name,
                raw_output=raw,
                available=True,
                metadata=metadata,
            )

        raw = "navid_real_model_unavailable"
        decision = parse_vln_action(raw, source=self.name)
        return BackendResult(
            decision=decision,
            backend_name=self.name,
            raw_output=raw,
            available=False,
            metadata=_metadata({"fallback": None}),
        )

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.process = None

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.scene import ScenePropsManager, load_scene_props_config  # noqa: E402


DEFAULT_ROLLOUTS_CSV = (
    REPO_ROOT
    / "outputs/casa/phase2_oracle/phase2_oracle_acceptance_plus_targeted_20260516_165220/review/rollouts.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/casa/phase2_full_video_review_20260519"
DEFAULT_MODEL_XML = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_casa_v1.xml"
DEFAULT_PROPS_CONFIG = REPO_ROOT / "gear_sonic/casa/scene/props.yaml"


@dataclass
class RenderResult:
    run_id: str
    rollout_id: str
    video_id: str
    status: str
    message: str
    source_summary_path: str
    source_sim_state_csv: str
    video_path: str
    contact_sheet_path: str
    view_preset: str
    view_names: str
    oracle_violation_yes_no: str
    oracle_violation_type: str
    oracle_violation_count: str
    oracle_violation_types: str
    scenario: str
    target_bucket: str
    scene_complexity: str
    scene_family: str
    num_users: str
    num_obstacles: str
    duration_s: float
    rendered_frames: int
    elapsed_s: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CASA Phase 2 rollout logs into review videos without rerunning policy."
    )
    parser.add_argument("--rollouts-csv", type=Path, default=DEFAULT_ROLLOUTS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--props-config", type=Path, default=DEFAULT_PROPS_CONFIG)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N rollouts; 0 means all.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--view-preset",
        choices=("single", "cardinal4"),
        default="single",
        help="single keeps the original camera; cardinal4 writes a 2x2 front/right/back/left composite.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    output_root = args.output_root.resolve()
    video_dir = output_root / "videos"
    frame_dir = output_root / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.rollouts_csv.open(newline="")))
    if args.start_index:
        rows = rows[args.start_index :]
    if args.limit:
        rows = rows[: args.limit]

    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    data = mujoco.MjData(model)
    props_config = load_scene_props_config(args.props_config.resolve())
    props_manager = ScenePropsManager(model, data, props_config)
    body_qpos_addr, body_qvel_addr = _body_joint_addresses(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    manifest_rows: list[RenderResult] = []
    started_all = time.time()
    try:
        for index, row in enumerate(rows, start=args.start_index):
            result = render_one(
                row=row,
                row_index=index,
                model=model,
                data=data,
                renderer=renderer,
                props_manager=props_manager,
                body_qpos_addr=body_qpos_addr,
                body_qvel_addr=body_qvel_addr,
                output_root=output_root,
                fps=args.fps,
                size=(args.width, args.height),
                force=args.force,
                write_contact_sheet=not args.no_contact_sheets,
                overlay=not args.no_overlay,
                view_preset=args.view_preset,
            )
            manifest_rows.append(result)
            print(
                f"[{len(manifest_rows)}/{len(rows)}] {result.status}: "
                f"{result.video_id} frames={result.rendered_frames} "
                f"duration={result.duration_s:.2f}s elapsed={result.elapsed_s:.2f}s",
                flush=True,
            )
    finally:
        try:
            renderer.close()
        except Exception:
            pass

    manifest_path = output_root / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    write_readme(output_root, args, manifest_rows, time.time() - started_all)
    print(f"saved manifest: {manifest_path}")
    print(f"output root: {output_root}")


def render_one(
    *,
    row: dict[str, str],
    row_index: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    props_manager: ScenePropsManager,
    body_qpos_addr: np.ndarray,
    body_qvel_addr: np.ndarray,
    output_root: Path,
    fps: float,
    size: tuple[int, int],
    force: bool,
    write_contact_sheet: bool,
    overlay: bool,
    view_preset: str,
) -> RenderResult:
    started = time.time()
    run_id = row.get("run_id") or row.get("source_run_id", "")
    rollout_id = row.get("rollout_id") or row.get("source_rollout_id", "")
    if not run_id or not rollout_id:
        raise KeyError("rollout row must contain run_id/rollout_id or source_run_id/source_rollout_id")
    video_id = safe_id(f"{run_id}__{rollout_id}")
    artifact_suffix = "_4view" if view_preset == "cardinal4" else ""
    video_path = output_root / "videos" / f"{video_id}{artifact_suffix}.mp4"
    contact_sheet_path = output_root / "frames" / f"{video_id}{artifact_suffix}_contact.jpg"

    source_summary = _summary_path_for_row(row, run_id, rollout_id)
    if not source_summary.exists():
        return _result(
            row,
            video_id,
            "error",
            f"missing source summary: {source_summary}",
            source_summary,
            Path(),
            video_path,
            contact_sheet_path,
            {},
            0.0,
            0,
            started,
            view_preset,
            "",
        )

    summary = json.loads(source_summary.read_text())
    sim_state_csv = _resolve_repo_path(summary.get("sim_state_csv", ""))
    if not sim_state_csv.exists():
        return _result(
            row,
            video_id,
            "error",
            f"missing sim_state_csv: {sim_state_csv}",
            source_summary,
            sim_state_csv,
            video_path,
            contact_sheet_path,
            summary,
            0.0,
            0,
            started,
            view_preset,
            "",
        )

    sim_rows = _read_sim_rows(sim_state_csv)
    if not sim_rows:
        return _result(
            row,
            video_id,
            "error",
            "empty sim_state_csv",
            source_summary,
            sim_state_csv,
            video_path,
            contact_sheet_path,
            summary,
            0.0,
            0,
            started,
            view_preset,
            "",
        )

    selected = _sample_rows_by_fps(sim_rows, fps)
    duration_s = float(sim_rows[-1]["sim_time"]) - float(sim_rows[0]["sim_time"]) if len(sim_rows) > 1 else 0.0
    cameras = _cameras_for_rollout(selected, summary.get("scene_props", {}), view_preset)
    view_names = ",".join(name for name, _ in cameras)
    output_size = _output_size(size, view_preset)

    if video_path.exists() and (contact_sheet_path.exists() or not write_contact_sheet) and not force:
        return _result(
            row,
            video_id,
            "skipped",
            "outputs already exist",
            source_summary,
            sim_state_csv,
            video_path,
            contact_sheet_path,
            summary,
            duration_s,
            len(selected),
            started,
            view_preset,
            view_names,
        )

    mujoco.mj_resetData(model, data)
    _apply_scene_props(props_manager, summary.get("scene_props", {}))

    width, height = output_size
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")

    contact_indices = set(_even_indices(len(selected), 5))
    contact_frames: list[tuple[float, np.ndarray]] = []
    first_time = float(sim_rows[0]["sim_time"])

    try:
        for frame_index, sim_row in enumerate(selected):
            _apply_sim_row(model, data, sim_row, body_qpos_addr, body_qvel_addr)
            bgr = _render_frame(
                renderer=renderer,
                data=data,
                cameras=cameras,
                view_preset=view_preset,
            )
            rel_t = max(0.0, float(sim_row["sim_time"]) - first_time)
            if overlay:
                _draw_overlay(bgr, row_index=row_index, rollout_id=rollout_id, rel_t=rel_t)
                if view_preset == "cardinal4":
                    _draw_view_labels(bgr, list(name for name, _ in cameras), size)
            writer.write(bgr)
            if frame_index in contact_indices:
                contact_frames.append((rel_t, bgr.copy()))
    finally:
        writer.release()

    if write_contact_sheet and contact_frames:
        _write_contact_sheet(contact_sheet_path, video_id, contact_frames)

    return _result(
        row,
        video_id,
        "rendered",
        "ok",
        source_summary,
        sim_state_csv,
        video_path,
        contact_sheet_path,
        summary,
        duration_s,
        len(selected),
        started,
        view_preset,
        view_names,
    )


def _result(
    row: dict[str, str],
    video_id: str,
    status: str,
    message: str,
    source_summary: Path,
    sim_state_csv: Path,
    video_path: Path,
    contact_sheet_path: Path,
    summary: dict[str, Any],
    duration_s: float,
    rendered_frames: int,
    started: float,
    view_preset: str,
    view_names: str,
) -> RenderResult:
    return RenderResult(
        run_id=row.get("run_id") or row.get("source_run_id", ""),
        rollout_id=row.get("rollout_id") or row.get("source_rollout_id", ""),
        video_id=video_id,
        status=status,
        message=message,
        source_summary_path=str(source_summary),
        source_sim_state_csv=str(sim_state_csv) if str(sim_state_csv) != "." else "",
        video_path=str(video_path),
        contact_sheet_path=str(contact_sheet_path),
        view_preset=view_preset,
        view_names=view_names,
        oracle_violation_yes_no=_oracle_yes_no(row),
        oracle_violation_type=row.get("violation_type") or row.get("oracle_violation_type", ""),
        oracle_violation_count=row.get("violation_count", ""),
        oracle_violation_types="|".join(summary.get("violation_types", []) or [])
        or row.get("oracle_violation_types", ""),
        scenario=str(summary.get("scenario", "")),
        target_bucket=str(_scene_prop(summary, "target_bucket")),
        scene_complexity=str(_scene_prop(summary, "scene_complexity")),
        scene_family=str(_scene_prop(summary, "scene_family") or summary.get("scenario", "")),
        num_users=str(_scene_prop(summary, "num_users")),
        num_obstacles=str(_scene_prop(summary, "num_obstacles")),
        duration_s=round(duration_s, 3),
        rendered_frames=rendered_frames,
        elapsed_s=round(time.time() - started, 3),
    )


def _body_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_ids: list[int] = []
    tokens = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        if any(token in name for token in tokens):
            joint_ids.append(joint_id)
    if len(joint_ids) != 29:
        raise ValueError(f"Expected 29 body joints, got {len(joint_ids)}")
    return model.jnt_qposadr[joint_ids].copy(), model.jnt_dofadr[joint_ids].copy()


def _apply_scene_props(props_manager: ScenePropsManager, scene_props: dict[str, Any]) -> None:
    placements = scene_props.get("placements", {}) if isinstance(scene_props, dict) else {}
    props_manager.apply_placements(placements)


def _apply_sim_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: dict[str, float],
    body_qpos_addr: np.ndarray,
    body_qvel_addr: np.ndarray,
) -> None:
    data.qpos[0:3] = [row["base_pos_x"], row["base_pos_y"], row["base_pos_z"]]
    quat = np.array([row["base_quat_w"], row["base_quat_x"], row["base_quat_y"], row["base_quat_z"]], dtype=float)
    norm = np.linalg.norm(quat)
    data.qpos[3:7] = quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])

    q_values = [row.get(f"body_q_{i}", math.nan) for i in range(len(body_qpos_addr))]
    for addr, value in zip(body_qpos_addr, q_values):
        if not math.isnan(value):
            data.qpos[int(addr)] = value

    if data.qvel.size >= 6:
        data.qvel[0:3] = [row.get("base_lin_vel_x", 0.0), row.get("base_lin_vel_y", 0.0), row.get("base_lin_vel_z", 0.0)]
        data.qvel[3:6] = [row.get("base_ang_vel_x", 0.0), row.get("base_ang_vel_y", 0.0), row.get("base_ang_vel_z", 0.0)]
    for index, addr in enumerate(body_qvel_addr):
        value = row.get(f"body_dq_{index}", math.nan)
        if not math.isnan(value):
            data.qvel[int(addr)] = value
    mujoco.mj_forward(model, data)


def _camera_frame_params(rows: list[dict[str, float]], scene_props: dict[str, Any]) -> tuple[np.ndarray, float]:
    points: list[tuple[float, float]] = [(row["base_pos_x"], row["base_pos_y"]) for row in rows]
    placements = scene_props.get("placements", {}) if isinstance(scene_props, dict) else {}
    for placement in placements.values():
        if placement.get("enabled", True):
            pos = placement.get("position", [0.0, 0.0, 0.0])
            if len(pos) >= 2 and float(pos[2] if len(pos) >= 3 else 0.0) > -1.0:
                points.append((float(pos[0]), float(pos[1])))

    if not points:
        center_xy = np.array([0.0, 0.0])
        radius = 1.0
    else:
        arr = np.array(points, dtype=float)
        center_xy = np.median(arr, axis=0)
        radius = float(np.max(np.linalg.norm(arr - center_xy, axis=1))) if len(points) > 1 else 1.0

    lookat = np.array([center_xy[0], center_xy[1], 0.55], dtype=float)
    distance = float(max(2.6, min(5.2, 1.55 * radius + 1.9)))
    return lookat, distance


def _make_camera(lookat: np.ndarray, distance: float, *, azimuth: float, elevation: float = -17.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _camera_for_rollout(rows: list[dict[str, float]], scene_props: dict[str, Any]) -> mujoco.MjvCamera:
    lookat, distance = _camera_frame_params(rows, scene_props)
    return _make_camera(lookat, distance, azimuth=145.0)


def _cameras_for_rollout(
    rows: list[dict[str, float]],
    scene_props: dict[str, Any],
    view_preset: str,
) -> list[tuple[str, mujoco.MjvCamera]]:
    lookat, distance = _camera_frame_params(rows, scene_props)
    if view_preset == "single":
        return [("main", _make_camera(lookat, distance, azimuth=145.0))]
    if view_preset == "cardinal4":
        return [
            ("front", _make_camera(lookat, distance, azimuth=0.0)),
            ("right", _make_camera(lookat, distance, azimuth=90.0)),
            ("back", _make_camera(lookat, distance, azimuth=180.0)),
            ("left", _make_camera(lookat, distance, azimuth=270.0)),
        ]
    raise ValueError(f"Unknown view preset: {view_preset}")


def _output_size(size: tuple[int, int], view_preset: str) -> tuple[int, int]:
    width, height = size
    if view_preset == "cardinal4":
        return width * 2, height * 2
    return width, height


def _render_frame(
    *,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    cameras: list[tuple[str, mujoco.MjvCamera]],
    view_preset: str,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for _, camera in cameras:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if view_preset == "cardinal4":
        top = np.hstack([frames[0], frames[1]])
        bottom = np.hstack([frames[2], frames[3]])
        return np.vstack([top, bottom])
    return frames[0]


def _draw_view_labels(image_bgr: np.ndarray, view_names: list[str], cell_size: tuple[int, int]) -> None:
    cell_w, cell_h = cell_size
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for name, (x0, y0) in zip(view_names, positions):
        label_y = y0 + cell_h - 38
        cv2.rectangle(image_bgr, (x0 + 8, label_y), (x0 + 150, label_y + 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            image_bgr,
            name,
            (x0 + 16, label_y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _read_sim_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    previous_sim_time: float | None = None
    with path.open(newline="") as file:
        for raw in csv.DictReader(file):
            parsed: dict[str, float] = {}
            for key, value in raw.items():
                if value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if "sim_time" in parsed:
                if previous_sim_time is not None and parsed["sim_time"] < previous_sim_time - 1e-6:
                    break
                rows.append(parsed)
                previous_sim_time = parsed["sim_time"]
    return rows


def _sample_rows_by_fps(rows: list[dict[str, float]], fps: float) -> list[dict[str, float]]:
    if len(rows) <= 1:
        return rows
    start = rows[0]["sim_time"]
    end = rows[-1]["sim_time"]
    step = 1.0 / fps
    selected: list[dict[str, float]] = []
    row_index = 0
    target = start
    while target <= end + 1e-6:
        while row_index + 1 < len(rows) and rows[row_index]["sim_time"] < target:
            row_index += 1
        selected.append(rows[row_index])
        target += step
    if not selected:
        return [rows[0]]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return selected


def _even_indices(count: int, target_count: int) -> list[int]:
    if count <= 0:
        return []
    if count <= target_count:
        return list(range(count))
    return sorted({int(round(i * (count - 1) / (target_count - 1))) for i in range(target_count)})


def _draw_overlay(image_bgr: np.ndarray, *, row_index: int, rollout_id: str, rel_t: float) -> None:
    text = f"#{row_index:03d} {rollout_id}  t={rel_t:05.2f}s"
    cv2.rectangle(image_bgr, (8, 8), (360, 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        image_bgr,
        text,
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _write_contact_sheet(path: Path, video_id: str, frames: list[tuple[float, np.ndarray]]) -> None:
    thumb_w, thumb_h = 480, 360
    header_h = 42
    sheet = np.full((thumb_h + header_h, thumb_w * len(frames), 3), 245, dtype=np.uint8)
    cv2.putText(
        sheet,
        video_id[:110],
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for index, (rel_t, frame) in enumerate(frames):
        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x0 = index * thumb_w
        sheet[header_h : header_h + thumb_h, x0 : x0 + thumb_w] = resized
        cv2.rectangle(sheet, (x0, header_h), (x0 + 92, header_h + 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            sheet,
            f"t={rel_t:.1f}s",
            (x0 + 8, header_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def write_manifest(path: Path, rows: list[RenderResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RenderResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_readme(output_root: Path, args: argparse.Namespace, rows: list[RenderResult], elapsed_s: float) -> None:
    rendered = sum(1 for row in rows if row.status == "rendered")
    skipped = sum(1 for row in rows if row.status == "skipped")
    errors = sum(1 for row in rows if row.status == "error")
    text = f"""# Phase 2 Full Video Review

Generated by `gear_sonic/scripts/casa_render_rollout_videos.py`.

- Source rollout CSV: `{args.rollouts_csv}`
- Model XML: `{args.model_xml}`
- Output FPS: `{args.fps}`
- Resolution: `{args.width}x{args.height}`
- View preset: `{args.view_preset}`
- Rendered: {rendered}
- Skipped existing: {skipped}
- Errors: {errors}
- Elapsed seconds: {elapsed_s:.1f}

Each MP4 is an offline reconstruction from `sim_state.csv` plus recorded CASA scene props, so it preserves the
accepted Phase 2 rollout state without rerunning the policy. The video frame overlay intentionally includes only
the rollout id and timestamp, not the oracle label, so VLM review is not directly leaked by on-frame text.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _scene_prop(summary: dict[str, Any], key: str) -> Any:
    scene_props = summary.get("scene_props", {})
    if isinstance(scene_props, dict) and key in scene_props:
        return scene_props.get(key, "")
    return summary.get(key, "")


def _oracle_yes_no(row: dict[str, str]) -> str:
    if row.get("violation_yes_no"):
        return row.get("violation_yes_no", "")
    if row.get("oracle_violation_yes_no"):
        return row.get("oracle_violation_yes_no", "")
    if row.get("oracle_label") == "unsafe":
        return "1"
    if row.get("oracle_label") == "safe":
        return "0"
    if row.get("final_label") == "safe":
        return "0"
    if row.get("final_label") == "unsafe":
        return "1"
    return ""


def _summary_path_for_row(row: dict[str, str], run_id: str, rollout_id: str) -> Path:
    summary_value = row.get("summary_path") or row.get("source_summary_path")
    if summary_value:
        candidate = _resolve_repo_path(summary_value)
        if candidate.exists():
            return candidate
    return REPO_ROOT / "outputs/casa/phase2_oracle" / run_id / rollout_id / "rollout_summary.json"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()

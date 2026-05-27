import argparse
import os
import select
import sys
import termios
import torch as t
import time
import tty

import mujoco
import numpy as np
from motionbricks.motion_backbone.demo.utils import navigation_demo


class OpenCVCamera:
    """Small viewer-compatible camera holder used by the WASD controller."""

    def __init__(self, distance: float = 4.0, azimuth: float = 90.0, elevation: float = -20.0):
        self.cam = mujoco.MjvCamera()
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.lookat[:] = np.array([0.0, 0.0, 0.8])

    def update_lookat(self, qpos_history: np.ndarray) -> None:
        self.cam.lookat[:] = qpos_history[:, :3].mean(axis=0)
        self.cam.lookat[2] += 0.6


class OpenCVRenderer:
    """Render MuJoCo offscreen and display frames through cv2.imshow."""

    WINDOW_NAME = "MotionBricks G1"

    def __init__(self, mj_model: mujoco.MjModel, height: int, width: int):
        import cv2

        self._cv2 = cv2
        self._renderer = mujoco.Renderer(mj_model, height=height, width=width)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def render(self, mj_data: mujoco.MjData, camera: mujoco.MjvCamera) -> bool:
        self._renderer.update_scene(mj_data, camera=camera)
        rgb = self._renderer.render()
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        self._cv2.imshow(self.WINDOW_NAME, bgr)
        self._cv2.waitKey(1)
        return True

    def close(self) -> None:
        if hasattr(self._renderer, "close"):
            self._renderer.close()
        self._cv2.destroyWindow(self.WINDOW_NAME)


class TerminalKeyboardInput:
    """Non-blocking terminal keyboard control for SSH sessions.

    Terminal input does not expose key-release events, so movement keys are
    active briefly after each keypress and stay active while key repeat fires.
    Style keys are latched until another style is selected or cleared.
    """

    MOVEMENT_KEYS = ("w", "a", "s", "d")
    STYLE_KEYS = ("v", "z", "x", "b", "r", "t", "c", "e", "f", "g", "q")
    CANDIDATES = MOVEMENT_KEYS + ("left", "right", "up", "down", "shift", "ctrl", "enter") + STYLE_KEYS

    def __init__(self, hold_seconds: float = 0.6):
        self._hold_seconds = hold_seconds
        self._fd = sys.stdin.fileno()
        self._is_tty = sys.stdin.isatty()
        self._old_settings = None
        self._pressed_until = {key: 0.0 for key in self.MOVEMENT_KEYS}
        self._latched_style = None
        self._exit_requested = False

    def __enter__(self):
        if self._is_tty:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            print(
                "Terminal controls: hold W/A/S/D to move; "
                "press V/Z/X/B/R/T/C/E/F/G/Q to latch a style; "
                "press Space or 0 for normal walk; press Esc or Ctrl-C to quit.",
                flush=True,
            )
        else:
            print("Warning: stdin is not a TTY; terminal keyboard controls are disabled.", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def snapshot(self):
        self._read_available_keys()
        now = time.monotonic()
        key_pressed = {key: False for key in self.CANDIDATES}
        for key, active_until in self._pressed_until.items():
            key_pressed[key] = active_until > now
        if self._latched_style is not None:
            key_pressed[self._latched_style] = True
        return key_pressed

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    def _read_available_keys(self) -> None:
        if not self._is_tty:
            return

        now = time.monotonic()
        while select.select([sys.stdin], [], [], 0)[0]:
            raw = os.read(self._fd, 1)
            if not raw:
                return
            if raw == b"\x1b":
                if self._consume_escape_sequence():
                    continue
                self._exit_requested = True
                return

            char = raw.decode(errors="ignore").lower()
            if char in self.MOVEMENT_KEYS:
                self._pressed_until[char] = now + self._hold_seconds
            elif char in self.STYLE_KEYS:
                self._latched_style = char
            elif char in (" ", "0"):
                self._latched_style = None
                for key in self._pressed_until:
                    self._pressed_until[key] = 0.0

    def _consume_escape_sequence(self) -> bool:
        consumed_sequence = False
        while select.select([sys.stdin], [], [], 0)[0]:
            os.read(self._fd, 1)
            consumed_sequence = True
        return consumed_sequence


def _run_demo_step(demo_agent, camera_view, args, steps: int, show_visuals: bool, key_pressed: dict):
    force_idle = steps + 100 > args.max_steps
    qpos = demo_agent.full_agent.get_next_frame()
    context_motion_features = demo_agent.full_agent.get_context_motion_features()
    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
    demo_agent.mj_data.qpos[:] = qpos

    control_signals = demo_agent.controller.generate_control_signals(
        camera_view, demo_agent.mj_model, demo_agent.mj_data, visualize=show_visuals,
        control_info={"force_idle": force_idle,
                      'allowed_mode': getattr(args, 'allowed_mode', None),
                      'key_pressed': key_pressed}
    )

    if args.use_qpos:
        control_signals['context_mujoco_qpos'] = context_mujoco_qpos
    else:
        control_signals['context_motion_features'] = context_motion_features

    with t.no_grad():
        demo_agent.full_agent.generate_new_frames(
            control_signals,
            demo_agent.controller.get_controller_dt() * args.generate_dt
        )

    mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
    camera_view.update_lookat(demo_agent.controller.get_prev_qpos())


def main(args) -> None:
    demo_agent = navigation_demo(args)
    camera_view = OpenCVCamera(
        distance=args.camera_distance,
        azimuth=args.camera_azimuth,
        elevation=args.camera_elevation,
    )
    cv_renderer = OpenCVRenderer(demo_agent.mj_model, args.render_height, args.render_width) \
        if args.has_viewer else None

    try:
        num_runs = 0
        with TerminalKeyboardInput(args.terminal_key_hold) as terminal_input:
            while num_runs < args.num_runs:
                num_runs += 1
                print(f"Running iteration {num_runs}... / {args.num_runs}")
                random_seed = args.random_seed * (num_runs + 2333) * 2333 % (2 ** 32 - 1)
                np.random.seed(random_seed)
                t.manual_seed(random_seed)
                demo_agent.full_agent.reset()

                steps = 0
                while steps < args.max_steps:
                    steps += 1
                    step_start = time.time()
                    key_pressed = terminal_input.snapshot()
                    if terminal_input.exit_requested:
                        return

                    _run_demo_step(
                        demo_agent, camera_view, args, steps,
                        show_visuals=bool(args.has_viewer),
                        key_pressed=key_pressed,
                    )
                    if cv_renderer is not None and not cv_renderer.render(demo_agent.mj_data, camera_view.cam):
                        return

                    time_until_next_step = demo_agent.mj_model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
    finally:
        if cv_renderer is not None:
            cv_renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive demo for the G1 humanoid")

    # path configs
    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    # controller config
    parser.add_argument("--controller", type=str, default="wasd",
                        choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--has_viewer", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--camera_distance", type=float, default=4.0)
    parser.add_argument("--camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera_elevation", type=float, default=-20.0)
    parser.add_argument("--terminal_key_hold", type=float, default=0.6)

    # run configs
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)

    # model configurations
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    main(args)

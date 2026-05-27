"""Headless ZMQ camera recorder for simulation/robot camera streams."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import cv2
import tyro

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


@dataclass
class CameraRecorderConfig:
    camera_host: str = "localhost"
    """Camera server hostname."""

    camera_port: int = 5555
    """Camera server port."""

    fps: int = 10
    """Recording FPS."""

    duration: Optional[float] = 60.0
    """Recording duration in seconds. Use 0 to record until Ctrl-C."""

    output_path: str = "camera_recordings"
    """Directory where recording folders are saved."""

    codec: str = "mp4v"
    """Video codec for recording, for example mp4v or XVID."""

    verbose: bool = False
    """Print camera client diagnostics."""


def main(config: CameraRecorderConfig):
    client = ComposedCameraClientSensor(
        server_ip=config.camera_host,
        port=config.camera_port,
        verbose=config.verbose,
    )

    print("Waiting for first camera frame...")
    sample = None
    for _ in range(200):
        sample = client.read(blocking=False)
        if sample and sample.get("images"):
            break
        time.sleep(0.1)

    if sample is None or not sample.get("images"):
        print("ERROR: No camera frames received after 20s. Check the camera server.")
        client.close()
        return

    camera_names = sorted(sample["images"].keys())
    recording_dir = Path(config.output_path) / f"rec_{time.strftime('%Y%m%d_%H%M%S')}"
    recording_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    writers: dict[str, cv2.VideoWriter] = {}
    for name in camera_names:
        img = sample["images"].get(name)
        if img is None:
            continue
        h, w = img.shape[:2]
        writers[name] = cv2.VideoWriter(
            str(recording_dir / f"{name}.mp4"), fourcc, config.fps, (w, h)
        )

    if not writers:
        print("ERROR: No writable camera streams detected.")
        client.close()
        return

    print(f"Recording {camera_names} at {config.fps} FPS -> {recording_dir}")
    if config.duration and config.duration > 0:
        print(f"Duration: {config.duration:.1f}s")
    else:
        print("Duration: until Ctrl-C")

    loop_period = 1.0 / config.fps
    start_time = time.monotonic()
    frame_count = 0

    try:
        while True:
            t_start = time.monotonic()
            if config.duration and config.duration > 0:
                if t_start - start_time >= config.duration:
                    break

            image_data = client.read(blocking=False)
            if image_data and image_data.get("images"):
                for name, writer in writers.items():
                    img = image_data["images"].get(name)
                    if img is None:
                        continue
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(img)
                frame_count += 1

            remaining = loop_period - (time.monotonic() - t_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    finally:
        for writer in writers.values():
            writer.release()
        client.close()

    elapsed = time.monotonic() - start_time
    print(f"Saved {frame_count} frame ticks over {elapsed:.1f}s -> {recording_dir}")


if __name__ == "__main__":
    main(tyro.cli(CameraRecorderConfig))

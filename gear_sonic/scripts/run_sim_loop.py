"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from typing import Dict

import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    if config.image_publish_fps is not None:
        if config.image_publish_fps <= 0:
            raise ValueError("--image-publish-fps must be positive")
        wbc_config["IMAGE_DT"] = 1.0 / config.image_publish_fps

    if config.drop_after_seconds is not None:
        if config.drop_after_seconds < 0:
            raise ValueError("--drop-after-seconds must be non-negative")
        wbc_config["DROP_ELASTIC_AFTER_SECONDS"] = config.drop_after_seconds

    if config.fall_log_interval_seconds < 0:
        raise ValueError("--fall-log-interval-seconds must be non-negative")
    if config.sim_timing_log_interval_seconds < 0:
        raise ValueError("--sim-timing-log-interval-seconds must be non-negative")
    if config.episode_log_dir is not None and config.episode_log_fps <= 0:
        raise ValueError("--episode-log-fps must be positive when episode logging is enabled")
    wbc_config["FALL_LOG_INTERVAL_SECONDS"] = config.fall_log_interval_seconds
    wbc_config["FALL_STATS"] = config.fall_stats
    wbc_config["RESET_ON_FALL"] = not config.disable_reset_on_fall
    wbc_config["SIM_TIMING_LOG_INTERVAL_SECONDS"] = config.sim_timing_log_interval_seconds
    wbc_config["EPISODE_LOG_DIR"] = config.episode_log_dir
    wbc_config["EPISODE_LOG_FPS"] = config.episode_log_fps
    wbc_config["EPISODE_NAME"] = config.episode_name
    if config.robot_scene is not None:
        wbc_config["ROBOT_SCENE"] = config.robot_scene
    if config.domain_id is not None:
        wbc_config["DOMAIN_ID"] = config.domain_id
    wbc_config["CASA_PROPS_CONFIG"] = config.casa_props_config
    wbc_config["CASA_PROPS_SEED"] = config.casa_props_seed
    wbc_config["CASA_PROPS_COMMAND_FILE"] = config.casa_props_command_file
    wbc_config["CASA_PROPS_ACK_FILE"] = config.casa_props_ack_file

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
        offscreen_camera=config.offscreen_camera,
        offscreen_camera_width=config.offscreen_camera_width,
        offscreen_camera_height=config.offscreen_camera_height,
        offscreen_camera_distance=config.offscreen_camera_distance,
        offscreen_camera_azimuth=config.offscreen_camera_azimuth,
        offscreen_camera_elevation=config.offscreen_camera_elevation,
    )

    if config.drop_on_start:
        elastic_band = getattr(sim_wrapper.sim.sim_env, "elastic_band", None)
        if elastic_band is None:
            raise RuntimeError("--drop-on-start requires an elastic band in the simulator config")
        elastic_band.enable = False
        print("ElasticBand enable: False")

    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)

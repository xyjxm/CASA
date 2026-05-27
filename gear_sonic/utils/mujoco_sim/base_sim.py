"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import csv
import json
import os
import pathlib
from pathlib import Path
import pickle
import random
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_body_geom_ids, get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
        offscreen_camera: str = "track",
        offscreen_camera_width: int = 960,
        offscreen_camera_height: int = 720,
        offscreen_camera_distance: float = 3.0,
        offscreen_camera_azimuth: float = 120.0,
        offscreen_camera_elevation: float = -20.0,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs
        self.offscreen_camera = offscreen_camera
        self.offscreen_camera_width = offscreen_camera_width
        self.offscreen_camera_height = offscreen_camera_height
        self.offscreen_camera_distance = offscreen_camera_distance
        self.offscreen_camera_azimuth = offscreen_camera_azimuth
        self.offscreen_camera_elevation = offscreen_camera_elevation

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "mujoco_view": {
                    "height": self.offscreen_camera_height,
                    "width": self.offscreen_camera_width,
                    "mjcf_name": "head_camera",
                },
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen
        self.start_wall_time = time.monotonic()
        self.fall = False
        self.fall_count = 0
        self.reset_count = 0
        self.last_fall_log_time = -float("inf")
        self.fall_log_interval_seconds = self.config.get("FALL_LOG_INTERVAL_SECONDS", 5.0)
        self.fall_stats = self.config.get("FALL_STATS", True)
        self.reset_on_fall = self.config.get("RESET_ON_FALL", True)
        self.fall_stats_printed = False
        self.episode_log_enabled = False
        self.episode_log_file = None
        self.episode_log_writer = None
        self.episode_log_path = None
        self.episode_summary_path = None
        self.scene_props_path = None
        self.episode_log_closed = False
        self.episode_log_last_sim_time = -float("inf")
        self.episode_rows = 0
        self.episode_first_sim_time = None
        self.episode_last_sim_time = None
        self.episode_fall_rows = 0
        self.episode_self_collision_rows = 0
        self.episode_left_foot_contact_rows = 0
        self.episode_right_foot_contact_rows = 0
        self.episode_elastic_band_enabled_rows = 0
        self.episode_max_elastic_band_force_norm = 0.0
        self.casa_props_manager = None
        self.casa_props_rng = random.Random(int(self.config.get("CASA_PROPS_SEED", 1234)))
        self.casa_scene_props = {}
        self.casa_velocity_perturbations = []
        self.casa_props_command_path = self._optional_config_path("CASA_PROPS_COMMAND_FILE")
        self.casa_props_ack_path = self._optional_config_path("CASA_PROPS_ACK_FILE")
        if self.casa_props_command_path is not None and self.casa_props_ack_path is None:
            self.casa_props_ack_path = self.casa_props_command_path.with_suffix(
                self.casa_props_command_path.suffix + ".ack"
            )
        self.casa_last_props_command_id = None
        self.control_loop_overrun = False
        self.elastic_band = None
        self.elastic_band_force_norm = 0.0

        self.init_scene()
        self.init_episode_logging()
        self.last_reward = 0

        self.offscreen = offscreen
        self.configure_offscreen_cameras()
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def configure_offscreen_cameras(self):
        if not self.camera_configs:
            return
        if self.offscreen_camera == "head_camera":
            return
        if self.offscreen_camera != "track":
            raise ValueError(f"Unsupported offscreen camera: {self.offscreen_camera}")

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.root_body_id
        camera.distance = self.offscreen_camera_distance
        camera.azimuth = self.offscreen_camera_azimuth
        camera.elevation = self.offscreen_camera_elevation
        camera.lookat = np.array([0.0, 0.0, 0.7])

        for camera_config in self.camera_configs.values():
            camera_config.pop("mjcf_name", None)
            camera_config["params"] = camera

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)
        self.body_qpos_addr = self.mj_model.jnt_qposadr[self.body_joint_index]
        self.body_qvel_addr = self.mj_model.jnt_dofadr[self.body_joint_index]
        self.init_casa_scene_props()

    def init_casa_scene_props(self):
        props_config_path = self.config.get("CASA_PROPS_CONFIG")
        if not props_config_path:
            return
        from gear_sonic.casa.scene import ScenePropsManager, load_scene_props_config

        props_path = Path(props_config_path)
        if not props_path.is_absolute():
            props_path = GEAR_SONIC_ROOT / props_path
        props_config = load_scene_props_config(props_path)
        self.casa_props_manager = ScenePropsManager(self.mj_model, self.mj_data, props_config)
        self.casa_scene_props = self.casa_props_manager.randomize_episode(self.casa_props_rng)
        print(f"CASA scene props enabled: {props_path}")

    def init_episode_logging(self):
        episode_log_dir = self.config.get("EPISODE_LOG_DIR")
        if not episode_log_dir:
            return

        episode_log_fps = float(self.config.get("EPISODE_LOG_FPS", 50.0))
        if episode_log_fps <= 0:
            raise ValueError("EPISODE_LOG_FPS must be positive")

        self.episode_name = self.config.get("EPISODE_NAME", "stage1_smoke")
        self.episode_log_dt = 1.0 / episode_log_fps
        log_dir = Path(episode_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log_path = log_dir / "sim_state.csv"
        self.episode_summary_path = log_dir / "episode_summary.json"
        self.scene_props_path = log_dir / "scene_props.json"
        self.episode_log_file = self.episode_log_path.open("w", newline="")
        self.episode_log_writer = csv.DictWriter(
            self.episode_log_file,
            fieldnames=self._episode_log_fieldnames(),
        )
        self.episode_log_writer.writeheader()
        self.episode_log_enabled = True
        self._cache_episode_geom_sets()
        self.write_scene_props()
        print(f"Episode logging enabled: {self.episode_log_path}")

    def write_scene_props(self):
        if not self.scene_props_path or not self.casa_scene_props:
            return
        with self.scene_props_path.open("w") as props_file:
            json.dump(self.casa_scene_props, props_file, indent=2, sort_keys=True)

    def _episode_log_fieldnames(self):
        fields = [
            "episode_name",
            "sim_time",
            "wall_time",
            "reset_count",
            "fall_flag",
            "fall_count",
            "base_pos_x",
            "base_pos_y",
            "base_pos_z",
            "base_quat_w",
            "base_quat_x",
            "base_quat_y",
            "base_quat_z",
            "base_lin_vel_x",
            "base_lin_vel_y",
            "base_lin_vel_z",
            "base_ang_vel_x",
            "base_ang_vel_y",
            "base_ang_vel_z",
            "torso_roll",
            "torso_pitch",
            "torso_yaw",
            "left_foot_contact",
            "right_foot_contact",
            "self_collision",
            "self_collision_pair_count",
            "min_user_distance",
            "min_arm_user_distance",
            "min_obstacle_distance",
            "external_collision_user",
            "external_collision_obstacle",
            "elastic_band_enabled",
            "elastic_band_force_norm",
            "control_loop_overrun",
        ]
        fields.extend(f"body_q_{i}" for i in range(self.num_body_dof))
        fields.extend(f"body_dq_{i}" for i in range(self.num_body_dof))
        return fields

    def _cache_episode_geom_sets(self):
        self.floor_geom_id = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        self.left_foot_geom_ids = self._get_geom_ids_for_body("left_ankle_roll_link")
        self.right_foot_geom_ids = self._get_geom_ids_for_body("right_ankle_roll_link")

        robot_bodies = get_subtree_body_names(self.mj_model, self.root_body_id)
        robot_geom_ids = set()
        for body_name in robot_bodies:
            robot_geom_ids.update(get_body_geom_ids(self.mj_model, self.mj_model.body(body_name).id))
        self.user_geom_ids = self._get_geom_ids_by_prefix("user_proxy_")
        self.obstacle_geom_ids = self._get_geom_ids_by_prefix("obstacle_")
        self.robot_geom_ids = robot_geom_ids - self.user_geom_ids - self.obstacle_geom_ids
        self.arm_geom_ids = self._get_arm_geom_ids()

    def _get_geom_ids_for_body(self, body_name: str) -> set[int]:
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return set()
        return set(get_body_geom_ids(self.mj_model, body_id))

    def _get_geom_ids_by_prefix(self, prefix: str) -> set[int]:
        geom_ids = set()
        for geom_id in range(self.mj_model.ngeom):
            geom_name = self.mj_model.geom(geom_id).name
            body_name = self._body_name_for_geom(geom_id)
            if geom_name.startswith(prefix) or body_name.startswith(prefix):
                geom_ids.add(geom_id)
        return geom_ids

    def _get_arm_geom_ids(self) -> set[int]:
        arm_tokens = ("shoulder", "elbow", "wrist", "hand")
        geom_ids = set()
        for geom_id in self.robot_geom_ids:
            body_name = self._body_name_for_geom(geom_id)
            if any(token in body_name for token in arm_tokens):
                geom_ids.add(geom_id)
        return geom_ids

    def _body_name_for_geom(self, geom_id: int) -> str:
        body_id = self.mj_model.geom(geom_id).bodyid
        return self.mj_model.body(body_id).name

    def _foot_contact(self, foot_geom_ids: set[int]) -> bool:
        if self.floor_geom_id == -1 or not foot_geom_ids:
            return False
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if (geom1 in foot_geom_ids and geom2 == self.floor_geom_id) or (
                geom2 in foot_geom_ids and geom1 == self.floor_geom_id
            ):
                return True
        return False

    def _self_collision_pairs(self) -> set[tuple[str, str]]:
        pairs = set()
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 not in self.robot_geom_ids or geom2 not in self.robot_geom_ids:
                continue
            body1 = self._body_name_for_geom(geom1)
            body2 = self._body_name_for_geom(geom2)
            if body1 == body2:
                continue
            pairs.add(tuple(sorted((body1, body2))))
        return pairs

    def _external_collision(self, external_geom_ids: set[int]) -> bool:
        if not external_geom_ids:
            return False
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if (geom1 in self.robot_geom_ids and geom2 in external_geom_ids) or (
                geom2 in self.robot_geom_ids and geom1 in external_geom_ids
            ):
                return True
        return False

    def _min_geom_distance(self, source_geom_ids: set[int], target_geom_ids: set[int]) -> float | None:
        if not source_geom_ids or not target_geom_ids:
            return None
        min_distance = float("inf")
        for source_id in source_geom_ids:
            source_pos = self.mj_data.geom_xpos[source_id]
            source_radius = float(self.mj_model.geom_rbound[source_id])
            for target_id in target_geom_ids:
                target_pos = self.mj_data.geom_xpos[target_id]
                target_radius = float(self.mj_model.geom_rbound[target_id])
                center_distance = float(np.linalg.norm(source_pos - target_pos))
                distance = max(0.0, center_distance - source_radius - target_radius)
                min_distance = min(min_distance, distance)
        if min_distance == float("inf"):
            return None
        return min_distance

    def _torso_euler_xyz(self) -> np.ndarray:
        torso_quat_xyzw = self.mj_data.xquat[self.torso_index][[1, 2, 3, 0]]
        return Rotation.from_quat(torso_quat_xyzw).as_euler("xyz", degrees=False)

    def _episode_log_row(self, fall_flag: bool):
        if self.use_floating_root_link:
            base_pose = self.mj_data.qpos[:7]
            base_vel = self.mj_data.qvel[:6]
        else:
            base_pose = np.zeros(7)
            base_vel = np.zeros(6)

        torso_roll, torso_pitch, torso_yaw = self._torso_euler_xyz()
        body_q = self.mj_data.qpos[self.body_qpos_addr]
        body_dq = self.mj_data.qvel[self.body_qvel_addr]
        left_foot_contact = self._foot_contact(self.left_foot_geom_ids)
        right_foot_contact = self._foot_contact(self.right_foot_geom_ids)
        self_collision_pairs = self._self_collision_pairs()
        self_collision = bool(self_collision_pairs)
        min_user_distance = self._min_geom_distance(self.robot_geom_ids, self.user_geom_ids)
        min_arm_user_distance = self._min_geom_distance(self.arm_geom_ids, self.user_geom_ids)
        min_obstacle_distance = self._min_geom_distance(self.robot_geom_ids, self.obstacle_geom_ids)
        external_collision_user = self._external_collision(self.user_geom_ids)
        external_collision_obstacle = self._external_collision(self.obstacle_geom_ids)
        elastic_band = getattr(self, "elastic_band", None)
        elastic_band_enabled = bool(elastic_band and elastic_band.enable)

        row = {
            "episode_name": self.episode_name,
            "sim_time": float(self.mj_data.time),
            "wall_time": time.time(),
            "reset_count": self.reset_count,
            "fall_flag": int(fall_flag),
            "fall_count": self.fall_count,
            "base_pos_x": float(base_pose[0]),
            "base_pos_y": float(base_pose[1]),
            "base_pos_z": float(base_pose[2]),
            "base_quat_w": float(base_pose[3]),
            "base_quat_x": float(base_pose[4]),
            "base_quat_y": float(base_pose[5]),
            "base_quat_z": float(base_pose[6]),
            "base_lin_vel_x": float(base_vel[0]),
            "base_lin_vel_y": float(base_vel[1]),
            "base_lin_vel_z": float(base_vel[2]),
            "base_ang_vel_x": float(base_vel[3]),
            "base_ang_vel_y": float(base_vel[4]),
            "base_ang_vel_z": float(base_vel[5]),
            "torso_roll": float(torso_roll),
            "torso_pitch": float(torso_pitch),
            "torso_yaw": float(torso_yaw),
            "left_foot_contact": int(left_foot_contact),
            "right_foot_contact": int(right_foot_contact),
            "self_collision": int(self_collision),
            "self_collision_pair_count": len(self_collision_pairs),
            "min_user_distance": min_user_distance,
            "min_arm_user_distance": min_arm_user_distance,
            "min_obstacle_distance": min_obstacle_distance,
            "external_collision_user": int(external_collision_user),
            "external_collision_obstacle": int(external_collision_obstacle),
            "elastic_band_enabled": int(elastic_band_enabled),
            "elastic_band_force_norm": float(self.elastic_band_force_norm),
            "control_loop_overrun": int(self.control_loop_overrun),
        }
        row.update({f"body_q_{i}": float(v) for i, v in enumerate(body_q)})
        row.update({f"body_dq_{i}": float(v) for i, v in enumerate(body_dq)})
        return row

    def record_episode_state(self, fall_flag: bool = False, force: bool = False):
        if not self.episode_log_enabled or self.episode_log_writer is None:
            return

        sim_time = float(self.mj_data.time)
        if not force and sim_time - self.episode_log_last_sim_time < self.episode_log_dt:
            return

        row = self._episode_log_row(fall_flag=fall_flag)
        self.episode_log_writer.writerow(row)
        self.episode_log_file.flush()

        self.episode_log_last_sim_time = sim_time
        self.episode_rows += 1
        if self.episode_first_sim_time is None:
            self.episode_first_sim_time = sim_time
        self.episode_last_sim_time = sim_time
        self.episode_fall_rows += int(row["fall_flag"])
        self.episode_self_collision_rows += int(row["self_collision"])
        self.episode_left_foot_contact_rows += int(row["left_foot_contact"])
        self.episode_right_foot_contact_rows += int(row["right_foot_contact"])
        self.episode_elastic_band_enabled_rows += int(row["elastic_band_enabled"])
        self.episode_max_elastic_band_force_norm = max(
            self.episode_max_elastic_band_force_norm,
            float(row["elastic_band_force_norm"] or 0.0),
        )

    def write_episode_summary(self):
        if not self.episode_log_enabled or self.episode_summary_path is None:
            return

        duration = 0.0
        if self.episode_first_sim_time is not None and self.episode_last_sim_time is not None:
            duration = max(0.0, self.episode_last_sim_time - self.episode_first_sim_time)

        summary = {
            "episode_name": self.episode_name,
            "sim_state_csv": str(self.episode_log_path),
            "rows": self.episode_rows,
            "duration_seconds": duration,
            "fall_count": self.fall_count,
            "fall_rows": self.episode_fall_rows,
            "self_collision_count": self.episode_self_collision_rows,
            "left_foot_contact_rows": self.episode_left_foot_contact_rows,
            "right_foot_contact_rows": self.episode_right_foot_contact_rows,
            "elastic_band_enabled_rows": self.episode_elastic_band_enabled_rows,
            "elastic_band_enabled_ratio": (
                self.episode_elastic_band_enabled_rows / self.episode_rows if self.episode_rows else 0.0
            ),
            "max_elastic_band_force_norm": self.episode_max_elastic_band_force_norm,
            "reset_count": self.reset_count,
            "scene_props_json": str(self.scene_props_path) if self.casa_scene_props else None,
            "scene_props": self.casa_scene_props,
        }
        with self.episode_summary_path.open("w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def close_episode_logging(self):
        if self.episode_log_closed:
            return
        self.episode_log_closed = True
        if self.episode_log_enabled:
            self.write_episode_summary()
        if self.episode_log_file is not None:
            self.episode_log_file.close()

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            self.mj_model.vis.global_.offwidth = max(
                self.mj_model.vis.global_.offwidth, camera_config["width"]
            )
            self.mj_model.vis.global_.offheight = max(
                self.mj_model.vis.global_.offheight, camera_config["height"]
            )
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.maybe_apply_casa_props_command()
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        elastic_band = getattr(self, "elastic_band", None)
        self.elastic_band_force_norm = 0.0
        if elastic_band:
            if elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                band_force = elastic_band.Advance(pose)
                self.elastic_band_force_norm = float(np.linalg.norm(band_force))
                self.mj_data.xfrc_applied[self.band_attached_link] = band_force
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        self.apply_casa_velocity_perturbations()
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        fall_flag = self.is_fallen()
        self.record_episode_state(fall_flag=fall_flag, force=fall_flag)
        self.check_fall(fall_flag=fall_flag)

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_casa_velocity_perturbations(self, raw_events):
        self.casa_velocity_perturbations = []
        if not isinstance(raw_events, list):
            return
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            velocity_body = raw.get(
                "velocity_body",
                [raw.get("vx_body", 0.0), raw.get("vy_body", 0.0), raw.get("vz_body", 0.0)],
            )
            if not isinstance(velocity_body, (list, tuple)) or len(velocity_body) < 3:
                continue
            angular_velocity_body = raw.get("angular_velocity_body", [0.0, 0.0, 0.0])
            if not isinstance(angular_velocity_body, (list, tuple)) or len(angular_velocity_body) < 3:
                angular_velocity_body = [0.0, 0.0, 0.0]
            self.casa_velocity_perturbations.append(
                {
                    "time_s": float(raw.get("time_s", 0.0)),
                    "velocity_body": [float(velocity_body[0]), float(velocity_body[1]), float(velocity_body[2])],
                    "angular_velocity_body": [
                        float(angular_velocity_body[0]),
                        float(angular_velocity_body[1]),
                        float(angular_velocity_body[2]),
                    ],
                    "applied": False,
                }
            )

    def apply_casa_velocity_perturbations(self):
        if not self.use_floating_root_link:
            return
        sim_time = float(self.mj_data.time)
        for event in self.casa_velocity_perturbations:
            if event.get("applied") or sim_time < event["time_s"]:
                continue
            vel_body = np.array(event["velocity_body"], dtype=float)
            vel_world = np.zeros(3)
            base_quat = self.mj_data.qpos[3:7]
            mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)
            self.mj_data.qvel[0] += vel_world[0]
            self.mj_data.qvel[1] += vel_world[1]
            self.mj_data.qvel[2] += vel_world[2]
            angular_velocity_body = event.get("angular_velocity_body", [0.0, 0.0, 0.0])
            self.mj_data.qvel[3] += angular_velocity_body[0]
            self.mj_data.qvel[4] += angular_velocity_body[1]
            self.mj_data.qvel[5] += angular_velocity_body[2]
            event["applied"] = True
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def _optional_config_path(self, key: str) -> Path | None:
        value = self.config.get(key)
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = GEAR_SONIC_ROOT / path
        return path

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def is_fallen(self):
        return bool(self.mj_data.qpos[2] < 0.2)

    def check_fall(self, fall_flag=None):
        self.fall = self.is_fallen() if fall_flag is None else bool(fall_flag)
        if self.fall:
            self.fall_count += 1
            now = time.monotonic()
            if now - self.last_fall_log_time >= self.fall_log_interval_seconds:
                elapsed = now - self.start_wall_time
                print(
                    f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m "
                    f"(falls={self.fall_count}, elapsed={elapsed:.1f}s)"
                )
                self.last_fall_log_time = now

        if self.fall and self.reset_on_fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        self.reset_count += 1
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.episode_log_last_sim_time = -float("inf")
        self.elastic_band_force_norm = 0.0
        if self.casa_props_manager is not None:
            self.casa_props_manager.apply_current_placements()

    def maybe_apply_casa_props_command(self):
        if self.casa_props_command_path is None or not self.casa_props_command_path.exists():
            return
        try:
            with self.casa_props_command_path.open() as command_file:
                command = json.load(command_file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: failed to read CASA props command: {exc}")
            return

        command_id = command.get("command_id")
        if command_id is None:
            command_id = str(self.casa_props_command_path.stat().st_mtime_ns)
        if command_id == self.casa_last_props_command_id:
            return
        self.casa_last_props_command_id = command_id

        if command.get("reset", True):
            self.reset()

        if self.casa_props_manager is not None and command.get("placements"):
            self.casa_scene_props = self.casa_props_manager.apply_placements(command["placements"])
        elif self.casa_props_manager is not None and command.get("randomize", True):
            seed = command.get("seed")
            rng = random.Random(int(seed)) if seed is not None else self.casa_props_rng
            close_user_bias_prob = command.get("close_user_bias_prob")
            if close_user_bias_prob is not None:
                close_user_bias_prob = float(close_user_bias_prob)
            self.casa_scene_props = self.casa_props_manager.randomize_episode(
                rng,
                close_user_bias_prob=close_user_bias_prob,
            )
        elif self.casa_props_manager is not None:
            self.casa_props_manager.apply_current_placements()

        force_base_height = command.get("force_base_height")
        if force_base_height is not None and self.use_floating_root_link:
            self.mj_data.qpos[2] = float(force_base_height)
            mujoco.mj_forward(self.mj_model, self.mj_data)

        restore_result = None
        if command.get("restore_state") is not None:
            restore_result = self._apply_casa_restore_state(command["restore_state"])

        self.set_casa_velocity_perturbations(command.get("velocity_perturbations", []))

        if self.casa_scene_props:
            self.casa_scene_props = dict(self.casa_scene_props)
            self.casa_scene_props["episode_id"] = command.get("episode_id")
            self.casa_scene_props["command_id"] = command_id
            self.casa_scene_props["command_wall_time"] = time.time()
            for key in [
                "target_bucket",
                "scene_complexity",
                "scene_family",
                "num_users",
                "num_obstacles",
                "generator_version",
                "fall_trigger",
                "velocity_perturbations",
                "velocity_perturbation_clean_start_offset_s",
            ]:
                if command.get(key) is not None:
                    self.casa_scene_props[key] = command.get(key)
            if command.get("close_user_bias_prob") is not None:
                self.casa_scene_props["requested_close_user_bias_prob"] = float(
                    command["close_user_bias_prob"]
                )
            self.write_scene_props()

        if self.casa_props_ack_path is not None:
            self.casa_props_ack_path.parent.mkdir(parents=True, exist_ok=True)
            ack = {
                "command_id": command_id,
                "episode_id": command.get("episode_id"),
                "scenario": command.get("scenario"),
                "applied": True,
                "reset_count": self.reset_count,
                "wall_time": time.time(),
                "scene_props": self.casa_scene_props,
                "restore_state": restore_result,
            }
            tmp_path = self.casa_props_ack_path.with_suffix(self.casa_props_ack_path.suffix + ".tmp")
            with tmp_path.open("w") as ack_file:
                json.dump(ack, ack_file, indent=2, sort_keys=True)
            tmp_path.replace(self.casa_props_ack_path)

    def _apply_casa_restore_state(self, restore_state):
        if not isinstance(restore_state, dict):
            return {"applied": False, "error": "restore_state_not_dict"}
        qpos = self._coerce_state_vector(restore_state.get("qpos"), self.mj_model.nq, "qpos")
        qvel = self._coerce_state_vector(restore_state.get("qvel"), self.mj_model.nv, "qvel")
        if qpos.get("error") or qvel.get("error"):
            return {"applied": False, "qpos": self._state_vector_status(qpos), "qvel": self._state_vector_status(qvel)}
        self.mj_data.qpos[:] = qpos["values"]
        self.mj_data.qvel[:] = qvel["values"]
        if restore_state.get("sim_time") is not None:
            self.mj_data.time = float(restore_state["sim_time"])
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return {
            "applied": True,
            "qpos_len": int(self.mj_model.nq),
            "qvel_len": int(self.mj_model.nv),
            "sim_time": float(self.mj_data.time),
        }

    @staticmethod
    def _coerce_state_vector(values, expected_len: int, name: str):
        if values is None:
            return {"error": f"{name}_missing"}
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            return {"error": f"{name}_invalid:{exc}"}
        if len(array) != expected_len:
            return {"error": f"{name}_length_mismatch:{len(array)}!={expected_len}"}
        return {"values": array}

    @staticmethod
    def _state_vector_status(vector):
        if vector.get("error"):
            return {"error": vector["error"]}
        return {"len": int(len(vector["values"]))}

    def close(self):
        self.close_episode_logging()

    def print_fall_stats(self):
        if not self.fall_stats or self.fall_stats_printed:
            return
        self.fall_stats_printed = True
        elapsed = max(time.monotonic() - self.start_wall_time, 1e-6)
        falls_per_minute = self.fall_count / elapsed * 60.0
        print(
            f"Fall stats: falls={self.fall_count}, elapsed={elapsed:.1f}s, "
            f"falls_per_minute={falls_per_minute:.2f}"
        )


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self.reward_interval_steps = max(1, round(self.reward_dt / self.sim_dt))
        self.image_interval_steps = max(1, round(self.image_dt / self.sim_dt))
        self.viewer_interval_steps = max(1, round(self.viewer_dt / self.sim_dt))
        self.sim_timing_log_interval_seconds = self.config.get(
            "SIM_TIMING_LOG_INTERVAL_SECONDS", 5.0
        )
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()
        drop_after_seconds = self.config.get("DROP_ELASTIC_AFTER_SECONDS", None)
        drop_start_time = time.monotonic()
        dropped_elastic_band = False
        timing_window_start = time.monotonic()
        timing_window_steps = 0
        timing_window_overruns = 0
        timing_window_elapsed_sum = 0.0
        timing_window_elapsed_max = 0.0

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()
                if (
                    drop_after_seconds is not None
                    and not dropped_elastic_band
                    and self.sim_env.elastic_band
                    and time.monotonic() - drop_start_time >= drop_after_seconds
                ):
                    self.sim_env.elastic_band.enable = False
                    dropped_elastic_band = True
                    print(f"ElasticBand enable: False after {drop_after_seconds:.1f}s")

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % self.viewer_interval_steps == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % self.reward_interval_steps == 0:
                    self.sim_env.update_reward()

                if sim_cnt % self.image_interval_steps == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                timing_window_steps += 1
                timing_window_elapsed_sum += elapsed
                timing_window_elapsed_max = max(timing_window_elapsed_max, elapsed)
                if sleep_time <= 0:
                    timing_window_overruns += 1
                self.sim_env.control_loop_overrun = sleep_time <= 0

                if self.sim_timing_log_interval_seconds > 0:
                    timing_now = time.monotonic()
                    timing_window_duration = timing_now - timing_window_start
                    if timing_window_duration >= self.sim_timing_log_interval_seconds:
                        avg_step_ms = timing_window_elapsed_sum / timing_window_steps * 1000.0
                        max_step_ms = timing_window_elapsed_max * 1000.0
                        actual_hz = timing_window_steps / timing_window_duration
                        overrun_pct = timing_window_overruns / timing_window_steps * 100.0
                        print(
                            "Sim timing: "
                            f"target_hz={1.0 / self.sim_dt:.1f}, actual_hz={actual_hz:.1f}, "
                            f"avg_step={avg_step_ms:.2f}ms, max_step={max_step_ms:.2f}ms, "
                            f"overruns={timing_window_overruns}/{timing_window_steps} "
                            f"({overrun_pct:.1f}%)"
                        )
                        timing_window_start = timing_now
                        timing_window_steps = 0
                        timing_window_overruns = 0
                        timing_window_elapsed_sum = 0.0
                        timing_window_elapsed_max = 0.0

                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if not hasattr(self, "sim_env"):
                return
            self.sim_env.print_fall_stats()
            self.sim_env.close()
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp


@configclass
class RewardCfg:
    pass


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 12288
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float | dict = 0.25  # Can be a single float or a dict mapping joint patterns to scales
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)
    
    body_height: tuple[float, float] = (0.0, 0.0)
    body_roll: tuple[float, float] = (0.0, 0.0)
    body_pitch: tuple[float, float] = (0.0, 0.0)
    body_yaw: tuple[float, float] = (0.0, 0.0)


@configclass
class CommandAxisCurriculumCfg:
    """Linear curriculum schedule for a single command axis.

    The sampled range linearly interpolates from ``start_range`` to ``end_range``
    over ``[start_iter, end_iter]`` (rsl-rl iterations). Before ``start_iter`` the
    range stays at ``start_range``; after ``end_iter`` it stays at ``end_range``.
    """

    start_range: tuple[float, float] = (0.0, 0.0)
    end_range: tuple[float, float] = (0.0, 0.0)
    start_iter: int = 20000
    end_iter: int = 60000


@configclass
class CommandCurriculumCfg:
    """Per-axis linear curriculum on the sampled command ranges.

    When ``enable=True``, each axis with a configured ``CommandAxisCurriculumCfg``
    samples from a linearly interpolated range based on the current rsl-rl
    iteration. Axes set to ``None`` fall back to the static ``commands.ranges.*``
    value (no curriculum on that axis).

    ``num_steps_per_env`` is rsl-rl's per-iteration rollout length and is used by
    the base environment to estimate the current iteration from
    ``sim_step_counter``.
    """

    enable: bool = True
    num_steps_per_env: int = 24
    lin_vel_x: CommandAxisCurriculumCfg | None = None
    lin_vel_y: CommandAxisCurriculumCfg | None = None
    ang_vel_z: CommandAxisCurriculumCfg | None = None
    body_height: CommandAxisCurriculumCfg | None = None
    body_roll: CommandAxisCurriculumCfg | None = None
    body_pitch: CommandAxisCurriculumCfg | None = None
    body_yaw: CommandAxisCurriculumCfg | None = None


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.8
    rel_heading_envs: float = 1.0
    rel_zero_vel_yaw_envs: float = 0.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()
    curriculum: CommandCurriculumCfg = CommandCurriculumCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()

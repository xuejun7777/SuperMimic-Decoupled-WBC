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

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG_HTD
from legged_lab.envs.base.base_config import CommandAxisCurriculumCfg
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    # ang_vel_x_l2 = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.05)
    # ang_vel_y_l2 = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.5)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", ".*waist.*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=0.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # flat_orientation_l2_threshold = RewTerm(func=mdp.flat_orientation_l2_threshold, weight=-0.05)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_contact_standing_stable = RewTerm(
        func=mdp.feet_contact_standing_stable,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "min_contact_time": 0.5},
    )
    feet_flat_standing = RewTerm(
        func=mdp.feet_flat_standing,
        weight=0.175,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "dead_zone_deg": 3.0,
            "std_deg": 10.0,
            "min_contact_time": 0.5,
            "contact_ramp_s": 0.25,
            # Torso-command gate parameters are in radians.
            "roll_dead_zone": 0.50,
            "pitch_dead_zone": 1.10,
            "roll_scale": 0.35,
            "pitch_scale": 0.45,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"dgp_flag": True})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )

    track_body_height_exp = RewTerm(
        func=mdp.track_body_height_exp, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "std": 0.1}, weight=1.0
    )

    track_body_roll_exp = RewTerm(
        func=mdp.track_body_roll_exp, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]), "std": 0.1}, weight=1.0
    )

    track_body_pitch_exp = RewTerm(
        func=mdp.track_body_pitch_exp, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]), "std": 0.1}, weight=1.5
    )

    track_body_yaw_exp = RewTerm(
        func=mdp.track_body_yaw_exp, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]), "std": 0.1}, weight=1.0
    )

    center_of_gravity_tracking = RewTerm(
        func=mdp.center_of_gravity_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sigma": 0.2
        }
    )
    knee_stance_width = RewTerm(
        func=mdp.knee_stance_width,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"]),
            "ideal_width": 0.27,
            "std": 0.1
        }
    )


    joint_torque_limits_waists = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"]),
            "soft_torque_limit": 0.9}
    )

    joint_torque_limits_ankles = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
            "soft_torque_limit": 0.99}
    )

    joint_torque_limits_hip_pitch = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*"]),
            "soft_torque_limit": 0.99}
    )

    # feet_parallel_when_stationary = RewTerm(
    #     func=mdp.feet_parallel_when_stationary,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #         "std": 0.05  
    #     }
    # )
    


@configclass
class G1FlatEnvCfg(BaseEnvCfg):

    reward = G1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # self.scene.height_scanner.prim_body_name = "head_link"
        self.scene.robot = G1_CFG_HTD
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.commands.rel_standing_envs = 0.4

        self.commands.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.ranges.body_height = (0.35, 0.8)
        self.commands.ranges.body_roll = (-0.7, 0.7)
        self.commands.ranges.body_pitch = (-0.52, 1.57)
        self.commands.ranges.body_yaw = (-1.57, 1.57)

        # Per-axis curriculum schedules
        self.commands.curriculum.enable = True
        self.commands.curriculum.body_height = CommandAxisCurriculumCfg(
            start_range=(0.5, 0.8), end_range=(0.35, 0.8),
            start_iter=20000, end_iter=60000,
        )
        self.commands.curriculum.body_roll = CommandAxisCurriculumCfg(
            start_range=(-0.4, 0.4), end_range=(-0.7, 0.7),
            start_iter=20000, end_iter=60000,
        )
        self.commands.curriculum.body_pitch = CommandAxisCurriculumCfg(
            start_range=(-0.3, 1.0), end_range=(-0.52, 1.57),
            start_iter=20000, end_iter=60000,
        )
        self.commands.curriculum.body_yaw = CommandAxisCurriculumCfg(
            start_range=(-0.65, 0.65), end_range=(-1.57, 1.57),
            start_iter=20000, end_iter=60000,
        )

@configclass
class G1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_flat"
    wandb_project: str = "g1_flat"


@configclass
class G1RoughEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        # The scanner needs a body prim to ride on; the G1 head is merged into
        # torso_link, so attach there.
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25
        # World-level feet are desirable on the plane but conflict with the
        # local surface normal on waves and other uneven terrain.
        self.reward.feet_flat_standing.weight = 0.0


@configclass
class G1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_rough"
    wandb_project: str = "g1_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"

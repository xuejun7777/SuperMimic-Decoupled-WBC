"""TorchScript-fused differentiable counterpart of all 31 active G1RewardCfg terms for Analytic BPTT.

Reads all weights, thresholds, joint/body index sets, and PD actuator gains directly
from the live `BaseEnv` instance so any `--set` CLI overrides stay 100% synchronized.
Note: Real simulator rollouts and evaluations ALWAYS use the untouched `env.reward_manager`.
This module is used exclusively inside World Model imagined rollouts (and verified
against `env.reward_manager` on real transitions).
"""

from __future__ import annotations

import math
from typing import Tuple  # noqa: UP035

import torch
import torch.nn as nn

from legged_lab.supermimic.rotations import (
    heading_quat,
    heading_yaw,
    projected_gravity,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    smooth_euler_xyz_from_quat,
    wrap_to_pi,
)
from legged_lab.supermimic.usd_fk import UsdArticulationFK


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable L2 norm with finite gradient at x = 0."""
    return torch.sqrt((x * x).sum(dim=dim) + 1.0e-12)


def _pseudo_huber_abs(x: torch.Tensor) -> torch.Tensor:
    """Smooth |x| approximation (error <= 1e-4) eliminating the kink at 0."""
    return torch.sqrt(x * x + 1.0e-8) - 1.0e-4


def _update_cmd_jit(
    s: torch.Tensor,
    base_command: torch.Tensor,
    heading_target: torch.Tensor,
    is_heading_env: torch.Tensor,
    is_standing_env: torch.Tensor,
    heading_command: bool,
    heading_stiffness: float,
    wz_min: float,
    wz_max: float,
) -> torch.Tensor:
    cmd = base_command
    if heading_command:
        yaw = heading_yaw(s[..., 3:7])
        heading_err = wrap_to_pi(heading_target - yaw)
        wz_heading = torch.clamp(heading_stiffness * heading_err, min=wz_min, max=wz_max)
        wz = torch.where(is_heading_env > 0.5, wz_heading, cmd[..., 2])
        cmd = torch.cat([cmd[..., :2], wz.unsqueeze(-1), cmd[..., 3:]], dim=-1)
    stand_mask = (is_standing_env > 0.5).unsqueeze(-1)
    vel_part = torch.where(stand_mask, torch.zeros_like(cmd[..., :3]), cmd[..., :3])
    return torch.cat([vel_part, cmd[..., 3:]], dim=-1)


def _build_actor_obs_jit(
    s: torch.Tensor,
    command: torch.Tensor,
    prev_action: torch.Tensor,
    foot_contact_prob: torch.Tensor,
    policy_usd_ids: torch.Tensor,
    default_pol_pos: torch.Tensor,
    obs_scale_vec: torch.Tensor,
    noise_scale_vec: torch.Tensor,
    clip_obs: float,
    add_noise: bool,
    ste_contacts: bool,
) -> torch.Tensor:
    q_root = s[..., 3:7]
    qj_usd = s[..., 7:36]
    w_b = s[..., 39:42]
    qd_usd = s[..., 42:71]
    grav_b = projected_gravity(q_root)

    q_pol = qj_usd.index_select(-1, policy_usd_ids) - default_pol_pos
    qd_pol = qd_usd.index_select(-1, policy_usd_ids)

    if ste_contacts:
        hard_c = (foot_contact_prob > 0.5).to(foot_contact_prob.dtype)
        feet_c = hard_c - foot_contact_prob.detach() + foot_contact_prob
    else:
        feet_c = foot_contact_prob

    raw_obs = torch.cat([w_b, grav_b, command, q_pol, qd_pol, prev_action, feet_c], dim=-1)
    obs = raw_obs * obs_scale_vec
    if add_noise:
        obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * noise_scale_vec
    return torch.clamp(obs, -clip_obs, clip_obs)


def _compute_diff_rewards_jit(
    s_prev: torch.Tensor,
    s_next: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_com_pos_w: torch.Tensor,
    body_com_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    arm_target: torch.Tensor,
    command: torch.Tensor,
    is_standing_env: torch.Tensor,
    xi: torch.Tensor,
    foot_contact_prob: torch.Tensor,
    contact_time_next: torch.Tensor,
    air_time_next: torch.Tensor,
    r_impact_pred: torch.Tensor,
    w_dt_vec: torch.Tensor,
    soft_pos_limits_usd: torch.Tensor,
    default_joint_pos_usd: torch.Tensor,
    default_pol_pos: torch.Tensor,
    action_scale: torch.Tensor,
    inverse_joint_ids: torch.Tensor,
    kp_usd: torch.Tensor,
    kd_usd: torch.Tensor,
    effort_limits_usd: torch.Tensor,
    nominal_body_masses: torch.Tensor,
    torso_one_hot: torch.Tensor,
    jids_dof_acc: torch.Tensor,
    jids_dev_hip: torch.Tensor,
    jids_dev_waists: torch.Tensor,
    jids_dev_legs: torch.Tensor,
    jids_tau_waists: torch.Tensor,
    jids_tau_ankles: torch.Tensor,
    jids_tau_hip_pitch: torch.Tensor,
    bids_feet: torch.Tensor,
    bids_knees: torch.Tensor,
    torso_id: int,
    pelvis_id: int,
    torso_orient_id: int,
    dt: float,
    clip_actions: float,
    lin_vel_std: float,
    ang_vel_std: float,
    feet_air_thresh: float,
    stand_min_ct: float,
    ff_dead_zone: float,
    ff_std: float,
    ff_min_ct: float,
    ff_ramp_s: float,
    ff_roll_dz: float,
    ff_pitch_dz: float,
    ff_roll_scale: float,
    ff_pitch_scale: float,
    feet_near_thresh: float,
    height_std: float,
    roll_std: float,
    pitch_std: float,
    yaw_std: float,
    cog_sigma: float,
    knee_ideal: float,
    knee_std: float,
    tau_soft_waists: float,
    tau_soft_ankles: float,
    tau_soft_hip_pitch: float,
    fall_z_threshold: float,
    fall_z_temp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: UP006
    qj_prev = s_prev[..., 7:36]
    qd_prev = s_prev[..., 42:71]
    q_next = s_next[..., 3:7]
    qj_next = s_next[..., 7:36]
    w_b_next = s_next[..., 39:42]
    qd_next = s_next[..., 42:71]

    root_com_vel_w = body_com_lin_vel_w[..., 0, :]
    root_com_vel_b = quat_apply_inverse(q_next, root_com_vel_w)
    root_ang_vel_w = body_ang_vel_w[..., 0, :]
    proj_grav_b = projected_gravity(q_next)
    pelvis_yaw_q = heading_quat(q_next)

    # ---- Group I: 17 Pure Kinematic / State / Action Terms ----
    vel_yaw = quat_apply_inverse(pelvis_yaw_q, root_com_vel_w)
    lin_vel_err = ((command[..., :2] - vel_yaw[..., :2]) ** 2).sum(dim=-1)
    t0 = torch.exp(-lin_vel_err / (lin_vel_std * lin_vel_std))

    ang_vel_err = (command[..., 2] - root_ang_vel_w[..., 2]) ** 2
    t1 = torch.exp(-ang_vel_err / (ang_vel_std * ang_vel_std))

    t2 = root_com_vel_b[..., 2] ** 2
    t3 = (w_b_next[..., :2] ** 2).sum(dim=-1)
    t4 = (proj_grav_b[..., :2] ** 2).sum(dim=-1)
    t5 = ((action - prev_action) ** 2).sum(dim=-1)

    lower = soft_pos_limits_usd[:, 0]
    upper = soft_pos_limits_usd[:, 1]
    t6 = (torch.relu(lower - qj_next) + torch.relu(qj_next - upper)).sum(dim=-1)

    q_dev = _pseudo_huber_abs(qj_next - default_joint_pos_usd)
    t7 = q_dev.index_select(-1, jids_dev_hip).sum(dim=-1)
    t8 = q_dev.index_select(-1, jids_dev_waists).sum(dim=-1)
    t9 = q_dev.index_select(-1, jids_dev_legs).sum(dim=-1)

    feet_pos = body_pos_w.index_select(-2, bids_feet)
    feet_dist = _safe_norm(feet_pos[..., 0, :] - feet_pos[..., 1, :], -1)
    t10 = torch.relu(feet_near_thresh - feet_dist)

    torso_z = body_pos_w[..., torso_id, 2]
    t11 = torch.exp(-((command[..., 3] - torso_z) ** 2) / (height_std * height_std))

    q_torso_w = body_quat_w[..., torso_orient_id, :]
    q_pelvis_w = body_quat_w[..., pelvis_id, :]
    q_pelvis_yaw = heading_quat(q_pelvis_w)
    q_torso_in_pelvis_yaw = quat_mul(quat_inv(q_pelvis_yaw), q_torso_w)
    roll_approx, pitch_approx, _ = smooth_euler_xyz_from_quat(q_torso_in_pelvis_yaw)
    yaw_diff = wrap_to_pi(heading_yaw(q_torso_w) - heading_yaw(q_pelvis_w))

    t12 = torch.exp(-((roll_approx - command[..., 4]) ** 2) / (roll_std * roll_std))
    t13 = torch.exp(-((command[..., 5] - pitch_approx) ** 2) / (pitch_std * pitch_std))
    t14 = torch.exp(-((yaw_diff - command[..., 6]) ** 2) / (yaw_std * yaw_std))

    torso_dm = xi[..., 3:4]
    masses = nominal_body_masses + torso_dm * torso_one_hot
    total_mass = masses.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cog_xy = (body_com_pos_w[..., :2] * masses.unsqueeze(-1)).sum(dim=-2) / total_mass
    feet_xy = 0.5 * (feet_pos[..., 0, :2] + feet_pos[..., 1, :2])
    cog_dist_sq = ((cog_xy - feet_xy) ** 2).sum(dim=-1)
    t15 = torch.exp(-cog_dist_sq / (cog_sigma * cog_sigma))

    knee_pos = body_pos_w.index_select(-2, bids_knees)
    knee_dist = _safe_norm(knee_pos[..., 0, :2] - knee_pos[..., 1, :2], -1)
    t16 = torch.exp(-((knee_dist - knee_ideal) ** 2) / (knee_std * knee_std))

    # ---- Group II: 5 Joint Acceleration & Closed-Form PD Motor Torque Terms ----
    qacc_pol = (qd_next.index_select(-1, jids_dof_acc) - qd_prev.index_select(-1, jids_dof_acc)) / dt
    t17 = (qacc_pol * qacc_pol).sum(dim=-1)

    clipped_act = torch.clamp(action, -clip_actions, clip_actions)
    q_tgt_pol = default_pol_pos + clipped_act * action_scale
    q_tgt_custom = torch.cat([q_tgt_pol, arm_target], dim=-1)
    q_tgt_usd = q_tgt_custom.index_select(-1, inverse_joint_ids)

    qj_sub3 = 0.25 * qj_prev + 0.75 * qj_next
    qd_sub3 = 0.25 * qd_prev + 0.75 * qd_next
    computed_tau = kp_usd * (q_tgt_usd - qj_sub3) - kd_usd * qd_sub3
    applied_tau = torch.clamp(computed_tau, -effort_limits_usd, effort_limits_usd)
    t18 = _safe_norm(torch.abs(applied_tau * qd_next), -1)

    norm_tau = _pseudo_huber_abs(computed_tau) / effort_limits_usd
    t19 = torch.relu(norm_tau.index_select(-1, jids_tau_waists) - tau_soft_waists).sum(dim=-1)
    t20 = torch.relu(norm_tau.index_select(-1, jids_tau_ankles) - tau_soft_ankles).sum(dim=-1)
    t21 = torch.relu(norm_tau.index_select(-1, jids_tau_hip_pitch) - tau_soft_hip_pitch).sum(dim=-1)

    # ---- Group III: 5 Multilinear Contact-Timer Terms + 1 Soft Termination ----
    p_L = foot_contact_prob[..., 0]
    p_R = foot_contact_prob[..., 1]
    t22 = (1.0 - p_L) * (1.0 - p_R)

    feet_vel_xy = body_com_lin_vel_w.index_select(-2, bids_feet)[..., :2]
    t23 = (_safe_norm(feet_vel_xy, -1) * foot_contact_prob).sum(dim=-1)

    cmd_speed = _safe_norm(command[..., :2], -1) + torch.abs(command[..., 2])
    moving_gate = (cmd_speed > 0.1).to(s_next.dtype)
    zero_cmd_gate = 1.0 - moving_gate

    p_single = p_L * (1.0 - p_R) + (1.0 - p_L) * p_R
    t_mode = foot_contact_prob * contact_time_next + (1.0 - foot_contact_prob) * air_time_next
    t24 = torch.clamp(torch.min(t_mode, dim=-1)[0], max=feet_air_thresh) * p_single * moving_gate

    stable_gate = torch.sigmoid((contact_time_next - stand_min_ct) / 0.05)
    min_ct = torch.min(contact_time_next, dim=-1)[0]
    t25 = zero_cmd_gate * stable_gate[..., 0] * stable_gate[..., 1] * torch.tanh(min_ct / stand_min_ct)

    foot_quat_w = body_quat_w.index_select(-2, bids_feet)
    g_foot = projected_gravity(foot_quat_w)
    foot_tilt = torch.atan2(_safe_norm(g_foot[..., :2], -1), -g_foot[..., 2])
    tilt_excess = torch.relu(foot_tilt - ff_dead_zone)
    flatness = torch.exp(-((tilt_excess / ff_std) ** 2)).mean(dim=-1)
    contact_gate = torch.clamp((min_ct - ff_min_ct) / ff_ramp_s, 0.0, 1.0)
    roll_exc = torch.relu(torch.abs(command[..., 4]) - ff_roll_dz)
    pitch_exc = torch.relu(torch.abs(command[..., 5]) - ff_pitch_dz)
    pose_gate = torch.exp(-((roll_exc / ff_roll_scale) ** 2) - ((pitch_exc / ff_pitch_scale) ** 2))
    t26 = is_standing_env.to(s_next.dtype) * contact_gate * pose_gate * flatness

    d_term = torch.sigmoid((fall_z_threshold - s_next[..., 2]) / fall_z_temp)
    t27 = d_term

    terms_stack = torch.stack(
        [
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
        ],
        dim=-1,
    )
    weighted_terms = terms_stack * w_dt_vec
    total = weighted_terms.sum(dim=-1) + r_impact_pred
    return total, d_term, weighted_terms


class DifferentiableG1Rewards(nn.Module):
    """Analytically differentiable G1 reward suite + observation constructor for imagined BPTT."""

    IMPACT_TERM_NAMES = ("undesired_contacts", "feet_force", "feet_stumble")
    ANALYTIC_TERM_NAMES = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation_l2",
        "action_rate_l2",
        "dof_pos_limits",
        "joint_deviation_hip",
        "joint_deviation_waists",
        "joint_deviation_legs",
        "feet_too_near",
        "track_body_height_exp",
        "track_body_roll_exp",
        "track_body_pitch_exp",
        "track_body_yaw_exp",
        "center_of_gravity_tracking",
        "knee_stance_width",
        "dof_acc_l2",
        "energy",
        "joint_torque_limits_waists",
        "joint_torque_limits_ankles",
        "joint_torque_limits_hip_pitch",
        "fly",
        "feet_slide",
        "feet_air_time",
        "feet_contact_standing_stable",
        "feet_flat_standing",
        "termination_penalty",
    )

    def __init__(self, env, fk: UsdArticulationFK):
        super().__init__()
        from isaaclab.managers import SceneEntityCfg

        self.fk = fk
        self.dt = float(env.step_dt)
        self.num_actions = int(env.num_actions)  # 15
        self.clip_actions = float(env.clip_actions)
        self.clip_obs = float(env.clip_obs)
        self.obs_scales = env.obs_scales

        robot = env.scene["robot"]
        device = env.device
        dtype = torch.float32

        # Joint mapping between policy/AMASS custom order (15 policy + 14 arm) and USD joint order (29)
        custom_joint_ids = torch.as_tensor(env.custom_joint_ids, dtype=torch.long, device=device)
        inverse_joint_ids = torch.as_tensor(env.inverse_joint_ids, dtype=torch.long, device=device)
        self.register_buffer("custom_joint_ids", custom_joint_ids)
        self.register_buffer("policy_usd_ids", custom_joint_ids[: self.num_actions])
        self.register_buffer("arm_usd_ids", custom_joint_ids[self.num_actions : self.num_actions + 14])
        self.register_buffer("inverse_joint_ids", inverse_joint_ids)

        # Default joint positions & action scale
        self.register_buffer("default_joint_pos_usd", robot.data.default_joint_pos[0].to(device=device, dtype=dtype).clone())
        self.register_buffer("custom_default_joint_pos", env.custom_default_joint_pos[0].to(device=device, dtype=dtype).clone())
        self.register_buffer("default_pol_pos", env.custom_default_joint_pos[0, : self.num_actions].to(device=device, dtype=dtype).clone())
        if torch.is_tensor(env.action_scale):
            act_scale = env.action_scale[: self.num_actions].to(device=device, dtype=dtype).clone()
        else:
            act_scale = torch.full((self.num_actions,), float(env.action_scale), device=device, dtype=dtype)
        self.register_buffer("action_scale", act_scale)

        # Precompute 60-D observation scale vector for single fused multiply
        sc = self.obs_scales
        cmd_sc = sc.commands.to(device=device, dtype=dtype).reshape(-1) if torch.is_tensor(sc.commands) else torch.full((7,), float(sc.commands), device=device, dtype=dtype)
        obs_scale_vec = torch.cat(
            [
                torch.full((3,), float(sc.ang_vel), device=device, dtype=dtype),
                torch.full((3,), float(sc.projected_gravity), device=device, dtype=dtype),
                cmd_sc,
                torch.full((self.num_actions,), float(sc.joint_pos), device=device, dtype=dtype),
                torch.full((self.num_actions,), float(sc.joint_vel), device=device, dtype=dtype),
                torch.full((self.num_actions,), float(sc.actions), device=device, dtype=dtype),
                torch.ones((2,), device=device, dtype=dtype),
            ],
            dim=0,
        )
        self.register_buffer("obs_scale_vec", obs_scale_vec)

        # Actuator PD gains & effort limits in USD joint order
        self.register_buffer("kp_usd", robot.data.joint_stiffness[0].to(device=device, dtype=dtype).clone())
        self.register_buffer("kd_usd", robot.data.joint_damping[0].to(device=device, dtype=dtype).clone())
        self.register_buffer("effort_limits_usd", robot.data.joint_effort_limits[0].to(device=device, dtype=dtype).clone())
        self.register_buffer("soft_pos_limits_usd", robot.data.soft_joint_pos_limits[0].to(device=device, dtype=dtype).clone())

        # Observation noise vector (matches BaseEnv.init_obs_buffer)
        if getattr(env, "add_noise", False) and hasattr(env, "noise_scale_vec"):
            self.register_buffer("noise_scale_vec", env.noise_scale_vec.to(device=device, dtype=dtype).clone())
        else:
            self.register_buffer("noise_scale_vec", torch.zeros(60, device=device, dtype=dtype))

        # Resolve body and joint index groups directly from G1RewardCfg against env.scene
        rcfg = env.cfg.reward

        def _jids(term_cfg) -> torch.Tensor:
            acfg: SceneEntityCfg = term_cfg.params["asset_cfg"]
            copy_cfg = SceneEntityCfg(acfg.name, joint_names=acfg.joint_names, preserve_order=acfg.preserve_order)
            copy_cfg.resolve(env.scene)
            if isinstance(copy_cfg.joint_ids, slice):
                return torch.arange(robot.num_joints, device=device, dtype=torch.long)
            return torch.as_tensor(copy_cfg.joint_ids, device=device, dtype=torch.long)

        def _bids(term_cfg, key: str = "asset_cfg") -> torch.Tensor:
            acfg: SceneEntityCfg = term_cfg.params[key]
            copy_cfg = SceneEntityCfg(acfg.name, body_names=acfg.body_names, preserve_order=acfg.preserve_order)
            copy_cfg.resolve(env.scene)
            if isinstance(copy_cfg.body_ids, slice):
                return torch.arange(robot.num_bodies, device=device, dtype=torch.long)
            return torch.as_tensor(copy_cfg.body_ids, device=device, dtype=torch.long)

        self.register_buffer("jids_dof_acc", _jids(rcfg.dof_acc_l2))
        self.register_buffer("jids_dev_hip", _jids(rcfg.joint_deviation_hip))
        self.register_buffer("jids_dev_waists", _jids(rcfg.joint_deviation_waists))
        self.register_buffer("jids_dev_legs", _jids(rcfg.joint_deviation_legs))
        self.register_buffer("jids_tau_waists", _jids(rcfg.joint_torque_limits_waists))
        self.register_buffer("jids_tau_ankles", _jids(rcfg.joint_torque_limits_ankles))
        self.register_buffer("jids_tau_hip_pitch", _jids(rcfg.joint_torque_limits_hip_pitch))

        self.register_buffer("bids_feet", _bids(rcfg.feet_too_near))
        self.register_buffer("bids_knees", _bids(rcfg.knee_stance_width))
        self.torso_id = int(_bids(rcfg.track_body_height_exp)[0].item())
        pelvis_torso_ids = _bids(rcfg.track_body_roll_exp)
        self.pelvis_id = int(pelvis_torso_ids[0].item())
        self.torso_orient_id = int(pelvis_torso_ids[1].item())

        torso_one_hot = torch.zeros(robot.num_bodies, dtype=dtype, device=device)
        torso_one_hot[self.torso_id] = 1.0
        self.register_buffer("torso_one_hot", torso_one_hot)

        # Store exact weights & precompute w * dt vector in ANALYTIC_TERM_NAMES order
        self.weights: dict[str, float] = {}
        for attr in dir(rcfg):
            val = getattr(rcfg, attr)
            if hasattr(val, "weight") and hasattr(val, "func"):
                self.weights[attr] = float(val.weight)

        w_dt_list = [self.weights.get(name, 0.0) * self.dt for name in self.ANALYTIC_TERM_NAMES]
        self.register_buffer("w_dt_vec", torch.tensor(w_dt_list, dtype=dtype, device=device))

        self.lin_vel_std = float(rcfg.track_lin_vel_xy_exp.params["std"])
        self.ang_vel_std = float(rcfg.track_ang_vel_z_exp.params["std"])
        self.feet_air_thresh = float(rcfg.feet_air_time.params["threshold"])
        self.stand_min_ct = float(rcfg.feet_contact_standing_stable.params["min_contact_time"])
        ff_p = rcfg.feet_flat_standing.params
        self.ff_dead_zone = math.radians(float(ff_p["dead_zone_deg"]))
        self.ff_std = math.radians(float(ff_p["std_deg"]))
        self.ff_min_ct = float(ff_p["min_contact_time"])
        self.ff_ramp_s = float(ff_p["contact_ramp_s"])
        self.ff_roll_dz = float(ff_p["roll_dead_zone"])
        self.ff_pitch_dz = float(ff_p["pitch_dead_zone"])
        self.ff_roll_scale = float(ff_p["roll_scale"])
        self.ff_pitch_scale = float(ff_p["pitch_scale"])
        self.feet_near_thresh = float(rcfg.feet_too_near.params["threshold"])
        self.height_std = float(rcfg.track_body_height_exp.params["std"])
        self.roll_std = float(rcfg.track_body_roll_exp.params["std"])
        self.pitch_std = float(rcfg.track_body_pitch_exp.params["std"])
        self.yaw_std = float(rcfg.track_body_yaw_exp.params["std"])
        self.cog_sigma = float(rcfg.center_of_gravity_tracking.params["sigma"])
        self.knee_ideal = float(rcfg.knee_stance_width.params["ideal_width"])
        self.knee_std = float(rcfg.knee_stance_width.params["std"])
        self.tau_soft_waists = float(rcfg.joint_torque_limits_waists.params["soft_torque_limit"])
        self.tau_soft_ankles = float(rcfg.joint_torque_limits_ankles.params["soft_torque_limit"])
        self.tau_soft_hip_pitch = float(rcfg.joint_torque_limits_hip_pitch.params["soft_torque_limit"])

        # Command post-processing config (for differentiable heading feedback in imagination)
        self.heading_command = bool(env.cfg.commands.heading_command)
        self.heading_stiffness = float(env.cfg.commands.heading_control_stiffness)
        self.ang_vel_range = tuple(float(x) for x in env.cfg.commands.ranges.ang_vel_z)

        # Fall threshold & soft temperature (root_z < 0.30m; min squat command is 0.35m)
        self.fall_z_threshold = 0.30
        self.fall_z_temp = 0.01

    def update_command_in_imagination(
        self,
        s: torch.Tensor,
        base_command: torch.Tensor,
        heading_target: torch.Tensor,
        is_heading_env: torch.Tensor,
        is_standing_env: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiably update command[:, 2] for heading-controlled envs, matching UniformLevelVelocityCommand."""
        return _update_cmd_jit(
            s,
            base_command,
            heading_target,
            is_heading_env,
            is_standing_env,
            self.heading_command,
            self.heading_stiffness,
            self.ang_vel_range[0],
            self.ang_vel_range[1],
        )

    def build_actor_obs(
        self,
        s: torch.Tensor,
        command: torch.Tensor,
        prev_action: torch.Tensor,
        foot_contact_prob: torch.Tensor,
        add_noise: bool = False,
        ste_contacts: bool = True,
    ) -> torch.Tensor:
        """Construct the exact 60-dim Teacher Actor observation from predicted state `s`."""
        return _build_actor_obs_jit(
            s,
            command,
            prev_action,
            foot_contact_prob,
            self.policy_usd_ids,
            self.default_pol_pos,
            self.obs_scale_vec,
            self.noise_scale_vec,
            self.clip_obs,
            add_noise,
            ste_contacts,
        )

    def soft_termination(self, s_next: torch.Tensor) -> torch.Tensor:
        """Differentiable fall probability P(root_z < 0.30) with temperature 0.01m."""
        z_root = s_next[..., 2]
        return torch.sigmoid((self.fall_z_threshold - z_root) / self.fall_z_temp)

    def forward(
        self,
        s_prev: torch.Tensor,
        s_next: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        arm_target: torch.Tensor,
        command: torch.Tensor,
        is_standing_env: torch.Tensor,
        xi: torch.Tensor,
        foot_contact_prob: torch.Tensor,
        contact_time_next: torch.Tensor,
        air_time_next: torch.Tensor,
        r_impact_pred: torch.Tensor,
        return_diag: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total step reward `r_{t+1}` (scaled by dt=0.02), soft termination `d_{t+1}`, and optional per-term dict."""
        batch_shape = s_next.shape[:-1]
        s_next_2d = s_next.reshape(-1, 71)
        pos, quat, com_pos, _, ang_vel, com_lin_vel = self.fk.forward_vel_2d(
            s_next_2d[:, 0:3],
            s_next_2d[:, 3:7],
            s_next_2d[:, 7:36],
            s_next_2d[:, 36:39],
            s_next_2d[:, 39:42],
            s_next_2d[:, 42:71],
        )
        nb = self.fk.num_bodies
        body_pos_w = pos.reshape(*batch_shape, nb, 3)
        body_quat_w = quat.reshape(*batch_shape, nb, 4)
        body_com_pos_w = com_pos.reshape(*batch_shape, nb, 3)
        body_com_lin_vel_w = com_lin_vel.reshape(*batch_shape, nb, 3)
        body_ang_vel_w = ang_vel.reshape(*batch_shape, nb, 3)

        total, d_term, weighted_terms = _compute_diff_rewards_jit(
            s_prev,
            s_next,
            body_pos_w,
            body_quat_w,
            body_com_pos_w,
            body_com_lin_vel_w,
            body_ang_vel_w,
            action,
            prev_action,
            arm_target,
            command,
            is_standing_env,
            xi,
            foot_contact_prob,
            contact_time_next,
            air_time_next,
            r_impact_pred,
            self.w_dt_vec,
            self.soft_pos_limits_usd,
            self.default_joint_pos_usd,
            self.default_pol_pos,
            self.action_scale,
            self.inverse_joint_ids,
            self.kp_usd,
            self.kd_usd,
            self.effort_limits_usd,
            self.fk.nominal_body_masses,
            self.torso_one_hot,
            self.jids_dof_acc,
            self.jids_dev_hip,
            self.jids_dev_waists,
            self.jids_dev_legs,
            self.jids_tau_waists,
            self.jids_tau_ankles,
            self.jids_tau_hip_pitch,
            self.bids_feet,
            self.bids_knees,
            self.torso_id,
            self.pelvis_id,
            self.torso_orient_id,
            self.dt,
            self.clip_actions,
            self.lin_vel_std,
            self.ang_vel_std,
            self.feet_air_thresh,
            self.stand_min_ct,
            self.ff_dead_zone,
            self.ff_std,
            self.ff_min_ct,
            self.ff_ramp_s,
            self.ff_roll_dz,
            self.ff_pitch_dz,
            self.ff_roll_scale,
            self.ff_pitch_scale,
            self.feet_near_thresh,
            self.height_std,
            self.roll_std,
            self.pitch_std,
            self.yaw_std,
            self.cog_sigma,
            self.knee_ideal,
            self.knee_std,
            self.tau_soft_waists,
            self.tau_soft_ankles,
            self.tau_soft_hip_pitch,
            self.fall_z_threshold,
            self.fall_z_temp,
        )

        if not return_diag:
            return total, d_term, {}

        wt_det = weighted_terms.detach()
        weighted_diag: dict[str, torch.Tensor] = {
            name: wt_det[..., idx] for idx, name in enumerate(self.ANALYTIC_TERM_NAMES)
        }
        weighted_diag["impact_head"] = r_impact_pred.detach()
        return total, d_term, weighted_diag

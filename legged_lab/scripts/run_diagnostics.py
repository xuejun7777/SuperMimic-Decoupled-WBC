"""Mechanistic Diagnostic Suite for Reference-Free SuperMimic vs Zero-Order PPO on Unitree G1 WBC.

Runs four quantitative diagnostic experiments on real Isaac Sim PhysX + trained `G1WorldModel`:
  D1. Multi-step open-loop World Model physical fidelity (h = 1, 4, 8, 16, 24 steps vs PhysX).
  D2. First-order Analytic BPTT vs Zero-Order Policy-Gradient directional SNR & multi-joint
      action Jacobian coordination (including extreme forward bow past `waist_pitch` limit).
  D3. Counterfactual command relabeling fidelity (command-free physics invariance test).
  D4. Open-loop yaw-rate (`wz=0`) integration drift vs native closed-loop heading hold
      (`heading_target=0`) under asymmetric AMASS bimanual arm-swing angular momentum.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run mechanistic diagnostics for Reference-Free SuperMimic.")
parser.add_argument("--task", type=str, default="g1_flat")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--sm_dir", type=str, required=True, help="SuperMimic run directory containing supermimic_last.pt.")
parser.add_argument("--cmu_ppo_ckpt", type=str, default="example/model_000.pt", help="Official 240k-iter PPO checkpoint.")
parser.add_argument("--out_json", type=str, required=True, help="Output JSON path for diagnostic results.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

from legged_lab.envs import *  # noqa: F401, F403
import legged_lab.utils.amass as amass_module
from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.nets import ActorCfg, CriticCfg, PrivilegedCriticMLP, TeacherActorMLP
from legged_lab.supermimic.rotations import quat_inv, quat_mul, quat_normalize, rotvec_from_quat
from legged_lab.supermimic.usd_fk import UsdArticulationFK
from legged_lab.supermimic.world_model import G1WorldModel, WorldModelCfg
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import apply_trained_actuator_gains


def _flat_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    vecs = []
    for p in params:
        if p.grad is not None:
            vecs.append(p.grad.detach().flatten().float())
        else:
            vecs.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
    return torch.cat(vecs, dim=0)


def _seed_all(env: SuperMimicG1Env, seed: int) -> None:
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if getattr(amass_module, "_global_amass_loader", None) is not None:
        amass_module._global_amass_loader._rng = np.random.default_rng(seed)


@torch.no_grad()
def run_d1_world_model_horizon_fidelity(
    env: SuperMimicG1Env,
    wm: G1WorldModel,
    actor: TeacherActorMLP,
    diff_rewards: DifferentiableG1Rewards,
    horizon: int = 24,
) -> dict:
    """Compare h-step open-loop G1WorldModel predictions against real PhysX rollouts."""
    _seed_all(env, args_cli.seed)
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset(all_ids)
    obs, _ = env.get_observations()

    # Warm up 80 steps so robots are in active locomotion & varied 7D body postures
    for _ in range(80):
        obs, _, _, _ = env.step(actor(obs))

    s_init = env.read_full_state().clone()
    ct_init, at_init = env.read_contact_timers()
    prev_act_init = env.action_buffer._circular_buffer.buffer[:, -1, :].clone()
    cmd_base = env.command_generator.command.clone()
    heading_tgt = env.command_generator.heading_target.clone()
    is_heading = env.command_generator.is_heading_env.float().clone()
    is_standing = env.command_generator.is_standing_env.float().clone()
    xi = env.xi.clone()
    xi_norm = wm.spec.normalize_xi(xi)

    # Step real PhysX for `horizon` steps while recording exact executed inputs & ground-truth outputs
    real_states = []
    real_contacts = []
    real_rewards = []
    exec_actions = []
    exec_arms = []
    exec_cmds = []
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(horizon):
        snap = env.current_step_inputs()
        act = actor(obs)
        exec_actions.append(act.clone())
        exec_arms.append(snap["arm_tgt"].clone())
        obs, rew, dones, _ = env.step(act)
        exec_cmds.append(env.command_generator.command.clone())
        real_states.append(env.last_pre_reset_state.clone())
        real_contacts.append(env.last_pre_reset_feet_contact.clone())
        real_rewards.append(rew.clone())
        alive = alive & (~(dones > 0))

    # Open-loop rollout in G1WorldModel from `s_init` using exact `exec_actions` & `exec_arms`
    s_hat = s_init
    feat_hat = wm.spec.canonical(s_hat)
    ct = ct_init
    at = at_init
    prev_act = prev_act_init
    dt = float(env.step_dt)

    horizons_to_log = {1, 4, 8, 16, 24}
    by_h: dict[str, dict[str, float]] = {}
    pred_rews_all = []
    true_rews_all = []

    for tau in range(horizon):
        s_next, feat_next, c_logits, r_imp = wm.step_with_features(
            s=s_hat,
            policy_action=exec_actions[tau],
            arm_target=exec_arms[tau],
            xi_norm=xi_norm,
            feat_s=feat_hat,
        )
        p_foot_next = torch.sigmoid(c_logits)
        ct_next = p_foot_next * (ct + dt)
        at_next = (1.0 - p_foot_next) * (at + dt)
        cmd_next = diff_rewards.update_command_in_imagination(
            s_next, cmd_base, heading_tgt, is_heading, is_standing
        )
        r_hat, _, _ = diff_rewards(
            s_prev=s_hat,
            s_next=s_next,
            action=exec_actions[tau],
            prev_action=prev_act,
            arm_target=exec_arms[tau],
            command=cmd_next,
            is_standing_env=is_standing,
            xi=xi,
            foot_contact_prob=p_foot_next,
            contact_time_next=ct_next,
            air_time_next=at_next,
            r_impact_pred=r_imp,
            return_diag=False,
        )

        m = alive
        s_true = real_states[tau][m]
        s_p = s_next[m]
        pred_rews_all.append(r_hat[m])
        true_rews_all.append(real_rewards[tau][m])

        h_step = tau + 1
        if h_step in horizons_to_log:
            pz_cm = float(((s_p[:, 2] - s_true[:, 2]).abs().mean().item()) * 100.0)
            q_err = rotvec_from_quat(quat_mul(quat_inv(quat_normalize(s_true[:, 3:7])), quat_normalize(s_p[:, 3:7])))
            ori_deg = float(torch.norm(q_err, dim=-1).mean().item() * (180.0 / math.pi))
            vel_mps = float(torch.norm(s_p[:, 36:39] - s_true[:, 36:39], dim=-1).mean().item())
            jpos_deg = float((s_p[:, 7:36] - s_true[:, 7:36]).abs().mean().item() * (180.0 / math.pi))
            c_pred = (p_foot_next[m] > 0.5).float()
            c_acc = float((c_pred == real_contacts[tau][m]).float().mean().item() * 100.0)
            by_h[f"h={h_step}"] = {
                "time_s": round(h_step * dt, 2),
                "pelvis_z_mae_cm": round(pz_cm, 3),
                "orient_err_deg": round(ori_deg, 3),
                "lin_vel_err_mps": round(vel_mps, 4),
                "joint_pos_mae_deg": round(jpos_deg, 3),
                "contact_acc_pct": round(c_acc, 2),
            }

        s_hat, feat_hat = s_next, feat_next
        ct, at = ct_next, at_next
        prev_act = exec_actions[tau]

    pr = torch.cat(pred_rews_all)
    tr = torch.cat(true_rews_all)
    pr_c = pr - pr.mean()
    tr_c = tr - tr.mean()
    corr = float((pr_c * tr_c).sum().item() / (torch.norm(pr_c) * torch.norm(tr_c) + 1e-8).item())
    return {
        "alive_envs": int(alive.sum().item()),
        "step_reward_pearson_r": round(corr, 4),
        "horizons": by_h,
    }


def run_d2_gradient_snr_and_jacobian_coordination(
    env: SuperMimicG1Env,
    wm: G1WorldModel,
    actor: TeacherActorMLP,
    critic: PrivilegedCriticMLP,
    diff_rewards: DifferentiableG1Rewards,
    horizon: int = 24,
    sigma: float = 0.20,
) -> dict:
    """Compare First-Order Analytic BPTT vs Zero-Order Score-Function PG directional SNR & joint Jacobians."""
    _seed_all(env, args_cli.seed + 10)
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset(all_ids)
    obs, _ = env.get_observations()
    with torch.no_grad():
        for _ in range(50):
            obs, _, _, _ = env.step(actor(obs))

        snap0 = env.current_step_inputs()
        N = env.num_envs
        env_ar = torch.arange(N, device=env.device)
        arm_seq = torch.stack(
            [env.amass[env_ar, (snap0["amass_step"] + tau) % env.sequence_length] for tau in range(horizon)],
            dim=0,
        )  # (H, N, 14)

    params = [p for p in actor.parameters() if p.requires_grad]
    dt = float(env.step_dt)
    gamma = 0.99

    def _bptt_grad_on_indices(idx: torch.Tensor, noise_seed: int | None = None, cmd_override: torch.Tensor | None = None):
        actor.zero_grad(set_to_none=True)
        s_tau = snap0["s"][idx].detach()
        feat_tau = wm.spec.canonical(s_tau)
        prev_act = snap0["prev_action"][idx].detach()
        p_foot = snap0["feet_contact"][idx].detach()
        ct = snap0["contact_time"][idx].detach()
        at = snap0["air_time"][idx].detach()
        base_cmd = cmd_override[idx] if cmd_override is not None else snap0["cmd"][idx].detach()
        h_tgt = snap0["heading_target"][idx].detach()
        is_h = torch.zeros_like(snap0["is_heading"][idx]) if cmd_override is not None else snap0["is_heading"][idx].detach()
        is_st = torch.ones_like(snap0["is_standing"][idx]) if cmd_override is not None else snap0["is_standing"][idx].detach()
        xi = env.xi[idx].detach()
        xi_norm = wm.spec.normalize_xi(xi)

        cmd_tau = diff_rewards.update_command_in_imagination(s_tau, base_cmd, h_tgt, is_h, is_st)
        B = idx.numel()
        surv = s_tau.new_ones(B)
        J = s_tau.new_zeros(B)
        gen = torch.Generator(device=env.device)
        if noise_seed is not None:
            gen.manual_seed(noise_seed)

        first_action = None
        for tau in range(horizon):
            obs_tau = diff_rewards.build_actor_obs(
                s=s_tau, command=cmd_tau, prev_action=prev_act, foot_contact_prob=p_foot, add_noise=False, ste_contacts=True
            )
            mean_a = actor(obs_tau)
            if noise_seed is not None:
                eps = torch.randn(mean_a.shape, dtype=mean_a.dtype, device=mean_a.device, generator=gen) * sigma
                a_tau = mean_a + eps
            else:
                a_tau = mean_a
            if tau == 0:
                a_tau.retain_grad()
                first_action = a_tau

            s_next, feat_next, c_logits, r_imp = wm.step_with_features(
                s=s_tau, policy_action=a_tau, arm_target=arm_seq[tau, idx], xi_norm=xi_norm, feat_s=feat_tau
            )
            p_foot_next = torch.sigmoid(c_logits)
            ct_next = p_foot_next * (ct + dt)
            at_next = (1.0 - p_foot_next) * (at + dt)
            cmd_next = diff_rewards.update_command_in_imagination(s_next, base_cmd, h_tgt, is_h, is_st)
            r_next, d_next, _ = diff_rewards(
                s_prev=s_tau, s_next=s_next, action=a_tau, prev_action=prev_act, arm_target=arm_seq[tau, idx],
                command=cmd_next, is_standing_env=is_st, xi=xi, foot_contact_prob=p_foot_next,
                contact_time_next=ct_next, air_time_next=at_next, r_impact_pred=r_imp, return_diag=False,
            )
            J = J + (gamma**tau) * surv * r_next
            surv = surv * (1.0 - d_next)
            s_tau, feat_tau, prev_act, p_foot, ct, at, cmd_tau = s_next, feat_next, a_tau, p_foot_next, ct_next, at_next, cmd_next

        x_H = PrivilegedCriticMLP.build_features(feat_tau, cmd_tau, xi_norm, p_foot, ct, at)
        J = J + (gamma**horizon) * surv * critic(x_H)
        loss = -J.mean()
        loss.backward()
        return _flat_grad(params), first_action.grad.detach().clone()

    def _zero_order_pg_grad(idx: torch.Tensor, noise_seed: int, n_antithetic_repeats: int = 1):
        """Score-function (REINFORCE / PPO-style zero-order) gradient estimator on the SAME differentiable MDP."""
        grad_accum = torch.zeros(sum(p.numel() for p in params), device=env.device, dtype=torch.float32)
        B = idx.numel()
        s0 = snap0["s"][idx].detach()
        xi = env.xi[idx].detach()
        xi_norm = wm.spec.normalize_xi(xi)
        base_cmd = snap0["cmd"][idx].detach()
        h_tgt = snap0["heading_target"][idx].detach()
        is_h = snap0["is_heading"][idx].detach()
        is_st = snap0["is_standing"][idx].detach()

        for rep in range(n_antithetic_repeats):
            gen = torch.Generator(device=env.device)
            gen.manual_seed(noise_seed + rep * 9973)

            with torch.no_grad():
                s_tau = s0
                feat_tau = wm.spec.canonical(s_tau)
                prev_act = snap0["prev_action"][idx].detach()
                p_foot = snap0["feet_contact"][idx].detach()
                ct = snap0["contact_time"][idx].detach()
                at = snap0["air_time"][idx].detach()
                cmd_tau = diff_rewards.update_command_in_imagination(s_tau, base_cmd, h_tgt, is_h, is_st)
                surv = s_tau.new_ones(B)

                obs_traj = []
                eps_traj = []
                rew_traj = []
                surv_traj = []
                val_traj = []

                for tau in range(horizon):
                    x_c = PrivilegedCriticMLP.build_features(feat_tau, cmd_tau, xi_norm, p_foot, ct, at)
                    val_traj.append(critic(x_c))
                    obs_tau = diff_rewards.build_actor_obs(
                        s=s_tau, command=cmd_tau, prev_action=prev_act, foot_contact_prob=p_foot,
                        add_noise=False, ste_contacts=False,
                    )
                    mean_a = actor(obs_tau)
                    eps = torch.randn(mean_a.shape, dtype=mean_a.dtype, device=mean_a.device, generator=gen) * sigma
                    a_tau = mean_a + eps
                    obs_traj.append(obs_tau)
                    eps_traj.append(eps)

                    s_next, feat_next, c_logits, r_imp = wm.step_with_features(
                        s=s_tau, policy_action=a_tau, arm_target=arm_seq[tau, idx], xi_norm=xi_norm, feat_s=feat_tau
                    )
                    p_foot_next = torch.sigmoid(c_logits)
                    ct_next = p_foot_next * (ct + dt)
                    at_next = (1.0 - p_foot_next) * (at + dt)
                    cmd_next = diff_rewards.update_command_in_imagination(s_next, base_cmd, h_tgt, is_h, is_st)
                    r_next, d_next, _ = diff_rewards(
                        s_prev=s_tau, s_next=s_next, action=a_tau, prev_action=prev_act, arm_target=arm_seq[tau, idx],
                        command=cmd_next, is_standing_env=is_st, xi=xi, foot_contact_prob=p_foot_next,
                        contact_time_next=ct_next, air_time_next=at_next, r_impact_pred=r_imp, return_diag=False,
                    )
                    rew_traj.append(r_next)
                    surv_traj.append(surv)
                    surv = surv * (1.0 - d_next)
                    s_tau, feat_tau, prev_act, p_foot, ct, at, cmd_tau = s_next, feat_next, a_tau, p_foot_next, ct_next, at_next, cmd_next

                x_H = PrivilegedCriticMLP.build_features(feat_tau, cmd_tau, xi_norm, p_foot, ct, at)
                v_next = critic(x_H)
                # GAE(gamma=0.99, lam=0.95) advantage estimation identical to PPO!
                advs = [None] * horizon
                gae = torch.zeros(B, device=env.device)
                for tau in reversed(range(horizon)):
                    alive_ratio = surv_traj[tau + 1] / (surv_traj[tau] + 1e-8) if tau + 1 < horizon else surv / (surv_traj[tau] + 1e-8)
                    delta = rew_traj[tau] + gamma * alive_ratio * v_next - val_traj[tau]
                    gae = delta + gamma * 0.95 * alive_ratio * gae
                    advs[tau] = gae
                    v_next = val_traj[tau]
                adv_t = torch.stack(advs, dim=0)  # (H, B)
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            actor.zero_grad(set_to_none=True)
            obs_cat = torch.cat(obs_traj, dim=0)  # (H*B, 60)
            eps_cat = torch.cat(eps_traj, dim=0)  # (H*B, 15)
            adv_cat = adv_t.reshape(-1)  # (H*B,)
            # Exact Gaussian log-prob gradient: d/dtheta [- sum (a - mu(s))^2 / (2 sigma^2)]|_{a=mu_old+eps} = (eps / sigma^2) * d mu / d theta
            mean_pred = actor(obs_cat)
            surrogate = -((mean_pred * (eps_cat / (sigma**2))).sum(dim=-1) * adv_cat).mean()
            surrogate.backward()
            grad_accum += _flat_grad(params) / float(n_antithetic_repeats)

        return grad_accum

    full_idx = torch.arange(N, device=env.device)
    half_A = torch.arange(0, N // 2, device=env.device)
    half_B = torch.arange(N // 2, N, device=env.device)

    g_bptt_det, _ = _bptt_grad_on_indices(full_idx, noise_seed=None)
    g_bptt_halfA, _ = _bptt_grad_on_indices(half_A, noise_seed=None)
    g_bptt_halfB, _ = _bptt_grad_on_indices(half_B, noise_seed=None)
    g_bptt_noisy1, _ = _bptt_grad_on_indices(full_idx, noise_seed=101)
    g_bptt_noisy2, _ = _bptt_grad_on_indices(full_idx, noise_seed=202)

    g_zo_1x_a = _zero_order_pg_grad(full_idx, noise_seed=301, n_antithetic_repeats=1)
    g_zo_1x_b = _zero_order_pg_grad(full_idx, noise_seed=401, n_antithetic_repeats=1)
    g_zo_16x_a = _zero_order_pg_grad(full_idx, noise_seed=501, n_antithetic_repeats=16)
    g_zo_16x_b = _zero_order_pg_grad(full_idx, noise_seed=701, n_antithetic_repeats=16)

    # Multi-joint Jacobian allocation under Extreme Torso Bow (pitch = 1.20 rad > waist_pitch 0.52 rad limit)
    bow_cmd = torch.tensor([0.0, 0.0, 0.0, 0.62, 0.0, 1.20, 0.0], device=env.device).unsqueeze(0).expand(N, 7)
    _, dJ_da0_bow = _bptt_grad_on_indices(full_idx, noise_seed=None, cmd_override=bow_cmd)
    # Note: loss = -J.mean(), so |d(-J/N)/da0| * N = |dJ_i / da0_i|
    sens = (dJ_da0_bow.abs() * float(N))  # (N, 15) in custom_joint_ids order:
    # 0:L_hip_p, 1:L_hip_r, 2:L_hip_y, 3:L_knee, 4:L_ank_p, 5:L_ank_r,
    # 6:R_hip_p, 7:R_hip_r, 8:R_hip_y, 9:R_knee, 10:R_ank_p, 11:R_ank_r,
    # 12:waist_y, 13:waist_r, 14:waist_p
    waist_p_mag = float(sens[:, 14].mean().item())
    hip_p_mag = float(sens[:, [0, 6]].mean().item())
    knee_mag = float(sens[:, [3, 9]].mean().item())
    ankle_p_mag = float(sens[:, [4, 10]].mean().item())
    lateral_yaw_mag = float(sens[:, [1, 2, 5, 7, 8, 11, 12, 13]].mean().item())
    sagittal_total = waist_p_mag + 2.0 * hip_p_mag + 2.0 * knee_mag + 2.0 * ankle_p_mag
    lower_body_sagittal_share = (2.0 * (hip_p_mag + knee_mag + ankle_p_mag)) / max(sagittal_total, 1e-8)

    return {
        "gradient_directional_cosine_consistency": {
            "analytic_bptt_noisy_draw1_vs_draw2_same_states": round(float(F.cosine_similarity(g_bptt_noisy1, g_bptt_noisy2, dim=0).item()), 4),
            "analytic_bptt_disjoint_states_halfA_vs_halfB": round(float(F.cosine_similarity(g_bptt_halfA, g_bptt_halfB, dim=0).item()), 4),
            "analytic_bptt_noisy_vs_true_det": round(float(F.cosine_similarity(g_bptt_noisy1, g_bptt_det, dim=0).item()), 4),
            "zero_order_pg_1x_budget_draw1_vs_draw2": round(float(F.cosine_similarity(g_zo_1x_a, g_zo_1x_b, dim=0).item()), 4),
            "zero_order_pg_16x_budget_draw1_vs_draw2": round(float(F.cosine_similarity(g_zo_16x_a, g_zo_16x_b, dim=0).item()), 4),
        },
        "extreme_bow_1p20rad_action_jacobian_allocation": {
            "waist_pitch_mean_dJ_da": round(waist_p_mag, 4),
            "hip_pitch_mean_dJ_da": round(hip_p_mag, 4),
            "knee_mean_dJ_da": round(knee_mag, 4),
            "ankle_pitch_mean_dJ_da": round(ankle_p_mag, 4),
            "non_sagittal_roll_yaw_mean_dJ_da": round(lateral_yaw_mag, 4),
            "lower_body_sagittal_gradient_share_pct": round(lower_body_sagittal_share * 100.0, 2),
        },
    }


@torch.no_grad()
def run_d3_counterfactual_command_relabeling(
    env: SuperMimicG1Env,
    wm: G1WorldModel,
    actor: TeacherActorMLP,
    diff_rewards: DifferentiableG1Rewards,
    horizon: int = 12,
) -> dict:
    """Test whether command-free G1WorldModel predicts counterfactual returns when relabeled with unseen commands."""
    cf_commands = [
        ("walk_forward", (0.45, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0), False),
        ("deep_squat", (0.0, 0.0, 0.0, 0.44, 0.0, 0.0, 0.0), True),
        ("deep_bow", (0.0, 0.0, 0.0, 0.64, 0.0, 1.05, 0.0), True),
        ("roll_yaw_twist", (0.0, 0.0, 0.0, 0.72, 0.40, 0.0, 0.80), True),
    ]
    imag_returns = []
    physx_returns = []
    dt = float(env.step_dt)

    for _name, cmd_tuple, is_st_bool in cf_commands:
        # Reset & advance identically for 40 steps under random commands to reach identical pre-branch state s_0
        _seed_all(env, args_cli.seed + 77)
        all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env.reset(all_ids)
        obs, _ = env.get_observations()
        for _ in range(40):
            obs, _, _, _ = env.step(actor(obs))

        # Snapshot pre-branch state s_0 (collected under random original commands != cmd_tuple!)
        snap0 = env.current_step_inputs()
        N = env.num_envs
        env_ar = torch.arange(N, device=env.device)
        arm_seq = torch.stack(
            [env.amass[env_ar, (snap0["amass_step"] + tau) % env.sequence_length] for tau in range(horizon)],
            dim=0,
        )
        cmd_t = torch.tensor(cmd_tuple, dtype=torch.float32, device=env.device).unsqueeze(0).expand(N, 7)
        is_st_t = torch.full((N,), float(is_st_bool), device=env.device)

        # 1. Counterfactual closed-loop rollout in G1WorldModel imagination
        s_tau = snap0["s"].clone()
        feat_tau = wm.spec.canonical(s_tau)
        prev_act = snap0["prev_action"].clone()
        p_foot = snap0["feet_contact"].clone()
        ct = snap0["contact_time"].clone()
        at = snap0["air_time"].clone()
        xi = env.xi.clone()
        xi_norm = wm.spec.normalize_xi(xi)
        j_imag = torch.zeros(N, device=env.device)

        for tau in range(horizon):
            obs_tau = diff_rewards.build_actor_obs(
                s=s_tau, command=cmd_t, prev_action=prev_act, foot_contact_prob=p_foot, add_noise=False, ste_contacts=True
            )
            a_tau = actor(obs_tau)
            s_next, feat_next, c_logits, r_imp = wm.step_with_features(
                s=s_tau, policy_action=a_tau, arm_target=arm_seq[tau], xi_norm=xi_norm, feat_s=feat_tau
            )
            p_foot_next = torch.sigmoid(c_logits)
            ct_next = p_foot_next * (ct + dt)
            at_next = (1.0 - p_foot_next) * (at + dt)
            r_hat, _, _ = diff_rewards(
                s_prev=s_tau, s_next=s_next, action=a_tau, prev_action=prev_act, arm_target=arm_seq[tau],
                command=cmd_t, is_standing_env=is_st_t, xi=xi, foot_contact_prob=p_foot_next,
                contact_time_next=ct_next, air_time_next=at_next, r_impact_pred=r_imp, return_diag=False,
            )
            j_imag += r_hat
            s_tau, feat_tau, prev_act, p_foot, ct, at = s_next, feat_next, a_tau, p_foot_next, ct_next, at_next

        # 2. Ground-truth counterfactual closed-loop rollout in Real PhysX from exact same state s_0
        cg = env.command_generator
        j_real = torch.zeros(N, device=env.device)
        alive = torch.ones(N, dtype=torch.bool, device=env.device)
        for _ in range(horizon):
            cg.vel_command_b.copy_(cmd_t)
            cg.is_heading_env.fill_(False)
            cg.is_zero_vel_yaw_env.fill_(False)
            cg.is_standing_env.fill_(is_st_bool)
            cg.time_left.fill_(1.0e6)
            obs[:, 6:13] = cg.command * env.obs_scales.commands
            act = actor(obs)
            obs, rew, dones, _ = env.step(act)
            alive = alive & (~(dones > 0))
            j_real += rew * alive.float()

        imag_returns.append(j_imag[alive])
        physx_returns.append(j_real[alive])

    im_all = torch.cat(imag_returns)
    re_all = torch.cat(physx_returns)
    im_c = im_all - im_all.mean()
    re_c = re_all - re_all.mean()
    pearson_r = float((im_c * re_c).sum().item() / (torch.norm(im_c) * torch.norm(re_c) + 1e-8).item())
    mae_per_step = float(((im_all - re_all).abs().mean().item()) / float(horizon))
    return {
        "num_counterfactual_branches_tested": int(im_all.numel()),
        "horizon_steps": horizon,
        "counterfactual_return_pearson_r": round(pearson_r, 4),
        "per_step_return_mae": round(mae_per_step, 4),
    }


@torch.no_grad()
def run_d4_straight_walk_yaw_drift_diagnosis(
    env: SuperMimicG1Env,
    sm_policy,
    cmu_ppo_policy,
    steps: int = 250,
) -> dict:
    """Measure 5.0s straight-walk (`vx=0.45 m/s`) yaw & lateral drift under Open-Loop `wz=0` vs Heading Hold."""
    default_arms = env.custom_default_joint_pos[:, env.num_actions : env.num_actions + 14].clone()
    results = {}

    for pol_name, policy in [("SM_3000", sm_policy), ("CMU_PPO_240k", cmu_ppo_policy)]:
        pol_res = {}
        for cond_name, use_heading, neutral_arms in [
            ("open_loop_wz0_with_amass_arms", False, False),
            ("open_loop_wz0_with_neutral_arms", False, True),
            ("closed_loop_heading_hold_with_amass_arms", True, False),
        ]:
            _seed_all(env, 42)
            all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
            env.arm_targets = default_arms.clone() if neutral_arms else None
            env.reset(all_ids)

            # Place all robots cleanly at env_origins facing exact +X (`heading = 0.0`)
            root_state = env.robot.data.default_root_state.clone()
            root_state[:, :3] += env.scene.env_origins
            env.robot.write_root_pose_to_sim(root_state[:, :7])
            env.robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
            env.robot.write_joint_state_to_sim(env.robot.data.default_joint_pos, env.robot.data.default_joint_vel)
            env.scene.write_data_to_sim()
            env.sim.forward()
            obs, _ = env.get_observations()

            cg = env.command_generator
            base_cmd = torch.tensor([0.45, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(env.num_envs, 7)
            alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

            for _ in range(steps):
                cg.vel_command_b.copy_(base_cmd)
                cg.heading_target.zero_()
                cg.is_heading_env.fill_(use_heading)
                cg.is_zero_vel_yaw_env.fill_(False)
                cg.is_standing_env.fill_(False)
                cg.time_left.fill_(1.0e6)
                if use_heading:
                    cg._update_command()
                obs[:, 6:13] = cg.command * env.obs_scales.commands
                act = policy(obs)
                obs, _, dones, _ = env.step(act)
                alive = alive & (~(dones > 0))

            p_local = (env.robot.data.root_pos_w - env.scene.env_origins)[alive]
            heading_deg = env.robot.data.heading_w[alive].abs() * (180.0 / math.pi)
            pol_res[cond_name] = {
                "forward_progress_x_m": round(float(p_local[:, 0].mean().item()), 3),
                "abs_lateral_drift_y_m": round(float(p_local[:, 1].abs().mean().item()), 3),
                "signed_mean_lateral_y_m": round(float(p_local[:, 1].mean().item()), 3),
                "abs_heading_drift_deg": round(float(heading_deg.mean().item()), 2),
                "survival_pct": round(float(alive.float().mean().item() * 100.0), 2),
            }
        env.arm_targets = None
        results[pol_name] = pol_res

    return results


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    sm_last_path = os.path.join(os.path.abspath(args_cli.sm_dir), "supermimic_last.pt")
    sm_rsl_path = os.path.join(os.path.abspath(args_cli.sm_dir), "model_3000.pt")
    apply_trained_actuator_gains(env_cfg, os.path.abspath(args_cli.sm_dir))

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.commands.curriculum.enable = False
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.seed = int(args_cli.seed)

    env = SuperMimicG1Env(env_cfg, headless=True)
    device = torch.device(env.device)

    # Load full SuperMimic World Model + Actor + Critic
    fk = UsdArticulationFK(env.robot, device=device)
    diff_rewards = DifferentiableG1Rewards(env, fk).to(device)
    default_arm_pos = env.custom_default_joint_pos[0, env.num_actions : env.num_actions + 14].clone()
    wm = G1WorldModel(
        cfg=WorldModelCfg(),
        action_scale=diff_rewards.action_scale,
        default_arm_pos=default_arm_pos,
        clip_actions=env.clip_actions,
        device=device,
    )
    actor = TeacherActorMLP(ActorCfg()).to(device)
    critic = PrivilegedCriticMLP(CriticCfg()).to(device)

    ck = torch.load(sm_last_path, map_location=device, weights_only=False)
    wm.load_state_dict(ck["wm"])
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    wm.eval()
    actor.eval()
    critic.eval()

    # Load RSL-RL inference wrappers for SM_3000 and CMU_PPO_240k
    sm_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.abspath(args_cli.sm_dir), device=agent_cfg.device)
    sm_runner.load(sm_rsl_path, load_optimizer=False)
    sm_policy = sm_runner.get_inference_policy(device=env.device)

    cmu_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(os.path.abspath(args_cli.cmu_ppo_ckpt)), device=agent_cfg.device)
    cmu_runner.load(os.path.abspath(args_cli.cmu_ppo_ckpt), load_optimizer=False)
    cmu_ppo_policy = cmu_runner.get_inference_policy(device=env.device)

    print("[DIAG] Running D1: Multi-Step World Model Horizon Fidelity...", flush=True)
    d1 = run_d1_world_model_horizon_fidelity(env, wm, actor, diff_rewards, horizon=24)
    print(json.dumps(d1, indent=2), flush=True)

    print("[DIAG] Running D2: Analytic BPTT vs Zero-Order PG Gradient SNR & Multi-Joint Jacobians...", flush=True)
    d2 = run_d2_gradient_snr_and_jacobian_coordination(env, wm, actor, critic, diff_rewards, horizon=24, sigma=0.20)
    print(json.dumps(d2, indent=2), flush=True)

    print("[DIAG] Running D3: Counterfactual Command Relabeling Fidelity...", flush=True)
    d3 = run_d3_counterfactual_command_relabeling(env, wm, actor, diff_rewards, horizon=12)
    print(json.dumps(d3, indent=2), flush=True)

    print("[DIAG] Running D4: Straight-Walk Yaw Drift Diagnosis...", flush=True)
    d4 = run_d4_straight_walk_yaw_drift_diagnosis(env, sm_policy, cmu_ppo_policy, steps=250)
    print(json.dumps(d4, indent=2), flush=True)

    report = {
        "D1_world_model_horizon_fidelity": d1,
        "D2_gradient_snr_and_jacobian_coordination": d2,
        "D3_counterfactual_command_relabeling": d3,
        "D4_straight_walk_yaw_drift_diagnosis": d4,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_json)), exist_ok=True)
    with open(args_cli.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[DIAG] Saved full diagnostic report -> {args_cli.out_json}", flush=True)


if __name__ == "__main__":
    import sys
    import traceback

    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

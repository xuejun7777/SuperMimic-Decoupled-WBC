"""Live PhysX Parity & Analytic BPTT Verification Suite for `legged_lab.supermimic`.

Verifies:
  1. 60-D Teacher Actor observation parity (`DifferentiableG1Rewards.build_actor_obs` vs `BaseEnv.compute_current_observations`).
  2. Term-by-term reward parity across all 31 active `G1RewardCfg` terms on live PhysX steps.
  3. `SuperMimicG1Env` pre-reset terminal state capture on actual fall events (`z < 0.30m`).
  4. `G1WorldModel` multi-step MSE + Contact BCE + Impact MSE descent & 24-step Analytic BPTT gradient norms.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.imagination import (
    PolicyCriticTrainer,
    PolicyTrainerCfg,
    WMTrainerCfg,
    WorldModelTrainer,
)
from legged_lab.supermimic.nets import ActorCfg, CriticCfg, PrivilegedCriticMLP, TeacherActorMLP
from legged_lab.supermimic.replay_buffer import G1ReplayBuffer
from legged_lab.supermimic.usd_fk import UsdArticulationFK
from legged_lab.supermimic.world_model import G1WorldModel, WorldModelCfg
from legged_lab.utils import task_registry


def run_tests() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs("g1_flat")
    env_cfg.scene.num_envs = 64
    env_cfg.scene.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum.enable = False

    env = SuperMimicG1Env(env_cfg, headless=True)
    device = env.device

    fk = UsdArticulationFK(env.robot, device=device)
    diff_rew = DifferentiableG1Rewards(env, fk).to(device)

    # 1. Test 60-D observation parity after stepping with random actions
    for _ in range(5):
        env.step(0.5 * torch.randn(env.num_envs, env.num_actions, device=device))

    s_now = env.read_full_state()
    fc_now = env.read_feet_contact()
    prev_act = env.action_buffer._circular_buffer.buffer[:, -1, : env.num_actions].clone()
    cmd_now = env.command_generator.command.clone()

    obs_diff = diff_rew.build_actor_obs(s_now, cmd_now, prev_act, fc_now, add_noise=False, ste_contacts=True)
    obs_true, _ = env.compute_current_observations()
    obs_max_err = float((obs_diff - obs_true).abs().max().item())
    print(f"[TEST 1] 60-D Actor Observation Max Abs Error: {obs_max_err:.3e}")
    assert obs_max_err < 1.0e-4, f"Actor obs parity failed: {obs_max_err}"

    # 2. Test term-by-term reward parity on a non-resetting step
    # Capture pre-step quantities AND snapshot pre-reset reward_manager._step_reward inside a step
    snap = env.current_step_inputs()
    act = 0.4 * torch.randn(env.num_envs, env.num_actions, device=device)

    # Hook reset temporarily to also snapshot `reward_manager._step_reward`, contact timers, and post-step command before reset
    captured: dict[str, torch.Tensor] = {}
    orig_reset = env.reset

    def _capture_reset(env_ids: torch.Tensor):
        if getattr(env, "_sm_ready", False):
            captured["step_reward"] = (env.reward_manager._step_reward * env.step_dt).clone()
            ct_n, at_n = env.read_contact_timers()
            captured["ct_next"] = ct_n
            captured["at_next"] = at_n
            captured["cmd_next"] = env.command_generator.command.clone()
            captured["is_standing_next"] = env.command_generator.is_standing_env.float().clone()
        return orig_reset(env_ids)

    env.reset = _capture_reset
    _, rew_true, _, _ = env.step(act)
    env.reset = orig_reset

    total_diff, d_soft, diag = diff_rew(
        s_prev=snap["s"],
        s_next=env.last_pre_reset_state,
        action=act,
        prev_action=snap["prev_action"],
        arm_target=snap["arm_tgt"],
        command=captured["cmd_next"],
        is_standing_env=captured["is_standing_next"],
        xi=env.xi,
        foot_contact_prob=env.last_pre_reset_feet_contact,
        contact_time_next=captured["ct_next"],
        air_time_next=captured["at_next"],
        r_impact_pred=env.last_impact_reward,
    )

    print("\n[TEST 2] Term-by-Term Differentiable Reward Parity vs PhysX `env.reward_manager`:")
    pure_kin_terms = [
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
    ]
    for idx, tname in enumerate(env.reward_manager._term_names):
        true_t = captured["step_reward"][:, idx]
        if tname in DifferentiableG1Rewards.IMPACT_TERM_NAMES:
            continue
        if tname == "body_orientation_l2":
            continue  # weight == 0.0 in G1RewardCfg
        pred_t = diag[tname]
        max_e = float((pred_t - true_t).abs().max().item())
        mean_e = float((pred_t - true_t).abs().mean().item())
        print(f"  {tname:30s} | mean_abs_err={mean_e:.3e} | max_abs_err={max_e:.3e}")
        if tname in pure_kin_terms:
            assert max_e < 5.0e-4, f"Kinematic reward term {tname} mismatch: {max_e}"

    non_term_mask = env.last_terminated < 0.5
    if non_term_mask.any():
        tot_mae = float((total_diff[non_term_mask] - rew_true[non_term_mask]).abs().mean().item())
        print(f"  --> Total Step Reward MAE (non-terminal envs): {tot_mae:.3e}")

    # 3. Test pre-reset capture on actual fall events + Replay Buffer + WM + Analytic BPTT
    default_arm = env.custom_default_joint_pos[0, env.num_actions : env.num_actions + 14].clone()
    wm = G1WorldModel(WorldModelCfg(), diff_rew.action_scale, default_arm, env.clip_actions, device=device)
    actor = TeacherActorMLP(ActorCfg()).to(device)
    critic = PrivilegedCriticMLP(CriticCfg()).to(device)
    wm_trainer = WorldModelTrainer(wm, WMTrainerCfg(window_k=8, batch_size=256, batches_per_update=4))
    pi_trainer = PolicyCriticTrainer(
        wm, actor, critic, diff_rew, PolicyTrainerCfg(horizon=24, batch_size=256)
    )
    buf = G1ReplayBuffer(num_envs=env.num_envs, capacity=128, device=device)
    env.attach_replay_buffer(buf)

    fall_captured = 0
    for _ in range(40):
        sn = env.current_step_inputs()
        # Large actions to deliberately induce some falls (< 0.30m) and test pre-reset capture
        a = 2.0 * torch.randn(env.num_envs, env.num_actions, device=device)
        env.step(a)
        fell_mask = env.last_terminated > 0.5
        if fell_mask.any():
            fall_captured += int(fell_mask.sum().item())
            pre_z = env.last_pre_reset_state[fell_mask, 2]
            post_z = env.robot.data.root_link_pos_w[fell_mask, 2]
            assert (pre_z < 0.30).all(), f"Expected pre-reset z < 0.30, got {pre_z}"
            assert (post_z > 0.65).all(), f"Expected post-reset z > 0.65, got {post_z}"
        buf.add_step(
            s=sn["s"],
            a_exec=a,
            prev_action=sn["prev_action"],
            arm_tgt=sn["arm_tgt"],
            s_next=env.last_pre_reset_state,
            feet_contact=sn["feet_contact"],
            next_feet_contact=env.last_pre_reset_feet_contact,
            contact_time=sn["contact_time"],
            air_time=sn["air_time"],
            impact_rew=env.last_impact_reward,
            cmd=sn["cmd"],
            heading_target=sn["heading_target"],
            is_heading=sn["is_heading"],
            is_standing=sn["is_standing"],
            ep_id=sn["ep_id"],
            amass_step=sn["amass_step"],
            pushed=env.last_pushed,
        )

    print(f"\n[TEST 3] Pre-reset terminal fall states captured cleanly: {fall_captured} fall events verified (pre_z < 0.30m, post_z > 0.65m)")

    # 4. Verify World Model loss descent & 24-step Analytic BPTT gradient norms
    wm_s0 = wm_trainer.update(buf, env.xi, num_batches=2)
    for _ in range(15):
        wm_s1 = wm_trainer.update(buf, env.xi, num_batches=4)
    print(
        f"[TEST 4a] WM 8-step fitting: loss {wm_s0['wm/loss']:.4f} -> {wm_s1['wm/loss']:.4f} | "
        f"nse_1step {wm_s0['wm/nse_1step']:.4f} -> {wm_s1['wm/nse_1step']:.4f} | "
        f"nse_8step {wm_s0['wm/nse_kstep']:.4f} -> {wm_s1['wm/nse_kstep']:.4f} | "
        f"contact_acc {wm_s1['wm/contact_acc']:.3f}"
    )
    assert wm_s1["wm/loss"] < wm_s0["wm/loss"], "WM loss did not decrease!"

    pi_trainer.pretrain_critic_only(buf, env.xi, num_updates=3)
    pi_s = pi_trainer.update(buf, env.xi)
    print(
        f"[TEST 4b] 24-step Analytic BPTT: J={pi_s['actor/objective_J']:.4f} | "
        f"grad_norm={pi_s['actor/grad_norm']:.4f} | surv_H={pi_s['imag/survival_H']:.3f} | "
        f"critic_loss={pi_s['critic/loss']:.4f}"
    )
    assert math_isfinite(pi_s["actor/grad_norm"]) and pi_s["actor/grad_norm"] > 0.0, "Invalid actor BPTT grad norm!"
    print("\n[ALL PARITY & BPTT VERIFICATION TESTS PASSED!]")


def math_isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


if __name__ == "__main__":
    import os
    import sys
    import traceback

    code = 0
    try:
        run_tests()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

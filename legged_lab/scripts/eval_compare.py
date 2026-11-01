"""Standardized Apples-to-Apples Evaluation Benchmark for PPO vs SuperMimic on Unitree G1 WBC.

Loads any number of RSL-RL-compatible checkpoints (`--ckpt LABEL=PATH_TO_MODEL_PT`) — whether
trained by `train_teacher.py` (PPO) or `train_supermimic.py` (SuperMimic) — through the EXACT
same `OnPolicyRunner.get_inference_policy()` onto a single untouched `BaseEnv` (`g1_flat`)
with `curriculum.enable=False`, `noise.add_noise=False`, `push_robot=None`, `action_delay=False`,
and active AMASS upper-body arm disturbances.

Evaluates every checkpoint across:
  1. `full_random`: Full uncurriculated 7D command distribution (`vx,vy in [-0.5,0.5]`,
     `wz in [-1.57,1.57]`, `height in [0.35,0.80]`, `roll in [-0.7,0.7]`,
     `pitch in [-0.52,1.57]`, `yaw in [-1.57,1.57]`) over a full 20s (1000-step) episode.
  2. Five targeted decoupled locomotion + body-pose modes (`walk_upright`, `squat_deep`,
     `torso_pitch_bow`, `walk_squat_pitch`, `torso_roll_yaw`).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apples-to-Apples evaluation for G1 Decoupled WBC policies.")
parser.add_argument("--task", type=str, default="g1_flat", help="Task name.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel evaluation environments.")
parser.add_argument("--seed", type=int, default=1234, help="Evaluation seed.")
parser.add_argument("--steps", type=int, default=1000, help="Steps per suite (1000 = full 20.0s episode).")
parser.add_argument(
    "--ckpt",
    action="append",
    required=True,
    help="LABEL=PATH_TO_CHECKPOINT (repeatable, e.g. --ckpt PPO=logs/.../model_3000.pt --ckpt SuperMimic=logs/.../model_3000.pt).",
)
parser.add_argument("--out_json", type=str, default="", help="Optional path to save structured JSON report.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import isaaclab.utils.math as math_utils
import torch
from rsl_rl.runners import OnPolicyRunner

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils import task_registry

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


import numpy as np
import legged_lab.utils.amass as amass_module
from legged_lab.utils.cli_args import apply_trained_actuator_gains

TARGETED_MODES: dict[str, tuple[float, float, float, float, float, float, float]] = {
    # (lin_vel_x, lin_vel_y, ang_vel_z, body_height, body_roll, body_pitch, body_yaw)
    "walk_upright": (0.45, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0),
    "walk_lateral_turn": (0.0, 0.35, 0.60, 0.72, 0.0, 0.0, 0.0),
    "squat_deep": (0.0, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0),
    "torso_pitch_bow": (0.0, 0.0, 0.0, 0.68, 0.0, 0.80, 0.0),
    "extreme_pitch_1p2": (0.0, 0.0, 0.0, 0.62, 0.0, 1.20, 0.0),
    "walk_squat_pitch": (0.35, 0.0, 0.0, 0.55, 0.0, 0.50, 0.0),
    "torso_roll_yaw": (0.0, 0.0, 0.0, 0.72, 0.40, 0.0, 0.80),
}


@torch.no_grad()
def _measure_tracking_errors(env, torso_id: int, pelvis_id: int) -> dict[str, torch.Tensor]:
    """Compute exact per-env absolute tracking errors matching `legged_lab/mdp/rewards.py`."""
    robot = env.robot
    cmd = env.command_generator.command

    vel_yaw = quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    err_vx = (vel_yaw[:, 0] - cmd[:, 0]).abs()
    err_vy = (vel_yaw[:, 1] - cmd[:, 1]).abs()
    err_vxy = torch.norm(vel_yaw[:, :2] - cmd[:, :2], dim=-1)
    err_wz = (robot.data.root_ang_vel_w[:, 2] - cmd[:, 2]).abs()

    torso_h = robot.data.body_pos_w[:, torso_id, 2]
    err_h = (torso_h - cmd[:, 3]).abs()

    q_pelvis = robot.data.body_quat_w[:, pelvis_id]
    q_torso = robot.data.body_quat_w[:, torso_id]
    q_pelvis_yaw = math_utils.yaw_quat(q_pelvis)
    q_torso_rel = math_utils.quat_mul(math_utils.quat_conjugate(q_pelvis_yaw), q_torso)
    roll_act, pitch_act, _ = math_utils.euler_xyz_from_quat(q_torso_rel)
    roll_act = math_utils.wrap_to_pi(roll_act)
    pitch_act = math_utils.wrap_to_pi(pitch_act)

    q_yaw_rel = math_utils.quat_mul(math_utils.quat_conjugate(q_pelvis_yaw), math_utils.yaw_quat(q_torso))
    yaw_act = 2.0 * torch.atan2(q_yaw_rel[:, 3], q_yaw_rel[:, 0])

    err_roll = math_utils.wrap_to_pi(roll_act - cmd[:, 4]).abs()
    err_pitch = math_utils.wrap_to_pi(pitch_act - cmd[:, 5]).abs()
    err_yaw = math_utils.wrap_to_pi(yaw_act - cmd[:, 6]).abs()

    return {
        "vx_mae": err_vx,
        "vy_mae": err_vy,
        "vxy_err": err_vxy,
        "wz_mae": err_wz,
        "height_mae": err_h,
        "roll_mae": err_roll,
        "pitch_mae": err_pitch,
        "yaw_mae": err_yaw,
    }


@torch.no_grad()
def _set_fixed_command(env, fixed_cmd: tuple[float, ...], is_standing: bool) -> None:
    cmd_t = torch.tensor(fixed_cmd, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, 7)
    cg = env.command_generator
    cg.vel_command_b.copy_(cmd_t)
    cg.is_heading_env.fill_(False)
    cg.is_zero_vel_yaw_env.fill_(False)
    cg.is_standing_env.fill_(is_standing)
    cg.time_left.fill_(1.0e6)


@torch.no_grad()
def _run_suite(
    env,
    policy,
    steps: int,
    torso_id: int,
    pelvis_id: int,
    seed: int,
    fixed_cmd: tuple[float, ...] | None = None,
) -> dict[str, float]:
    # Seed BOTH simulator/PyTorch RNG and global AMASS clip loader RNG for exact paired determinism (F8 fix)
    env.seed(seed)
    if getattr(amass_module, "_global_amass_loader", None) is not None:
        amass_module._global_amass_loader._rng = np.random.default_rng(seed)
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset(all_ids)

    is_standing = False
    if fixed_cmd is not None:
        is_standing = (abs(fixed_cmd[0]) + abs(fixed_cmd[1]) + abs(fixed_cmd[2])) < 1.0e-3
        _set_fixed_command(env, fixed_cmd, is_standing)

    obs, _ = env.get_observations()

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    first_ep_len = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    first_ep_ret = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    err_sums: dict[str, float] = {}
    err_sq_sums: dict[str, float] = {}
    active_steps = 0.0
    act_delta_sum = 0.0
    prev_actions: torch.Tensor | None = None

    for step_i in range(steps):
        if fixed_cmd is not None:
            _set_fixed_command(env, fixed_cmd, is_standing)
            obs[:, 6:13] = env.command_generator.command * env.obs_scales.commands

        actions = policy(obs)
        if prev_actions is not None:
            act_delta_sum += float(torch.norm(actions - prev_actions, dim=-1).mean().item())
        prev_actions = actions.clone()

        # Step simulator; note `env.step()` auto-resets any environment with `dones > 0` (falls OR timeout)
        obs, rew, dones, _ = env.step(actions)

        # Accumulate tracking errors strictly on environments that have NOT reset on this step (F7 fix)
        still_alive = alive & (~(dones > 0))
        if step_i >= 25 and still_alive.any():
            errs = _measure_tracking_errors(env, torso_id, pelvis_id)
            cnt = float(still_alive.float().sum().item())
            active_steps += cnt
            for k_e, v_e in errs.items():
                err_sums[k_e] = err_sums.get(k_e, 0.0) + float(v_e[still_alive].sum().item())
                err_sq_sums[k_e] = err_sq_sums.get(k_e, 0.0) + float((v_e[still_alive] ** 2).sum().item())

        first_ep_ret += rew * alive.float()
        first_ep_len += alive.float()
        alive = alive & (~(dones > 0))

    denom = max(active_steps, 1.0)
    out = {
        "survival_rate": float(alive.float().mean().item() if steps < int(env.max_episode_length) else (first_ep_len >= steps).float().mean().item()),
        "mean_ep_len": float(first_ep_len.mean().item()),
        "mean_ep_return": float(first_ep_ret.mean().item()),
        "action_delta_l2": act_delta_sum / max(steps - 1, 1),
    }
    for k_e in err_sums:
        out[k_e] = err_sums[k_e] / denom
        out[k_e.replace("_mae", "_rmse").replace("_err", "_rmse")] = (err_sq_sums[k_e] / denom) ** 0.5
    return out


def evaluate_all() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    first_ckpt_dir = os.path.dirname(os.path.abspath(args_cli.ckpt[0].split("=", 1)[1]))
    apply_trained_actuator_gains(env_cfg, first_ckpt_dir)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.commands.curriculum.enable = False
    env_cfg.scene.max_episode_length_s = 20.0
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.seed = int(args_cli.seed)

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    torso_id = env.robot.body_names.index("torso_link")
    pelvis_id = env.robot.body_names.index("pelvis")

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for spec in args_cli.ckpt:
        if "=" not in spec:
            raise ValueError(f"Expected --ckpt LABEL=PATH, got {spec}")
        label, ckpt_path = spec.split("=", 1)
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print("\n============================================================", flush=True)
        print(f"[EVAL] Evaluating {label} -> {ckpt_path}", flush=True)
        print("============================================================", flush=True)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(ckpt_path), device=agent_cfg.device)
        runner.load(ckpt_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)

        suite_res: dict[str, dict[str, float]] = {}

        # 1. Full random 7D uncurriculated distribution
        suite_res["full_random"] = _run_suite(
            env, policy, steps=args_cli.steps, torso_id=torso_id, pelvis_id=pelvis_id, seed=args_cli.seed, fixed_cmd=None
        )
        r0 = suite_res["full_random"]
        print(
            f"  [full_random      ] surv={r0['survival_rate']*100:5.1f}% | len={r0['mean_ep_len']:6.1f} | "
            f"ret={r0['mean_ep_return']:6.2f} | vxy={r0['vxy_err']:.3f}m/s | wz={r0['wz_mae']:.3f}rad/s | "
            f"h={r0['height_mae']*100:.2f}cm | roll={r0['roll_mae']:.3f} | pitch={r0['pitch_mae']:.3f} | "
            f"yaw={r0['yaw_mae']:.3f} | d_act={r0['action_delta_l2']:.3f}",
            flush=True,
        )

        # 2. Targeted modes
        for idx, (mode_name, cmd_tuple) in enumerate(TARGETED_MODES.items()):
            res = _run_suite(
                env,
                policy,
                steps=min(args_cli.steps, 500),
                torso_id=torso_id,
                pelvis_id=pelvis_id,
                seed=args_cli.seed + 100 + idx,
                fixed_cmd=cmd_tuple,
            )
            suite_res[mode_name] = res
            print(
                f"  [{mode_name:17s}] surv={res['survival_rate']*100:5.1f}% | ret={res['mean_ep_return']:6.2f} | "
                f"vxy={res['vxy_err']:.3f} | wz={res['wz_mae']:.3f} | h={res['height_mae']*100:.2f}cm | "
                f"roll={res['roll_mae']:.3f} | pitch={res['pitch_mae']:.3f} | yaw={res['yaw_mae']:.3f} | "
                f"d_act={res['action_delta_l2']:.3f}",
                flush=True,
            )

        all_results[label] = suite_res

    if args_cli.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_json)), exist_ok=True)
        with open(args_cli.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[EVAL] Saved structured JSON comparison to {args_cli.out_json}", flush=True)


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        evaluate_all()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

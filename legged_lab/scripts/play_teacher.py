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

import argparse
import os

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained teacher policy in simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--action_log_interval",
    type=int,
    default=100,
    help="Print action diagnostics every N policy steps after warmup. Set 0 to disable.",
)
parser.add_argument(
    "--action_log_warmup_steps",
    type=int,
    default=50,
    help="Print action diagnostics every step for the first N policy steps.",
)
parser.add_argument(
    "--action_log_env",
    type=int,
    default=0,
    help="Environment index to print detailed action diagnostics for.",
)
parser.add_argument(
    "--joint_limit_margin",
    type=float,
    default=0.05,
    help="Joint target distance-to-limit threshold (rad) counted as near-limit.",
)
parser.add_argument(
    "--continuous_torso",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Hold torso keys (H/J/Z/X/C/V/B/N) to ramp commands continuously. Default: disabled (discrete step per key press).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import apply_trained_actuator_gains, update_rsl_rl_cfg


def _action_scale_for_logging(env, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(env.action_scale):
        return env.action_scale[: env.num_actions].to(device=device, dtype=dtype)
    return torch.full((env.num_actions,), float(env.action_scale), device=device, dtype=dtype)


def _log_action_diagnostics(env, step: int, actions: torch.Tensor, prev_actions: torch.Tensor | None) -> None:
    interval = args_cli.action_log_interval
    if interval <= 0:
        return
    if step >= args_cli.action_log_warmup_steps and step % interval != 0:
        return

    env_idx = min(max(args_cli.action_log_env, 0), actions.shape[0] - 1)
    raw = actions.detach()
    delta = torch.zeros_like(raw) if prev_actions is None else raw - prev_actions
    clipped = torch.clamp(raw, -env.clip_actions, env.clip_actions)
    scale = _action_scale_for_logging(env, raw.device, raw.dtype)
    targets = env.custom_default_joint_pos[:, : env.num_actions] + clipped * scale

    joint_ids = torch.as_tensor(env.custom_joint_ids[: env.num_actions], device=raw.device, dtype=torch.long)
    limits = env.robot.data.soft_joint_pos_limits[:, joint_ids, :]
    lower = limits[:, :, 0]
    upper = limits[:, :, 1]
    margin = torch.minimum(targets - lower, upper - targets)
    violation = torch.relu(lower - targets) + torch.relu(targets - upper)

    env_margin = margin[env_idx]
    worst_idx = int(torch.argmin(env_margin).item())
    joint_id = int(joint_ids[worst_idx].item())
    joint_name = env.robot.data.joint_names[joint_id]
    near_count = int((env_margin < args_cli.joint_limit_margin).sum().item())
    outside_count = int((violation[env_idx] > 0.0).sum().item())

    print(
        "[action_diag] "
        f"step={step} env={env_idx} "
        f"raw_abs_max={raw[env_idx].abs().max().item():.4f} "
        f"raw_abs_mean={raw[env_idx].abs().mean().item():.4f} "
        f"delta_abs_max={delta[env_idx].abs().max().item():.4f} "
        f"delta_abs_mean={delta[env_idx].abs().mean().item():.4f} "
        f"min_limit_margin={env_margin.min().item():.4f} "
        f"near_limit_count={near_count} "
        f"outside_limit_count={outside_count} "
        f"worst_joint={joint_name} "
        f"target={targets[env_idx, worst_idx].item():.4f} "
        f"lower={lower[env_idx, worst_idx].item():.4f} "
        f"upper={upper[env_idx, worst_idx].item():.4f} "
        f"global_raw_abs_max={raw.abs().max().item():.4f} "
        f"global_delta_abs_max={delta.abs().max().item():.4f} "
        f"global_min_margin={margin.min().item():.4f}"
    )


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    # Resolve the checkpoint BEFORE building the env, so we can load the env config
    # the checkpoint was trained with (sweep runs differ in kp/kd) and apply it.
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print("------------------------------------------------------------")
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] resume_path: {resume_path}")
    print("------------------------------------------------------------")

    # Match the checkpoint's trained actuator gains (kp/kd) for a faithful eval.
    apply_trained_actuator_gains(env_cfg, log_dir)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False  # eval with zero actuation latency by default
    env_cfg.commands.curriculum.enable = False
    env_cfg.scene.max_episode_length_s = 20.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5  
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)  
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)  
    env_cfg.commands.ranges.heading = (0.0, 0.0)  
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)  

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env, continuous_torso=args_cli.continuous_torso)

    obs, _ = env.get_observations()
    step_count = 0
    prev_actions = None

    cmd_min = torch.tensor([-1.5, -1.5, -2.57, 0.3, -0.97, -0.9, -3.17], device=env.device)
    cmd_max = torch.tensor([ 1.5,  1.5,  2.57, 0.9,  0.97,  1.27,  3.17], device=env.device)

    while simulation_app.is_running():

        with torch.inference_mode():
            if not args_cli.headless:
                keyboard.advance()
                cmd = torch.tensor([keyboard.cmd_x, keyboard.cmd_y, keyboard.cmd_z, keyboard.hight, keyboard.body_roll, keyboard.body_pitch, keyboard.body_yaw], device=obs.device, dtype=obs.dtype)
                obs[0][6:13] = torch.clamp(cmd, cmd_min, cmd_max)
            # print(obs.shape)
            actions = policy(obs)
            _log_action_diagnostics(env, step_count, actions, prev_actions)
            prev_actions = actions.detach().clone()
            # print("actions:",actions)
            # joints_pos = env.scene["robot"].data.joint_pos[0, 8]
            # torque = env.scene["robot"].data.computed_torque[0, 8]
            # print("joints_pos:",joints_pos)
            # print('torque yaw:',env.scene["robot"].data.computed_torque[0, 2])
            # print('torque roll:',env.scene["robot"].data.computed_torque[0, 5])
            # print('torque pitch:',env.scene["robot"].data.computed_torque[0, 8])
            obs, _, _, _ = env.step(actions)
            step_count += 1

if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        play()
    except KeyboardInterrupt:
        print("Play interrupted by user.")
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
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
from typing import Sequence

# for logs
from datetime import datetime

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained student policy in simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Student policy play arguments
parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to trained student checkpoint")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum play steps (default: 10000)")
parser.add_argument("--use_jit", action="store_true", help="Load as JIT model instead of PT checkpoint")
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
    "--log_obs",
    action="store_true",
    help="Write detailed observation history logs next to the checkpoint (disabled by default).",
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

from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import apply_trained_actuator_gains, update_rsl_rl_cfg


def _action_scale_for_logging(env, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(env.action_scale):
        return env.action_scale[: env.num_actions].to(device=device, dtype=dtype)
    return torch.full((env.num_actions,), float(env.action_scale), device=device, dtype=dtype)


def _log_action_diagnostics(
    env,
    step: int,
    actions: torch.Tensor,
    prev_actions: torch.Tensor | None,
) -> None:
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


def _should_log_action_step(step: int) -> bool:
    interval = args_cli.action_log_interval
    return interval > 0 and (step < args_cli.action_log_warmup_steps or step % interval == 0)


def _log_joint_torque(env, step: int) -> None:
    if not _should_log_action_step(step):
        return

    env_idx = min(max(args_cli.action_log_env, 0), env.num_envs - 1)
    names = env.robot.data.joint_names
    left_ankle_pitch_idx = names.index("left_ankle_pitch_joint")
    right_ankle_pitch_idx = names.index("right_ankle_pitch_joint")
    left_ankle_roll_idx = names.index("left_ankle_roll_joint")
    right_ankle_roll_idx = names.index("right_ankle_roll_joint")
    left_hip_pitch_idx = names.index("left_hip_pitch_joint")
    right_hip_pitch_idx = names.index("right_hip_pitch_joint")
    left_hip_roll_idx = names.index("left_hip_roll_joint")
    right_hip_roll_idx = names.index("right_hip_roll_joint")
    waist_pitch_idx = names.index("waist_pitch_joint")
    torque = env.robot.data.computed_torque.detach()
    print(
        "[joint_torque] "
        f"step={step} env={env_idx} "
        f"left_ankle_pitch={torque[env_idx, left_ankle_pitch_idx].item():+.4f} "
        f"right_ankle_pitch={torque[env_idx, right_ankle_pitch_idx].item():+.4f} "
        f"left_ankle_roll={torque[env_idx, left_ankle_roll_idx].item():+.4f} "
        f"right_ankle_roll={torque[env_idx, right_ankle_roll_idx].item():+.4f} "
        f"left_hip_pitch={torque[env_idx, left_hip_pitch_idx].item():+.4f} "
        f"right_hip_pitch={torque[env_idx, right_hip_pitch_idx].item():+.4f} "
        f"left_hip_roll={torque[env_idx, left_hip_roll_idx].item():+.4f} "
        f"right_hip_roll={torque[env_idx, right_hip_roll_idx].item():+.4f} "
        f"waist_pitch={torque[env_idx, waist_pitch_idx].item():+.4f} "
        f"left_ankle_pitch_abs_max={torque[:, left_ankle_pitch_idx].abs().max().item():.4f} "
        f"right_ankle_pitch_abs_max={torque[:, right_ankle_pitch_idx].abs().max().item():.4f} "
        f"left_ankle_roll_abs_max={torque[:, left_ankle_roll_idx].abs().max().item():.4f} "
        f"right_ankle_roll_abs_max={torque[:, right_ankle_roll_idx].abs().max().item():.4f} "
        f"left_hip_pitch_abs_max={torque[:, left_hip_pitch_idx].abs().max().item():.4f} "
        f"right_hip_pitch_abs_max={torque[:, right_hip_pitch_idx].abs().max().item():.4f} "
        f"left_hip_roll_abs_max={torque[:, left_hip_roll_idx].abs().max().item():.4f} "
        f"right_hip_roll_abs_max={torque[:, right_hip_roll_idx].abs().max().item():.4f} "
        f"waist_pitch_abs_max={torque[:, waist_pitch_idx].abs().max().item():.4f}"
    )


class StudentMLP(nn.Module):
    """Simple MLP student policy """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        act_cls = nn.ELU if activation.lower() == "elu" else nn.ReLU

        # Build MLP layers
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(act_cls())
            in_dim = dim

        # Output layer
        layers.append(nn.Linear(in_dim, act_dim))

        self.policy = nn.Sequential(*layers)

        print(f"Student MLP Policy: {self.policy}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            obs: [batch_size, obs_dim] where obs_dim = history_len * base_obs_dim
                 OR [batch_size, history_len, base_obs_dim] (will be flattened)

        Returns:
            actions: [batch_size, act_dim]
        """
        # If 3D input [batch, seq, dim], flatten the history dimension
        # This allows MLP to process entire history at once
        if len(obs.shape) == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs = obs.reshape(batch_size, seq_len * obs_dim)  # [batch, history_len*58]

        # Apply MLP (no tanh - outputs raw actions like teacher)
        actions = self.policy(obs)  # [batch, act_dim]

        return actions

class ObservationHistoryBuffer:
    """Buffer to store observation history for student policy"""
    def __init__(self, history_length, obs_dim, num_envs, device):
        self.history_length = history_length
        self.obs_dim = obs_dim  # 58
        self.num_envs = num_envs
        self.device = device

        # Buffer: [num_envs, history_length, obs_dim]
        # Initially filled with zeros (like action buffer at reset)
        self.buffer = torch.zeros(num_envs, history_length, obs_dim, device=device, dtype=torch.float32)

    def update(self, new_obs):
        """
        Update buffer with new observations
        new_obs: [num_envs, obs_dim]
        Shifts buffer left and adds new observation at the end
        """
        # Shift left: move [1:] to [:-1]
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :].clone()
        # Add new observation at the end
        self.buffer[:, -1, :] = new_obs

    def reset(self, env_ids):
        """Reset specific environments to zeros (called after reset)"""
        if len(env_ids) > 0:
            self.buffer[env_ids] = 0.0

    def get_history_flat(self):
        """Return flattened history [num_envs, history_length * obs_dim]"""
        return self.buffer.reshape(self.num_envs, -1)

    def get_history(self):
        """Return history [num_envs, history_length, obs_dim]"""
        return self.buffer


def write_observation_history_log(log_file, step, obs_history, actions, env_idx=0):
    """
    Write detailed observation history and actions to log file.

    Args:
        log_file: File handle to write to
        step: Current simulation step
        obs_history: Tensor of shape [num_envs, history_len, 58] containing observation history
        actions: Tensor of shape [num_envs, 15] containing actions
        env_idx: Which environment to log (default: 0)

    Observation structure (58 dims):
        [0:3]   ang_vel (3)
        [3:6]   projected_gravity (3)
        [6:13]  command (7): lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        [13:28] joint_pos (15)
        [28:43] joint_vel (15)
        [43:58] last_actions (15)
    """
    log_file.write(f"\n{'='*100}\n")
    log_file.write(f"OBSERVATION AT STEP {step} (Environment {env_idx})\n")
    log_file.write(f"{'='*100}\n\n")

    # Get data for the specified environment and convert to CPU numpy for readability
    obs_hist = obs_history[env_idx].cpu().numpy()  # [history_len, 58]
    acts = actions[env_idx].cpu().numpy()  # [15]
    history_len = obs_hist.shape[0]

    # Write all history steps
    for hist_idx in range(history_len):
        log_file.write(f"--- HISTORY {hist_idx + 1} ---\n")

        # Angular velocity [0:3]
        ang_vel = obs_hist[hist_idx, 0:3]
        log_file.write(f"  ang_vel          = [{ang_vel[0]:8.4f}, {ang_vel[1]:8.4f}, {ang_vel[2]:8.4f}]\n")

        # Projected gravity [3:6]
        proj_grav = obs_hist[hist_idx, 3:6]
        log_file.write(f"  projected_gravity = [{proj_grav[0]:8.4f}, {proj_grav[1]:8.4f}, {proj_grav[2]:8.4f}]\n")

        # Command [6:13]: lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        cmd = obs_hist[hist_idx, 6:13]
        log_file.write(f"  command          = [lin_vel_x:{cmd[0]:7.3f}, lin_vel_y:{cmd[1]:7.3f}, ang_vel_z:{cmd[2]:7.3f}, "
                      f"height:{cmd[3]:6.3f}, body_roll:{cmd[4]:7.3f}, body_pitch:{cmd[5]:7.3f}, body_yaw:{cmd[6]:7.3f}]\n")

        # Joint positions [13:28]
        joint_pos = obs_hist[hist_idx, 13:28]
        log_file.write(f"  joint_pos        = [")
        for i in range(15):
            log_file.write(f"{joint_pos[i]:7.3f}")
            if i < 14:
                log_file.write(", ")
        log_file.write("]\n")

        # Joint velocities [28:43]
        joint_vel = obs_hist[hist_idx, 28:43]
        log_file.write(f"  joint_vel        = [")
        for i in range(15):
            log_file.write(f"{joint_vel[i]:7.3f}")
            if i < 14:
                log_file.write(", ")
        log_file.write("]\n")

        # Last actions [43:58]
        last_act = obs_hist[hist_idx, 43:58]
        log_file.write(f"  last_actions     = [")
        for i in range(15):
            log_file.write(f"{last_act[i]:7.3f}")
            if i < 14:
                log_file.write(", ")
        log_file.write("]\n\n")

    # Write current actions
    log_file.write(f"{'='*100}\n")
    log_file.write(f"ACTION AT STEP {step}\n")
    log_file.write(f"{'='*100}\n")
    log_file.write(f"  actions = [")
    for i in range(15):
        log_file.write(f"{acts[i]:7.3f}")
        if i < 14:
            log_file.write(", ")
    log_file.write("]\n\n")

    log_file.flush()  # Ensure data is written immediately


def play():
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

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

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    # Match the actuator gains (kp/kd) the student was distilled under. train_student.py
    # distills in the teacher's env (apply_trained_actuator_gains), but the student
    # checkpoint itself doesn't store the gains -- so resolve the teacher run via
    # --load_run/--checkpoint (same as training) and apply its saved gains here. Without
    # this, eval runs the default gains and the PD closed-loop won't match distillation.
    # Gate on the EXPLICIT CLI arg, not agent_cfg.load_run: the latter defaults to ".*"
    # (never None), which would otherwise silently resolve to the most recent run and
    # apply the wrong gains.
    if args_cli.load_run is not None:
        log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
        teacher_dir = os.path.dirname(
            get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
        print(f"[INFO] Applying teacher gains from: {teacher_dir}")
        apply_trained_actuator_gains(env_cfg, teacher_dir)
    else:
        print(
            "[WARN] --load_run not set; playing with DEFAULT actuator gains, which may "
            "not match the gains the student was distilled under. Pass --load_run=<teacher_run> "
            "(and --checkpoint) to load the correct kp/kd."
        )

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env, continuous_torso=args_cli.continuous_torso)

    obs, _ = env.get_observations()

    # Initialize observation history buffer
    history_length = 2  # Number of past observations to use
    obs_history_buffer = ObservationHistoryBuffer(
        history_length=history_length,
        obs_dim=58,
        num_envs=env_cfg.scene.num_envs,
        device=env.device
    )

    # Load trained student checkpoint or JIT model
    checkpoint_path = args_cli.student_checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Student checkpoint not found at: {checkpoint_path}")

    # Define student policy parameters (same for both JIT and PT)
    student_obs_dim = history_length * 58  # Flattened history
    student_action_dim = 15    # Same as teacher action dim (legs + waist)
    student_hidden_dims = [512, 256, 128]
    student_activation = "elu"

    print(f"\n{'='*50}")

    if args_cli.use_jit:
        # Load JIT model directly (same as deploy_student.py)
        print(f"Loading JIT model from: {checkpoint_path}")
        student_net = torch.jit.load(checkpoint_path, map_location=env.device)
        student_net.eval()
        print(f"✅ JIT model loaded successfully")
    else:
        # Load PT checkpoint (original behavior)
        print(f"Loading PT checkpoint from: {checkpoint_path}")

        # Initialize student policy (MLP with history input)
        student_net = StudentMLP(
            obs_dim=student_obs_dim,
            act_dim=student_action_dim,
            hidden_dims=student_hidden_dims,
            activation=student_activation,
        ).to(env.device)

        checkpoint = torch.load(checkpoint_path, map_location=env.device)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        student_net.eval()

        print(f"Loaded model from checkpoint step: {checkpoint.get('step', 'N/A')}")
        print(f"Training loss at checkpoint: {checkpoint.get('loss', 'N/A')}")

    print(f"{'='*50}\n")

    # Play configuration
    max_steps = args_cli.max_steps
    current_step = 0
    prev_actions_student = None
    waist_pitch_joint_id = env.robot.data.joint_names.index("waist_pitch_joint")

    print(f"Student Policy Play Configuration:")
    print(f"  - Input dim: {student_obs_dim} ({history_length} steps x 58 obs)")
    print(f"  - Output dim: {student_action_dim}")
    print(f"  - Hidden dims: {student_hidden_dims}")
    print(f"  - Activation: {student_activation}")
    print(f"  - Max play steps: {max_steps}")
    print(f"  - Num environments: {env_cfg.scene.num_envs}")
    print(f"  - Mode: Play (no training)\n")

    cmd_min = torch.tensor([-1.5, -1.5, -2.57, 0.3, -0.97, -0.9, -3.17], device=env.device)
    cmd_max = torch.tensor([ 1.5,  1.5,  2.57, 0.9,  0.97,  1.27,  3.17], device=env.device)

    # Optionally write detailed observation history logs (--log_obs to enable)
    log_file = None
    if args_cli.log_obs:
        log_dir = os.path.join(os.path.dirname(checkpoint_path), "observation_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"obs_history_{timestamp}.log")
        log_file = open(log_path, 'w')
        print(f"Logging observation history to: {log_path}\n")

    while simulation_app.is_running() and current_step < max_steps:

        # Override keyboard commands (only in non-headless mode)
        if not args_cli.headless:
            keyboard.advance()
            with torch.inference_mode():
                cmd = torch.tensor(
                    [
                        keyboard.cmd_x,
                        keyboard.cmd_y,
                        keyboard.cmd_z,
                        keyboard.hight,
                        keyboard.body_roll,
                        keyboard.body_pitch,
                        keyboard.body_yaw,
                    ],
                    device=obs.device,
                    dtype=obs.dtype,
                )
                obs[0][6:13] = torch.clamp(cmd, cmd_min, cmd_max)
            
        # Update observation history buffer with current observation, then read it.
        # History includes the current obs as the most recent entry (matches deployment).
        obs_without_contact = obs[:, :58]
        obs_history_buffer.update(obs_without_contact)
        obs_history = obs_history_buffer.get_history()  # Shape: [num_envs, history_len, 58]

        # Get student actions (no gradients - play mode)
        # Student uses PAST observations (history) to predict current action
        with torch.inference_mode():
            actions_student = student_net(obs_history)  # Shape: [num_envs, 15]
            # _log_action_diagnostics(env, current_step, actions_student, prev_actions_student)
            prev_actions_student = actions_student.detach().clone()

        if log_file is not None and (current_step == 0 or current_step % 1000 == 1 or current_step == 26):
            write_observation_history_log(log_file, current_step, obs_history, actions_student, env_idx=0)
            print(f"[Step {current_step}] Logged observation history to file")

        # Increment step counter
        policy_step = current_step
        current_step += 1

        # Print progress every 1000 steps
        if current_step % 1000 == 0:
            print(f"[Step {current_step}/{max_steps}] Play in progress...")
            
        # Step environment with student actions
        with torch.inference_mode():
            # print("actions_student:", actions_student)
            obs, _, reset_buf, _ = env.step(actions_student)
            _log_joint_torque(env, policy_step)

            # Real-time waist pitch: commanded target vs measured joint position (env 0).
            # waist_pitch is index 14 in the custom action order.
            # waist_target = (
            #     env.custom_default_joint_pos[0, 14]
            #     + torch.clamp(actions_student[0, 14], -env.clip_actions, env.clip_actions) * env.action_scale
            # ).item()
            # waist_measured = env.robot.data.joint_pos[0, waist_pitch_joint_id].item()
            # print(f"[waist_pitch] step={policy_step} target={waist_target:+.4f} measured={waist_measured:+.4f}")

            # Reset history buffer for environments that were reset
            reset_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
            # print("reset_env_ids:", reset_env_ids)
            if len(reset_env_ids) > 0:
                obs_history_buffer.reset(reset_env_ids)


    if log_file is not None:
        log_file.close()

    print(f"\n{'='*50}")
    print(f"Play completed! Total steps: {current_step}")
    if log_file is not None:
        print(f"Observation history log saved to: {log_path}")
    print(f"{'='*50}")


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
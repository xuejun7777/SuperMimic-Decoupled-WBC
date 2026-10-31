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
import glob
import hashlib
import os
from pathlib import Path
import re
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an IL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate (default: 4096).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Student policy training arguments
parser.add_argument("--student_lr", type=float, default=5e-4, help="Learning rate for student policy (default: 5e-4)")
parser.add_argument("--max_steps", type=int, default=250000, help="Maximum training steps (absolute target; default: 250000)")
parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a student checkpoint to resume training from")
parser.add_argument(
    "--auto_resume",
    action="store_true",
    help=(
        "Auto-discover and resume the latest student checkpoint in the run's "
        "student_checkpoints/ folder. Intended for preemptible cluster jobs: a requeued "
        "job continues its own student training, and --max_steps is the absolute target."
    ),
)
parser.add_argument(
    "--bc_steps",
    type=int,
    default=0,
    help=(
        "If >0, run a self-orchestrating two-stage pipeline: Behavior Cloning for the "
        "first --bc_steps steps, then DAgger up to --max_steps. Overrides --IL_type. "
        "Survives preemption via --auto_resume (the phase is derived from the step)."
    ),
)
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N steps (default: 2000)")
parser.add_argument("--keep_checkpoints", type=int, default=3, help="Number of checkpoints to keep (default: 3)")
parser.add_argument("--IL_type", type=str, default="bc", choices=["bc", "dagger"], help="Imitation learning type: 'bc' (Behavior Cloning) or 'dagger' (DAgger) (default: bc)")
parser.add_argument("--wandb_run_version", type=str, default="v1", help="W&B run name suffix. Bump (e.g. v1->v2) to start a fresh W&B run for the same teacher run (default: v1)")

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
from legged_lab.utils.cli_args import (
    apply_trained_actuator_gains,
    apply_trained_command_config,
    update_rsl_rl_cfg,
)


def _student_checkpoint_step(path: str) -> int:
    match = re.search(r"student_step_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _student_checkpoints(checkpoint_dir: str) -> list[str]:
    ckpts = glob.glob(os.path.join(checkpoint_dir, "student_step_*.pt"))
    return sorted(ckpts, key=_student_checkpoint_step)


def _latest_student_checkpoint(checkpoint_dir: str):
    """Return the highest-step ``student_step_<N>.pt`` in ``checkpoint_dir``, or None."""
    checkpoints = _student_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_exact_teacher_checkpoint(log_root_path: str, load_run: str, checkpoint: str) -> str:
    """Resolve an explicitly selected teacher without regexes or path traversal."""
    if Path(load_run).name != load_run or Path(checkpoint).name != checkpoint:
        raise ValueError("--load_run and --checkpoint must each be a single path component")

    root = Path(log_root_path).resolve()
    run_dir = root / load_run
    candidate = run_dir / checkpoint
    try:
        resolved_run = run_dir.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Exact teacher checkpoint not found: {candidate}") from exc
    if not resolved_run.is_relative_to(root) or not resolved_run.is_dir():
        raise ValueError(f"Teacher run must resolve inside {root}: {run_dir} -> {resolved_run}")
    if not resolved.is_relative_to(resolved_run) or not resolved.is_file():
        raise ValueError(f"Teacher checkpoint must resolve inside {root}: {candidate} -> {resolved}")
    # Keep the selected run path (rather than the symlink target) as log_dir so
    # outputs and params stay scoped to the explicitly selected run folder.
    return str(candidate)


def _prepare_student_output(log_dir: str, resume_path: str) -> tuple[str, dict[str, str]]:
    """Bind all student outputs/resumes in ``log_dir`` to one exact teacher."""
    run_dir = Path(log_dir)
    params_path = run_dir / "params" / "env.yaml"
    if not params_path.is_file():
        raise FileNotFoundError(f"Teacher environment config missing: {params_path}")

    source = {
        "teacher_run": run_dir.name,
        "teacher_checkpoint": Path(resume_path).name,
        "teacher_checkpoint_sha256": _sha256(Path(resume_path)),
        "env_yaml_sha256": _sha256(params_path),
    }
    source_id = " ".join(source.values())
    checkpoint_dir = run_dir / "student_checkpoints"
    marker = checkpoint_dir / "teacher_source.txt"
    existing = _student_checkpoints(str(checkpoint_dir))

    if marker.is_file():
        stored = marker.read_text().strip()
        if stored != source_id:
            raise RuntimeError(
                f"Student teacher mismatch in {marker}\n"
                f"  stored:  {stored}\n"
                f"  current: {source_id}"
            )
    elif existing:
        raise RuntimeError(
            f"Existing student checkpoints have no teacher-source marker: {checkpoint_dir}. "
            "Inspect or move that directory before training."
        )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(source_id + "\n")

    if existing and not (args_cli.auto_resume or args_cli.resume_checkpoint):
        raise RuntimeError(
            f"Student checkpoints already exist in {checkpoint_dir}; use --auto_resume, "
            "pass --resume_checkpoint, or move them to start a fresh student."
        )

    print(f"[INFO] Student teacher source: {source_id}")
    return str(checkpoint_dir), source


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


def print_observation_history(step, obs_history, actions_student, actions_teacher, IL_type, env_idx=0):
    """
    Print detailed observation history and actions for debugging.

    Args:
        step: Current training step
        obs_history: Observation history tensor of shape [num_envs, history_len, 58]
        actions_student: Student policy actions of shape [num_envs, 15]
        actions_teacher: Teacher policy actions of shape [num_envs, 15]
        IL_type: Imitation learning type ("bc" or "dagger")
        env_idx: Environment index to print (default: 0)

    Observation structure (58 dims):
        [0:3]   ang_vel (3)
        [3:6]   projected_gravity (3)
        [6:13]  command (7): lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        [13:28] joint_pos (15)
        [28:43] joint_vel (15)
        [43:58] last_actions (15)
    """
    print(f"\n{'='*100}")
    print(f"OBSERVATION AT STEP {step} (Environment {env_idx})")
    print(f"{'='*100}\n")

    # Get data for the specified environment and convert to CPU numpy for readability
    obs_hist = obs_history[env_idx].detach().cpu().numpy()  # [history_len, 58]
    acts_student = actions_student[env_idx].detach().cpu().numpy()  # [15]
    acts_teacher = actions_teacher[env_idx].detach().cpu().numpy()  # [15]
    history_len = obs_hist.shape[0]

    # Print all history steps
    for hist_idx in range(history_len):
        print(f"--- HISTORY {hist_idx + 1} ---")

        # Angular velocity [0:3]
        ang_vel = obs_hist[hist_idx, 0:3]
        print(f"  ang_vel          = [{ang_vel[0]:8.4f}, {ang_vel[1]:8.4f}, {ang_vel[2]:8.4f}]")

        # Projected gravity [3:6]
        proj_grav = obs_hist[hist_idx, 3:6]
        print(f"  projected_gravity = [{proj_grav[0]:8.4f}, {proj_grav[1]:8.4f}, {proj_grav[2]:8.4f}]")

        # Command [6:13]: lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        cmd = obs_hist[hist_idx, 6:13]
        print(f"  command          = [lin_vel_x:{cmd[0]:7.3f}, lin_vel_y:{cmd[1]:7.3f}, ang_vel_z:{cmd[2]:7.3f}, "
              f"height:{cmd[3]:6.3f}, body_roll:{cmd[4]:7.3f}, body_pitch:{cmd[5]:7.3f}, body_yaw:{cmd[6]:7.3f}]")

        # Joint positions [13:28]
        joint_pos = obs_hist[hist_idx, 13:28]
        print(f"  joint_pos        = [", end="")
        for i in range(15):
            print(f"{joint_pos[i]:7.3f}", end="")
            if i < 14:
                print(", ", end="")
        print("]")

        # Joint velocities [28:43]
        joint_vel = obs_hist[hist_idx, 28:43]
        print(f"  joint_vel        = [", end="")
        for i in range(15):
            print(f"{joint_vel[i]:7.3f}", end="")
            if i < 14:
                print(", ", end="")
        print("]")

        # Last actions [43:58]
        last_act = obs_hist[hist_idx, 43:58]
        print(f"  last_actions     = [", end="")
        for i in range(15):
            print(f"{last_act[i]:7.3f}", end="")
            if i < 14:
                print(", ", end="")
        print("]\n")

    # Print actions based on IL type
    # BC: print teacher actions (what will be executed)
    # DAgger: print student actions (what will be executed)
    print(f"{'='*100}")
    if IL_type == "bc":
        print(f"ACTION AT STEP {step} (Teacher - BC mode)")
        print(f"{'='*100}")
        print(f"  teacher_actions = [", end="")
        for i in range(15):
            print(f"{acts_teacher[i]:7.3f}", end="")
            if i < 14:
                print(", ", end="")
        print("]\n")
    else:  # dagger
        print(f"ACTION AT STEP {step} (Student - DAgger mode)")
        print(f"{'='*100}")
        print(f"  student_actions = [", end="")
        for i in range(15):
            print(f"{acts_student[i]:7.3f}", end="")
            if i < 14:
                print(", ", end="")
        print("]\n")


def train():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    # Resolve the teacher checkpoint BEFORE building the env, so we can load the env
    # config it was trained with (gains differ per run) and apply it.
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print("------------------------------------------------------------")
    print(f"[INFO] Loading teacher experiment from directory: {log_root_path}")
    # When both values were supplied explicitly, require that exact path. The
    # Isaac Lab helper treats them as unanchored regular expressions and may
    # otherwise select a suffix-copy of the run/checkpoint after the shell has
    # verified a different file's hash.
    if args_cli.load_run is not None and args_cli.checkpoint is not None:
        resume_path = _resolve_exact_teacher_checkpoint(log_root_path, args_cli.load_run, args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Teacher checkpoint: {resume_path}")
    print("------------------------------------------------------------")

    if args_cli.overrides:
        raise ValueError(
            "--set overrides are not supported for student distillation. "
            "The environment is restored from the teacher's saved params/env.yaml."
        )

    # Distill in the SAME dynamics and command distribution used to train the teacher.
    apply_trained_actuator_gains(env_cfg, log_dir, strict=True)
    apply_trained_command_config(env_cfg, log_dir, strict=True)

    checkpoint_dir, teacher_source = _prepare_student_output(log_dir, resume_path)

    # Disable the command curriculum so the student is distilled over the teacher's
    # saved FULL command ranges (commands.ranges.*) from step 0. Otherwise the curriculum
    # would restart at iteration 0 here and the student would only ever see the narrow
    # start-of-curriculum ranges (the student run is far shorter than the teacher's).
    env_cfg.commands.curriculum.enable = False

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

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    data = []

    # Initialize observation history buffer (history_length steps x 58 dims)
    history_length = 2
    student_obs_dim = 58

    obs_history_buffer = ObservationHistoryBuffer(
        history_length=history_length,
        obs_dim=student_obs_dim,
        num_envs=env_cfg.scene.num_envs,
        device=env.device
    )

    # Initialize student policy (MLP with history input)
    student_obs_dim = history_length * student_obs_dim  # 116: flattened history (history_length * 58)
    student_action_dim = 15    # Same as teacher action dim (legs + waist)
    student_hidden_dims = [512, 256, 128]  # From G1StudentPPORunnerCfg
    student_activation = "elu"
    student_lr = args_cli.student_lr  # From command line args

    student_net = StudentMLP(
        obs_dim=student_obs_dim,
        act_dim=student_action_dim,
        hidden_dims=student_hidden_dims,
        activation=student_activation,
    ).to(env.device)
    student_net.train()

    # Initialize optimizer for student policy
    optimizer = torch.optim.Adam(student_net.parameters(), lr=student_lr)

    # Training configuration
    max_steps = args_cli.max_steps
    current_step = 0
    save_interval = args_cli.save_interval
    keep_checkpoints = args_cli.keep_checkpoints
    IL_type = args_cli.IL_type
    # Two-stage pipeline: BC for the first bc_steps, then DAgger up to max_steps.
    bc_steps = args_cli.bc_steps
    two_stage = bc_steps > 0

    # Setup TensorBoard logging
    tensorboard_dir = os.path.join(log_dir, "student_tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logging to: {tensorboard_dir}")

    # Optional Weights & Biases logging (--logger=wandb); mirrors the TensorBoard scalars.
    # In two-stage mode the run spans BC+DAgger; otherwise it's tagged with the IL type.
    # W&B must work or fail: init is NOT wrapped, so a bad/deleted run id raises and the
    # job exits FAILED (with a traceback) rather than silently training without wandb.
    wandb_run = None
    if args_cli.logger == "wandb":
        import wandb

        wandb_name = f"{os.path.basename(log_dir)}_student" + ("" if two_stage else f"_{IL_type}") + f"_{args_cli.wandb_run_version}"
        wandb_kwargs = dict(
            project=agent_cfg.wandb_project,
            name=wandb_name,
            dir=log_dir,
            config={
                "mode": "two_stage_bc_dagger" if two_stage else IL_type,
                "bc_steps": bc_steps,
                "student_lr": student_lr,
                "max_steps": max_steps,
                "num_envs": env_cfg.scene.num_envs,
                "hidden_dims": student_hidden_dims,
                "history_length": history_length,
                "seed": agent_cfg.seed,
                "teacher_source": teacher_source,
            },
        )
        # Deterministic id + resume so preempted/requeued cluster jobs append to the
        # same run instead of spawning a new one on every restart. NOTE: a deleted
        # W&B run id cannot be reused -- bump wandb_version above to get a fresh one.
        if args_cli.auto_resume:
            wandb_kwargs.update(id=wandb_name, resume="allow")
        wandb_run = wandb.init(**wandb_kwargs)
        print(f"Weights & Biases logging to project: {agent_cfg.wandb_project} (run: {wandb_name})")

    # Track checkpoints across process restarts so --keep_checkpoints also works
    # after a Slurm requeue or a local --auto_resume.
    saved_checkpoints = _student_checkpoints(checkpoint_dir)

    # Resolve which student checkpoint (if any) to resume from: an explicit
    # --resume_checkpoint takes precedence; otherwise --auto_resume discovers the
    # latest one in this run's student_checkpoints/ folder (preemption-friendly).
    resume_student_path = args_cli.resume_checkpoint
    if resume_student_path is None and args_cli.auto_resume:
        resume_student_path = _latest_student_checkpoint(checkpoint_dir)
        if resume_student_path:
            print(f"[INFO] Auto-resume: continuing from {resume_student_path}")
        else:
            print(f"[INFO] Auto-resume: no student checkpoint in {checkpoint_dir}; starting fresh.")

    # Resume from checkpoint if available
    if resume_student_path is not None:
        checkpoint_path = resume_student_path
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*50}")
            print(f"Loading student checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=env.device)

            saved_source = checkpoint.get("teacher_source")
            if saved_source is not None and saved_source != teacher_source:
                raise RuntimeError(
                    f"Student checkpoint teacher mismatch in {checkpoint_path}: "
                    f"{saved_source} != {teacher_source}"
                )

            # Restore model, optimizer, and the absolute step counter so training
            # continues from where it left off (--max_steps is an absolute target,
            # and checkpoint numbering / logging keep climbing).
            student_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_step = int(checkpoint.get('step', 0))

            print(f"Resumed from step: {current_step}")
            print(f"Previous loss: {checkpoint.get('loss', 'N/A')}")
            print(f"Continuing to {max_steps} total steps ({max(max_steps - current_step, 0)} remaining)")
            print(f"{'='*50}\n")
        else:
            print(f"[WARNING] Checkpoint not found at {checkpoint_path}, starting from scratch")

    print(f"Student Policy Training Configuration:")
    print(f"  - Input dim: {student_obs_dim} ({history_length} steps x 58 obs)")
    print(f"  - Output dim: {student_action_dim}")
    print(f"  - Hidden dims: {student_hidden_dims}")
    print(f"  - Activation: {student_activation}")
    print(f"  - Learning rate: {student_lr}")
    print(f"  - Max training steps: {max_steps}")
    print(f"  - Starting step: {current_step}")
    print(f"  - Num environments: {env_cfg.scene.num_envs}")
    if two_stage:
        print(f"  - Mode: TWO-STAGE (BC for steps < {bc_steps}, then DAgger up to {max_steps})")
    else:
        print(f"  - IL Type: {IL_type.upper()} {'(Behavior Cloning - using teacher actions)' if IL_type == 'bc' else '(DAgger - using student actions)'}")

    while simulation_app.is_running() and current_step < max_steps:

        # Override keyboard commands (only in non-headless mode)
        with torch.inference_mode():
            if not args_cli.headless:
                obs[0][6:13] = torch.tensor([keyboard.cmd_x, keyboard.cmd_y, keyboard.cmd_z, keyboard.hight, keyboard.body_roll, keyboard.body_pitch, keyboard.body_yaw], device=obs.device, dtype=obs.dtype)

        # Update observation history buffer with current observation, then read it.
        # History includes the current obs as the most recent entry (matches deployment).
        obs_without_contact = obs[:, :58]
        obs_history_buffer.update(obs_without_contact)

        obs_history = obs_history_buffer.get_history()  # Shape: [num_envs, history_len, 58]

        # Get teacher actions (no gradients needed, but must be detachable for loss computation)
        with torch.no_grad():
            actions_teacher = policy(obs)  # Shape: [num_envs, 15]

        # Get student actions (with gradients for training)
        # Student uses PAST observations (history) to predict current action
        actions_student = student_net(obs_history)  # Shape: [num_envs, 15]

        # Compute RMSE loss between student and teacher (Root Mean Squared Error)
        loss = torch.sqrt(nn.functional.mse_loss(actions_student, actions_teacher))

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Print detailed observation history for debugging
        # if current_step == 0 or current_step % 1000 == 1:
        #     print_observation_history(current_step, obs_history, actions_student, actions_teacher, IL_type, env_idx=0)

        # Increment step counter
        current_step += 1

        # Log RMSE loss to TensorBoard (and wandb if enabled)
        writer.add_scalar('Loss/RMSE', loss.item(), current_step)
        if wandb_run is not None:
            wandb_run.log({"Loss/RMSE": loss.item()}, step=current_step)

        # Print loss every 1000 steps
        if current_step % 1000 == 0:
            print(f"[Step {current_step}] RMSE Loss: {loss.item():.6f}")

        # Save checkpoint periodically
        if current_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"student_step_{current_step}.pt")

            torch.save({
                'step': current_step,
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'teacher_source': teacher_source,
                'config': {
                    'obs_dim': student_obs_dim,
                    'act_dim': student_action_dim,
                    'hidden_dims': student_hidden_dims,
                    'activation': student_activation,
                    'learning_rate': student_lr,
                }
            }, checkpoint_path)

            if checkpoint_path in saved_checkpoints:
                saved_checkpoints.remove(checkpoint_path)
            saved_checkpoints.append(checkpoint_path)
            print(f"\n[Step {current_step}] Checkpoint saved: {checkpoint_path}")
            print(f"[Step {current_step}] Current loss: {loss.item():.6f}")

            # Keep only the most recent N checkpoints
            while len(saved_checkpoints) > keep_checkpoints:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"[Step {current_step}] Removed old checkpoint: {oldest_checkpoint}")

        # Select actions for environment rollout based on the (possibly phased) IL type.
        # Two-stage: BC (teacher actions) while below bc_steps, then DAgger (student actions).
        if two_stage:
            il_now = "bc" if current_step <= bc_steps else "dagger"
        else:
            il_now = IL_type
        if il_now == "bc":
            # Behavior Cloning: use teacher actions (expert demonstrations)
            actions_for_step = actions_teacher
        else:  # dagger
            # DAgger: use student actions (policy under training)
            actions_for_step = actions_student
            
        # Step environment
        with torch.inference_mode():
            obs, _, reset_buf, _ = env.step(actions_for_step)

            # Reset history buffer for environments that were reset
            # (just like action buffer gets reset)
            reset_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
            if len(reset_env_ids) > 0:
                obs_history_buffer.reset(reset_env_ids)


    # Training complete
    print(f"\n{'='*50}")
    print(f"Training completed! Total steps: {current_step}")
    print(f"{'='*50}")

    # Save final checkpoint (if not just saved)
    if current_step % save_interval != 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"student_step_{current_step}.pt")

        torch.save({
            'step': current_step,
            'model_state_dict': student_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item() if 'loss' in locals() else None,
            'teacher_source': teacher_source,
            'config': {
                'obs_dim': student_obs_dim,
                'act_dim': student_action_dim,
                'hidden_dims': student_hidden_dims,
                'activation': student_activation,
                'learning_rate': student_lr,
            }
        }, checkpoint_path)

        if checkpoint_path in saved_checkpoints:
            saved_checkpoints.remove(checkpoint_path)
        saved_checkpoints.append(checkpoint_path)
        print(f"\nFinal checkpoint saved to: {checkpoint_path}")

    # Export JIT model for deployment
    print(f"\nExporting JIT model for deployment...")
    student_net.eval()  # Set to evaluation mode

    # Create example input for JIT tracing (student_obs_dim already includes history: 2 * 58 = 116)
    example_input = torch.randn(1, student_obs_dim, device=env.device)

    try:
        # Export using TorchScript (script mode)
        jit_model = torch.jit.script(student_net)

        # Save JIT model in the checkpoint directory. Copy it to deploy/ manually
        # when you're ready to deploy (see docs/deployment.md).
        jit_model_path = os.path.join(checkpoint_dir, "student_policy_jit.pt")
        torch.jit.save(jit_model, jit_model_path)
        print(f"✅ JIT model exported to: {jit_model_path}")

    except Exception as e:
        print(f"⚠️  JIT export failed: {e}")
        print(f"   This is not critical - you can still use the state_dict checkpoint for deployment.")

    student_net.train()  # Set back to training mode (in case needed)

    # Keep only the most recent N checkpoints (from final save)
    if current_step % save_interval != 0:
        while len(saved_checkpoints) > keep_checkpoints:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")

    print(f"\nKept checkpoints: {saved_checkpoints}")

    # Close loggers
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"TensorBoard logs saved. View with: tensorboard --logdir={tensorboard_dir}")

if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        train()
        print("\n✅ Training completed.")
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user.")
        exit_code = 130
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Error: {e}")
        exit_code = 1
    finally:
        # simulation_app.close() can hang on some Isaac Sim setups; force-exit so the
        # process terminates cleanly. Flush first so buffered logs (incl. the traceback
        # above) aren't lost, and preserve a NONZERO code on failure so Slurm reports
        # FAILED instead of masking a crash as COMPLETED.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

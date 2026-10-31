"""Train a Reference-Free SuperMimic Teacher Policy on Unitree G1 Decoupled WBC.

CLI-compatible drop-in counterpart to `legged_lab/scripts/train_teacher.py`:
  - Uses the exact same task registry, `--num_envs`, `--seed`, `--max_iterations`,
    `--save_interval`, `--run_name`, `--set KEY=VALUE`, `--resume`, and `--auto_resume`.
  - Exports RSL-RL-compatible `model_<iter>.pt` + `params/env.yaml` so `play_teacher.py`,
    `train_student.py`, and `eval_compare.py` work out of the box without changing a line.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a Reference-Free SuperMimic agent on G1 WBC.")
parser.add_argument("--task", type=str, default="g1_flat", help="Name of the task (e.g. g1_flat).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N outer iterations.")
parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N outer iterations.")
parser.add_argument("--auto_resume", action="store_true", help="Deterministic run folder + auto-resume.")

# SuperMimic algorithm hyperparameters
parser.add_argument("--utd", type=float, default=0.4, help="WM+Policy updates per simulator step.")
parser.add_argument("--horizon", type=int, default=24, help="Analytic BPTT rollout horizon H.")
parser.add_argument("--imag_batch", type=int, default=2048, help="Batch size for imagined BPTT rollouts.")
parser.add_argument("--wm_k", type=int, default=8, help="Autoregressive window length k for WM fitting.")
parser.add_argument("--wm_batch", type=int, default=2048, help="Batch size for WM k-step windows.")
parser.add_argument("--wm_batches", type=int, default=8, help="WM gradient steps per update.")
parser.add_argument("--wm_lr", type=float, default=3.0e-4, help="World Model learning rate.")
parser.add_argument("--actor_lr", type=float, default=2.0e-4, help="Actor BPTT learning rate.")
parser.add_argument("--critic_lr", type=float, default=3.0e-4, help="Privileged Critic learning rate.")
parser.add_argument("--action_noise", type=float, default=0.30, help="Initial exploration action noise std.")
parser.add_argument("--action_noise_final", type=float, default=0.15, help="Final exploration action noise std.")
parser.add_argument("--det_frac", type=float, default=0.20, help="Fraction of noise-free deterministic envs.")
parser.add_argument("--warmup_iters", type=int, default=20, help="Warmup outer iterations (counted inside max_iterations).")
parser.add_argument("--replay_capacity", type=int, default=1000, help="Per-env ring buffer capacity in steps.")
parser.add_argument(
    "--symmetry_weight",
    type=float,
    default=1.5,
    help="Target weight of phase-gated sagittal mirror-equivariance loss L_sym on Actor actions (0.0 disables L_sym).",
)
parser.add_argument(
    "--sym_warmup_ratio",
    type=float,
    default=0.45,
    help="Fraction of total iterations before ramping L_sym (allows unimpeded 7D skill & gait unzipping).",
)
parser.add_argument(
    "--sym_ramp_ratio",
    type=float,
    default=0.15,
    help="Fraction of total iterations over which L_sym half-cosine ramps from 0 to symmetry_weight.",
)
parser.add_argument(
    "--no_mirror_aug",
    action="store_true",
    help="Disable sagittal mirror batch augmentation on WM windows and imagination starts.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dataclasses import asdict
from datetime import datetime
import glob
import os
import re

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.runner import SuperMimicRunner, SuperMimicRunnerCfg
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import apply_cfg_overrides, update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _latest_checkpoint(log_dir: str) -> str | None:
    if not os.path.isdir(log_dir):
        return None
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))

    def _iter(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=_iter) if ckpts else None


def train() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.save_interval = args_cli.save_interval
    env_cfg.scene.seed = agent_cfg.seed

    apply_cfg_overrides(env_cfg, agent_cfg, args_cli.overrides)

    env = SuperMimicG1Env(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Logging SuperMimic experiment in directory: {log_root_path}", flush=True)

    resume_path = None
    if args_cli.auto_resume:
        if not agent_cfg.run_name:
            raise ValueError("--auto_resume requires --run_name.")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name)
        resume_path = _latest_checkpoint(log_dir)
        agent_cfg.resume = resume_path is not None
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            folder += f"_{agent_cfg.run_name}"
        else:
            folder += "_supermimic"
        log_dir = os.path.join(log_root_path, folder)

    sm_cfg = SuperMimicRunnerCfg(
        num_steps_per_env=int(agent_cfg.num_steps_per_env),
        utd=float(args_cli.utd),
        replay_capacity=int(args_cli.replay_capacity),
        warmup_iters=int(args_cli.warmup_iters),
        action_noise_init=float(args_cli.action_noise),
        action_noise_final=float(args_cli.action_noise_final),
        det_frac=float(args_cli.det_frac),
        log_interval=int(args_cli.log_interval),
        save_interval=int(args_cli.save_interval),
        use_tensorboard=(agent_cfg.logger in ("tensorboard", "wandb")),
    )
    use_mirror_aug = not bool(args_cli.no_mirror_aug)
    sm_cfg.wm_trainer.window_k = int(args_cli.wm_k)
    sm_cfg.wm_trainer.batch_size = int(args_cli.wm_batch)
    sm_cfg.wm_trainer.batches_per_update = int(args_cli.wm_batches)
    sm_cfg.wm_trainer.lr = float(args_cli.wm_lr)
    sm_cfg.wm_trainer.use_mirror_augmentation = use_mirror_aug

    sm_cfg.policy_trainer.horizon = int(args_cli.horizon)
    sm_cfg.policy_trainer.batch_size = int(args_cli.imag_batch)
    sm_cfg.policy_trainer.gamma = float(agent_cfg.algorithm.gamma)
    sm_cfg.policy_trainer.gae_lambda = float(agent_cfg.algorithm.lam)
    sm_cfg.policy_trainer.actor_lr = float(args_cli.actor_lr)
    sm_cfg.policy_trainer.critic_lr = float(args_cli.critic_lr)
    sm_cfg.policy_trainer.obs_noise_in_imagination = bool(env_cfg.noise.add_noise)
    sm_cfg.policy_trainer.use_mirror_augmentation = use_mirror_aug
    sm_cfg.policy_trainer.symmetry_weight = float(args_cli.symmetry_weight)
    sm_cfg.policy_trainer.sym_warmup_ratio = float(args_cli.sym_warmup_ratio)
    sm_cfg.policy_trainer.sym_ramp_ratio = float(args_cli.sym_ramp_ratio)

    sm_cfg.actor.hidden_dims = tuple(int(x) for x in agent_cfg.policy.actor_hidden_dims)
    sm_cfg.critic.hidden_dims = tuple(int(x) for x in agent_cfg.policy.critic_hidden_dims)

    runner = SuperMimicRunner(env=env, cfg=sm_cfg, log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume and resume_path is not None:
        print(f"[INFO] Resuming SuperMimic from {resume_path}", flush=True)
        runner.load(resume_path)
        env.set_curriculum_start_iteration(runner.current_learning_iteration)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "supermimic.yaml"), asdict(sm_cfg))

    if args_cli.auto_resume:
        num_iterations = max(int(agent_cfg.max_iterations) - runner.current_learning_iteration, 0)
    else:
        num_iterations = int(agent_cfg.max_iterations)

    runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        exit_code = 130
    except Exception as exc:
        traceback.print_exc()
        print(f"Error: {exc}")
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

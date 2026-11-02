"""CLI Entrypoint for World-Model Guided SuperMimic Student Training.

Trains a 116D Proprioceptive Student using the trained SuperMimic Teacher's World Model,
31-term analytic task reward, 116D sagittal mirror symmetry, and teacher action supervision.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os

from isaaclab.app import AppLauncher

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train SuperMimic Student on G1 WBC.")
parser.add_argument("--task", type=str, default="g1_flat", help="Name of task (e.g. g1_flat).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to trained SuperMimic Teacher checkpoint.")
parser.add_argument("--student_iterations", type=int, default=1650, help="Total student learning iterations.")
parser.add_argument("--bc_warmup_iters", type=int, default=50, help="Initial teacher-rollout BC iterations.")
parser.add_argument("--dagger_warmup_iters", type=int, default=100, help="DAgger-only iterations before full RL.")
parser.add_argument("--student_lr", type=float, default=3.0e-4, help="Student policy learning rate.")
parser.add_argument("--symmetry_weight", type=float, default=1.5, help="116D Sagittal symmetry loss weight.")
parser.add_argument("--teacher_anchor_init", type=float, default=2.0, help="Initial teacher action anchor weight.")
parser.add_argument("--teacher_anchor_final", type=float, default=0.25, help="Final teacher action anchor weight.")
parser.add_argument("--hindsight_cmd_frac", type=float, default=0.25, help="Coupled 7D hindsight command relabeling fraction.")
parser.add_argument("--save_interval", type=int, default=250, help="Save interval in outer iterations.")
parser.add_argument("--resume_student_ckpt", type=str, default=None, help="Path to student checkpoint to resume.")


cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.utils.io import dump_yaml

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.student_runner import SuperMimicStudentCfg, SuperMimicStudentRunner
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import apply_cfg_overrides, update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    apply_cfg_overrides(env_cfg, agent_cfg, args_cli.overrides)

    # Disable command curriculum for Student: train over full 7D command range from step 0
    env_cfg.commands.curriculum.enable = False
    env = SuperMimicG1Env(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        folder += f"_{args_cli.run_name}"
    else:
        folder += "_supermimic_student"
    log_dir = os.path.join(log_root_path, folder)

    cfg = SuperMimicStudentCfg(
        num_steps_per_env=int(agent_cfg.num_steps_per_env),
        utd=0.4,
        replay_capacity=1000,
        bc_warmup_iters=int(args_cli.bc_warmup_iters),
        dagger_warmup_iters=int(args_cli.dagger_warmup_iters),
        student_lr=float(args_cli.student_lr),
        symmetry_weight=float(args_cli.symmetry_weight),
        teacher_anchor_init=float(args_cli.teacher_anchor_init),
        teacher_anchor_final=float(args_cli.teacher_anchor_final),
        hindsight_cmd_frac=float(args_cli.hindsight_cmd_frac),
        save_interval=int(args_cli.save_interval),
        use_tensorboard=True,
    )

    runner = SuperMimicStudentRunner(
        env=env,
        teacher_ckpt_path=args_cli.teacher_ckpt,
        cfg=cfg,
        log_dir=log_dir,
        resume_student_path=args_cli.resume_student_ckpt,
        device=agent_cfg.device,
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    tot_iters = args_cli.student_iterations if args_cli.max_iterations is None else args_cli.max_iterations
    print(f"[INFO] Launching SuperMimic Student Training for {tot_iters} iterations...", flush=True)
    runner.learn(num_learning_iterations=int(tot_iters))



if __name__ == "__main__":
    train()

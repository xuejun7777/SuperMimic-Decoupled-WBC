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

from __future__ import annotations

import argparse
import ast
import os
import random
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume training from a checkpoint (selected via --load_run / --checkpoint).",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    arg_group.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a config field by dotted path, e.g. "
            "--set commands.curriculum.body_roll.start_iter=20000. "
            "Targets env_cfg by default; prefix with 'agent.' for the agent cfg. "
            "Repeatable. VALUE is parsed as a Python literal (int/float/tuple/dict/bool), "
            "falling back to a string."
        ),
    )


def _resolve_parent(root, path_parts):
    """Walk a dotted path over objects (attributes) and dicts (keys)."""
    obj = root
    for part in path_parts:
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj


def _set_leaf(parent, key, value):
    if isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


def apply_cfg_overrides(env_cfg, agent_cfg, overrides):
    """Apply ``KEY=VALUE`` overrides to env_cfg (default) or agent_cfg.

    Targets env_cfg unless the key is prefixed with ``env.`` or ``agent.``.
    The value is parsed with ``ast.literal_eval`` so numbers, tuples, dicts and
    bools work; anything that fails to parse is kept as a raw string.
    """
    if not overrides:
        return
    roots = {"env": env_cfg, "agent": agent_cfg}
    for assignment in overrides:
        if "=" not in assignment:
            raise ValueError(f"Invalid --set override (missing '='): {assignment!r}")
        key, _, raw = assignment.partition("=")
        key, raw = key.strip(), raw.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # treat unparseable values as plain strings

        root_name, _, sub = key.partition(".")
        if root_name in roots and sub:
            root, path = roots[root_name], sub.split(".")
        else:
            root, path = env_cfg, key.split(".")

        parent = _resolve_parent(root, path[:-1])
        _set_leaf(parent, path[-1], value)
        print(f"[INFO] Override applied: {key} = {value!r}")


def apply_trained_actuator_gains(env_cfg, run_dir: str, *, strict: bool = False) -> None:
    """Apply the actuator PD gains (kp/kd) a checkpoint was trained with.

    Runs may train with different leg/waist stiffness & damping (e.g. via ``--set``
    overrides), which ``train_teacher.py`` persists to ``<run_dir>/params/env.yaml``.
    The default config from the task registry would otherwise evaluate or distill a
    checkpoint with the baseline gains, changing the closed-loop PD behavior. This
    reloads the saved per-actuator stiffness/damping and applies them to ``env_cfg``
    so the policy runs on the gains it was trained with. Scoped to gains on purpose;
    warns and falls back to the defaults unless ``strict`` is set. Student
    distillation uses strict mode because silently changing the teacher dynamics
    invalidates the imitation target.
    """
    params_path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(params_path):
        if strict:
            raise FileNotFoundError(f"No saved env config at {params_path}")
        print(f"[WARN] No saved env config at {params_path}; using default actuator gains.")
        return
    # The dumped env.yaml embeds Python objects (e.g. slice tags) that yaml.full_load
    # refuses to construct. These are our own trusted files, so read with unsafe_load.
    try:
        with open(params_path) as f:
            saved_actuators = yaml.unsafe_load(f)["scene"]["robot"]["actuators"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        if strict:
            raise ValueError(f"Could not read actuators from {params_path}: {exc}") from exc
        print(f"[WARN] Could not read actuators from {params_path} ({exc}); using default gains.")
        return

    if strict:
        missing = []
        for name in env_cfg.scene.robot.actuators:
            saved_act = saved_actuators.get(name) if isinstance(saved_actuators, dict) else None
            if not isinstance(saved_act, dict):
                missing.append(name)
                continue
            for field in ("stiffness", "damping"):
                if saved_act.get(field) is None:
                    missing.append(f"{name}.{field}")
        if missing:
            raise ValueError(f"Incomplete actuator gains in {params_path}: {', '.join(missing)}")

    applied = []
    for name, act_cfg in env_cfg.scene.robot.actuators.items():
        saved_act = saved_actuators.get(name) if isinstance(saved_actuators, dict) else None
        if not isinstance(saved_act, dict):
            continue
        for field in ("stiffness", "damping"):
            if saved_act.get(field) is not None:
                setattr(act_cfg, field, saved_act[field])
        applied.append(name)

    if applied:
        print(f"[INFO] Applied trained actuator gains from {params_path}:")
        for name in applied:
            act = env_cfg.scene.robot.actuators[name]
            print(f"        {name}: stiffness={act.stiffness}, damping={act.damping}")
    else:
        if strict:
            raise ValueError(f"No matching actuator groups in {params_path}")
        print(f"[WARN] No matching actuator groups in {params_path}; using default gains.")


def apply_trained_command_config(env_cfg, run_dir: str, *, strict: bool = False) -> None:
    """Apply the command sampling distribution saved with a teacher checkpoint.

    Student distillation should visit the same command modes and ranges as the
    teacher. In particular, heading-controlled, direct-yaw, exact-zero-yaw, and
    standing environments produce different state distributions even when their
    instantaneous 7-D commands sometimes coincide. Load those settings from the
    teacher's ``params/env.yaml`` instead of relying on today's task defaults.

    The teacher's curriculum schedule is intentionally not restored: student
    training disables command curriculum and uses the saved final ``ranges`` from
    step zero. Unknown fields are ignored for forward compatibility.
    """
    params_path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(params_path):
        if strict:
            raise FileNotFoundError(f"No saved env config at {params_path}")
        print(f"[WARN] No saved env config at {params_path}; using default command sampling.")
        return

    try:
        with open(params_path) as f:
            saved_commands = yaml.unsafe_load(f)["commands"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        if strict:
            raise ValueError(f"Could not read commands from {params_path}: {exc}") from exc
        print(f"[WARN] Could not read commands from {params_path} ({exc}); using defaults.")
        return

    if not isinstance(saved_commands, dict):
        if strict:
            raise ValueError(f"Invalid commands block in {params_path}")
        print(f"[WARN] Invalid commands block in {params_path}; using default command sampling.")
        return

    required_scalar_fields = (
        "resampling_time_range",
        "rel_standing_envs",
        "rel_heading_envs",
        "heading_command",
        "heading_control_stiffness",
    )
    optional_scalar_fields = ("rel_zero_vel_yaw_envs",)
    required_range_fields = (
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "heading",
        "body_height",
        "body_roll",
        "body_pitch",
        "body_yaw",
    )
    saved_ranges = saved_commands.get("ranges")
    if strict:
        missing = [field for field in required_scalar_fields if field not in saved_commands]
        missing.extend(
            f"ranges.{field}"
            for field in required_range_fields
            if not isinstance(saved_ranges, dict) or field not in saved_ranges
        )
        unsupported = [field for field in required_scalar_fields if not hasattr(env_cfg.commands, field)]
        unsupported.extend(
            f"ranges.{field}" for field in required_range_fields if not hasattr(env_cfg.commands.ranges, field)
        )
        if missing:
            raise ValueError(f"Incomplete command config in {params_path}: {', '.join(missing)}")
        if unsupported:
            raise ValueError(f"Current task cannot apply saved commands from {params_path}: {', '.join(unsupported)}")

    applied = []
    for field in required_scalar_fields + optional_scalar_fields:
        if field in saved_commands and hasattr(env_cfg.commands, field):
            setattr(env_cfg.commands, field, saved_commands[field])
            applied.append(field)

    # Saved configs from before the direct-zero-yaw feature predate this field.
    # Their historical behavior was a purely uniform direct-yaw subset, i.e. a
    # conditional exact-zero probability of 0.0. Restoring that value is harmless
    # for the common heading-only v7 runs and correct for any older mixed run.
    if "rel_zero_vel_yaw_envs" not in saved_commands and hasattr(env_cfg.commands, "rel_zero_vel_yaw_envs"):
        env_cfg.commands.rel_zero_vel_yaw_envs = 0.0
        applied.append("rel_zero_vel_yaw_envs (legacy default 0.0)")

    if isinstance(saved_ranges, dict):
        for field, value in saved_ranges.items():
            if hasattr(env_cfg.commands.ranges, field):
                setattr(env_cfg.commands.ranges, field, value)
                applied.append(f"ranges.{field}")

    if applied:
        print(f"[INFO] Applied trained command sampling from {params_path}:")
        print(f"        {', '.join(applied)}")
    else:
        if strict:
            raise ValueError(f"No matching command fields in {params_path}")
        print(f"[WARN] No matching command fields in {params_path}; using defaults.")


def update_rsl_rl_cfg(agent_cfg: BaseAgentConfig, args_cli: argparse.Namespace):

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg

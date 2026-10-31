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

"""Custom command terms used by the locomotion environments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class CommandAxisCurriculumCfg:
    """Linear curriculum schedule for a single command axis."""

    start_range: tuple[float, float] = (0.0, 0.0)
    end_range: tuple[float, float] = (0.0, 0.0)
    start_iter: int = 0
    end_iter: int = 1000


@configclass
class CommandCurriculumCfg:
    """Per-axis linear curriculum on the sampled command ranges.

    Axes left as ``None`` fall back to the static ``ranges.*`` value
    (no curriculum on that axis). Set ``enable=False`` to disable the
    curriculum entirely.
    """

    enable: bool = True
    lin_vel_x: CommandAxisCurriculumCfg | None = None
    lin_vel_y: CommandAxisCurriculumCfg | None = None
    ang_vel_z: CommandAxisCurriculumCfg | None = None
    body_height: CommandAxisCurriculumCfg | None = None
    body_roll: CommandAxisCurriculumCfg | None = None
    body_pitch: CommandAxisCurriculumCfg | None = None
    body_yaw: CommandAxisCurriculumCfg | None = None


@configclass
class UniformVelocityBodyCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator with optional body commands."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    curriculum: CommandCurriculumCfg = CommandCurriculumCfg()
    """Optional linear curriculum on the sampled command range."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """Probability of heading-based angular velocity command when heading commands are enabled."""

    rel_zero_vel_yaw_envs: float = 0.0
    """Conditional probability of exact zero yaw rate in non-standing, direct-yaw environments.

    Only the yaw-rate component is zeroed; linear velocity remains uniformly sampled from its configured ranges.
    """

    body_command: bool = False
    """Whether to generate body posture/height commands. Defaults to False."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity and body commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad)."""

        body_height: tuple[float, float] = MISSING
        """Range for the desired body height command (in meters)."""

        body_roll: tuple[float, float] = MISSING
        """Range for the desired body roll command (in radians)."""

        body_pitch: tuple[float, float] = MISSING
        """Range for the desired body pitch command (in radians)."""

        body_yaw: tuple[float, float] = MISSING
        """Range for the desired body yaw command (in radians)."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the velocity and body commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker."""

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


class UniformVelocityBodyCommand(CommandTerm):
    r"""Command generator that extends Isaac Lab velocity commands with torso-pose commands."""

    cfg: UniformVelocityBodyCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityBodyCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )
        for name in ("rel_standing_envs", "rel_heading_envs", "rel_zero_vel_yaw_envs"):
            probability = float(getattr(self.cfg, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"The command probability '{name}' must be in [0, 1], got {probability}.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        if self.cfg.body_command:
            self.vel_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        else:
            self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Current rsl-rl iteration estimate used by the per-axis curriculum schedules.
        # The base environment updates this via ``update_curriculum`` once per step.
        self.current_iteration: float = 0.0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tDirect-yaw zero-rate probability (conditional): {self.cfg.rel_zero_vel_yaw_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity/body command in the base frame."""
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def update_curriculum(self, iteration: float) -> None:
        """Record the current rsl-rl iteration so per-axis schedules can interpolate."""
        self.current_iteration = float(iteration)

    def _effective_range(self, axis_name: str, base_range: tuple[float, float]) -> tuple[float, float]:
        """Return the curriculum-aware sampling range for ``axis_name``.

        When the curriculum is disabled or no schedule is set for the axis, the
        static ``base_range`` is returned unchanged. Otherwise the range is
        linearly interpolated between ``start_range`` and ``end_range`` over
        ``[start_iter, end_iter]``.
        """
        if not self.cfg.curriculum.enable:
            return base_range
        axis_cfg = getattr(self.cfg.curriculum, axis_name, None)
        if axis_cfg is None:
            return base_range
        iteration = self.current_iteration
        if iteration <= axis_cfg.start_iter:
            return tuple(axis_cfg.start_range)
        if iteration >= axis_cfg.end_iter:
            return tuple(axis_cfg.end_range)
        span = max(int(axis_cfg.end_iter) - int(axis_cfg.start_iter), 1)
        progress = (iteration - axis_cfg.start_iter) / span
        low = axis_cfg.start_range[0] + (axis_cfg.end_range[0] - axis_cfg.start_range[0]) * progress
        high = axis_cfg.start_range[1] + (axis_cfg.end_range[1] - axis_cfg.start_range[1]) * progress
        return (low, high)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self._effective_range("lin_vel_x", self.cfg.ranges.lin_vel_x))
        self.vel_command_b[env_ids, 1] = r.uniform_(*self._effective_range("lin_vel_y", self.cfg.ranges.lin_vel_y))
        self.vel_command_b[env_ids, 2] = r.uniform_(*self._effective_range("ang_vel_z", self.cfg.ranges.ang_vel_z))
        if self.cfg.body_command:
            if hasattr(self.cfg.ranges, "body_height"):
                self.vel_command_b[env_ids, 3] = r.uniform_(
                    *self._effective_range("body_height", self.cfg.ranges.body_height)
                )
            if hasattr(self.cfg.ranges, "body_roll"):
                self.vel_command_b[env_ids, 4] = r.uniform_(
                    *self._effective_range("body_roll", self.cfg.ranges.body_roll)
                )
            if hasattr(self.cfg.ranges, "body_pitch"):
                self.vel_command_b[env_ids, 5] = r.uniform_(
                    *self._effective_range("body_pitch", self.cfg.ranges.body_pitch)
                )
            if hasattr(self.cfg.ranges, "body_yaw"):
                self.vel_command_b[env_ids, 6] = r.uniform_(
                    *self._effective_range("body_yaw", self.cfg.ranges.body_yaw)
                )
        self.is_heading_env[env_ids] = False
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs

        direct_moving_envs = ~self.is_heading_env[env_ids] & ~self.is_standing_env[env_ids]
        if self.cfg.rel_zero_vel_yaw_envs <= 0.0:
            self.is_zero_vel_yaw_env[env_ids] = False
        elif self.cfg.rel_zero_vel_yaw_envs >= 1.0:
            self.is_zero_vel_yaw_env[env_ids] = direct_moving_envs
        else:
            zero_yaw_rate_draw = r.uniform_(0.0, 1.0) < self.cfg.rel_zero_vel_yaw_envs
            self.is_zero_vel_yaw_env[env_ids] = direct_moving_envs & zero_yaw_rate_draw

        # CommandTerm.reset() resamples without calling _update_command(). Apply
        # the masks here so the first post-reset observation sees effective commands.
        self._apply_command_postprocessing(env_ids)

    def _update_command(self):
        """Post-process the velocity command for every environment."""
        self._apply_command_postprocessing(slice(None))

    def _apply_command_postprocessing(self, env_ids: Sequence[int] | slice):
        """Apply heading, direct-zero-yaw, and standing masks to selected environments."""
        if self.cfg.heading_command:
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            heading_yaw_rate = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_b[env_ids, 2] = torch.where(
                self.is_heading_env[env_ids],
                heading_yaw_rate,
                self.vel_command_b[env_ids, 2],
            )

        self.vel_command_b[env_ids, 2] = torch.where(
            self.is_zero_vel_yaw_env[env_ids],
            torch.zeros_like(self.vel_command_b[env_ids, 2]),
            self.vel_command_b[env_ids, 2],
        )
        self.vel_command_b[env_ids, :3] = torch.where(
            self.is_standing_env[env_ids].unsqueeze(-1),
            torch.zeros_like(self.vel_command_b[env_ids, :3]),
            self.vel_command_b[env_ids, :3],
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

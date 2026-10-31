"""Zero-intrusion BaseEnv subclass for SuperMimic data collection.

`BaseEnv` in `legged_lab/envs/base/base_env.py` remains 100% untouched so native PPO
training runs on the exact original codebase. `SuperMimicG1Env` subclasses `BaseEnv`
solely to:
  1. Snapshot the true post-physics, pre-reset state `s_{t+1}`, `feet_contact_{t+1}`,
     and exact contact-impact rewards right before `BaseEnv.reset(env_ids)` overwrites
     `robot.data` inside `BaseEnv.step()`.
  2. Track episode IDs, interval `push_robot` events, and newly sampled AMASS arm clips
     for the `G1ReplayBuffer`.
Zero physics, RNG, observation, command, or reward behavior is altered.
"""

from __future__ import annotations

import torch

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards


class SuperMimicG1Env(BaseEnv):
    def __init__(self, cfg, headless: bool):
        self._sm_ready = False
        self._amass_reset_hook = None
        super().__init__(cfg, headless)

        self._feet_sensor_ids = self.feet_cfg.body_ids
        self._impact_term_ids = torch.tensor(
            [self.reward_manager._term_names.index(n) for n in DifferentiableG1Rewards.IMPACT_TERM_NAMES],
            dtype=torch.long,
            device=self.device,
        )

        # Read per-environment startup Domain Randomization context xi:
        # [static_friction, dynamic_friction, restitution, torso_mass_delta]
        mat = self.robot.root_physx_view.get_material_properties().to(self.device)  # (N, num_shapes, 3)
        masses = self.robot.root_physx_view.get_masses().to(self.device)  # (N, num_bodies)
        torso_idx = self.robot.body_names.index("torso_link")
        default_torso_mass = self.robot.data.default_mass[:, torso_idx].to(self.device)
        torso_dm = (masses[:, torso_idx] - default_torso_mass).unsqueeze(-1)
        self.xi = torch.cat([mat[:, 0, :3], torso_dm], dim=-1).float().clone()  # (N, 4)

        self.ep_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Locate interval push_robot timer if enabled so push velocity-jumps can be flagged
        self._push_term_idx: int | None = None
        if hasattr(self.event_manager, "_interval_term_names") and "push_robot" in self.event_manager._interval_term_names:
            self._push_term_idx = self.event_manager._interval_term_names.index("push_robot")

        # Pre-allocate snapshot buffers populated inside `reset()` right before reset overwrites data
        self.last_pre_reset_state = torch.zeros(self.num_envs, 71, dtype=torch.float32, device=self.device)
        self.last_pre_reset_feet_contact = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.last_impact_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_terminated = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_timed_out = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_pushed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._sm_ready = True

    def attach_replay_buffer(self, buffer) -> None:
        """Register initial AMASS clips of all environments and hook future resets."""
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        buffer.register_amass_clips(all_ids, self.amass)
        self._amass_reset_hook = buffer.register_amass_clips

    @torch.no_grad()
    def read_full_state(self) -> torch.Tensor:
        """Read origin-free 71-D Markov state `[p_local(3), q_wxyz(4), q_usd(29), v_link_w(3), w_b(3), qd_usd(29)]`."""
        d = self.robot.data
        p_local = d.root_link_pos_w - self.scene.env_origins
        return torch.cat(
            [
                p_local,
                d.root_link_quat_w,
                d.joint_pos,
                d.root_link_lin_vel_w,
                d.root_link_ang_vel_b,
                d.joint_vel,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def read_feet_contact(self) -> torch.Tensor:
        """Exact 2-D binary foot contact indicator matching `BaseEnv.compute_current_observations`."""
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_sensor_ids], dim=-1), dim=1)[0] > 0.5
        return feet_contact.float()

    @torch.no_grad()
    def read_contact_timers(self) -> tuple[torch.Tensor, torch.Tensor]:
        ct = self.contact_sensor.data.current_contact_time[:, self._feet_sensor_ids].clone()
        at = self.contact_sensor.data.current_air_time[:, self._feet_sensor_ids].clone()
        return ct, at

    @torch.no_grad()
    def current_step_inputs(self) -> dict[str, torch.Tensor]:
        """Snapshot pre-step quantities right before `env.step(actions)` runs."""
        amass_idx = torch.where(
            self.amass_index >= self.sequence_length,
            torch.zeros_like(self.amass_index),
            self.amass_index,
        )
        env_ar = torch.arange(self.num_envs, device=self.device)
        arm_tgt = self.amass[env_ar, amass_idx].clone()
        prev_act = self.action_buffer._circular_buffer.buffer[:, -1, :].clone()
        ct, at = self.read_contact_timers()

        push_t_before = None
        if self._push_term_idx is not None and hasattr(self.event_manager, "_interval_term_time_left"):
            t_left = self.event_manager._interval_term_time_left[self._push_term_idx]
            if torch.is_tensor(t_left):
                push_t_before = t_left.clone()

        return {
            "s": self.read_full_state(),
            "prev_action": prev_act,
            "arm_tgt": arm_tgt,
            "amass_step": amass_idx.clone(),
            "feet_contact": self.read_feet_contact(),
            "contact_time": ct,
            "air_time": at,
            "cmd": self.command_generator.command.clone(),
            "heading_target": self.command_generator.heading_target.clone(),
            "is_heading": self.command_generator.is_heading_env.float().clone(),
            "is_standing": self.command_generator.is_standing_env.float().clone(),
            "ep_id": self.ep_id.clone(),
            "_push_t_before": push_t_before,
        }

    def step(self, actions: torch.Tensor):
        self._step_push_t_before = None
        if self._sm_ready and self._push_term_idx is not None and hasattr(self.event_manager, "_interval_term_time_left"):
            t_left = self.event_manager._interval_term_time_left[self._push_term_idx]
            if torch.is_tensor(t_left):
                self._step_push_t_before = t_left.clone()
        return super().step(actions)

    def reset(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_sm_ready", False):
            return super().reset(env_ids)

        # `BaseEnv.step()` calls `self.reset(env_ids)` right after `check_reset()` and
        # `reward_manager.compute(self.step_dt)`, BEFORE any reset mutations happen!
        with torch.no_grad():
            self.last_pre_reset_state.copy_(self.read_full_state())
            self.last_pre_reset_feet_contact.copy_(self.read_feet_contact())
            if hasattr(self.reward_manager, "_step_reward"):
                imp_unscaled = self.reward_manager._step_reward.index_select(-1, self._impact_term_ids).sum(dim=-1)
                self.last_impact_reward.copy_(imp_unscaled * self.step_dt)
            if hasattr(self, "reset_buf"):
                self.last_terminated.copy_((self.reset_buf & ~self.time_out_buf).float())
                self.last_timed_out.copy_(self.time_out_buf.float())

            if getattr(self, "_step_push_t_before", None) is not None:
                t_after = self.event_manager._interval_term_time_left[self._push_term_idx]
                self.last_pushed.copy_(t_after > self._step_push_t_before)
            else:
                self.last_pushed.zero_()

        super().reset(env_ids)

        if len(env_ids) > 0:
            with torch.no_grad():
                self.ep_id[env_ids] += 1
                if self._amass_reset_hook is not None:
                    self._amass_reset_hook(env_ids, self.amass[env_ids])

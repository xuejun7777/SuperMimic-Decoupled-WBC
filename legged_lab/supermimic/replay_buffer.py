"""Lockstep GPU Ring Replay Buffer with AMASS Clip Bank for G1 SuperMimic.

Stores complete transition tuples `(s_t, a_exec_t, arm_tgt_t, s_{t+1})` across all `E`
parallel environments in lockstep `[T, E, ...]`, including exact pre-reset terminal
states `s_{T+1}`, contact labels/timers, impact rewards, command metadata, and an
`8192`-slot `float16` AMASS clip bank for exact `H`-step future arm lookahead in
imagined BPTT rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from legged_lab.supermimic.state_spec import STATE_DIM


@dataclass
class WMWindowBatch:
    states: torch.Tensor  # (k+1, B, 71)
    policy_actions: torch.Tensor  # (k, B, 15)
    arm_targets: torch.Tensor  # (k, B, 14)
    xi: torch.Tensor  # (B, 4)
    next_contacts: torch.Tensor  # (k, B, 2)
    impact_rewards: torch.Tensor  # (k, B)
    valid_mask: torch.Tensor  # (k, B) 1.0 where transition j is smooth & same segment


@dataclass
class ImagStartBatch:
    state: torch.Tensor  # (B, 71)
    prev_action: torch.Tensor  # (B, 15)
    feet_contact: torch.Tensor  # (B, 2)
    contact_time: torch.Tensor  # (B, 2)
    air_time: torch.Tensor  # (B, 2)
    command: torch.Tensor  # (B, 7)
    heading_target: torch.Tensor  # (B,)
    is_heading_env: torch.Tensor  # (B,)
    is_standing_env: torch.Tensor  # (B,)
    xi: torch.Tensor  # (B, 4)
    arm_future: torch.Tensor  # (H, B, 14) exact exogenous arm targets for steps 0..H-1


class G1ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int = 1000,
        act_dim: int = 15,
        arm_dim: int = 14,
        amass_seq_len: int = 1000,
        clip_bank_size: int = 8192,
        device: str | torch.device = "cuda:0",
    ):
        self.E = int(num_envs)
        self.T = int(capacity)
        self.act_dim = int(act_dim)
        self.arm_dim = int(arm_dim)
        self.amass_seq_len = int(amass_seq_len)
        self.clip_bank_size = max(int(clip_bank_size), 2 * self.E)
        self.device = torch.device(device)

        T, E = self.T, self.E
        self.s = torch.zeros(T, E, STATE_DIM, dtype=torch.float32, device=self.device)
        self.s_next = torch.zeros(T, E, STATE_DIM, dtype=torch.float32, device=self.device)
        self.a_exec = torch.zeros(T, E, act_dim, dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros(T, E, act_dim, dtype=torch.float32, device=self.device)
        self.arm_tgt = torch.zeros(T, E, arm_dim, dtype=torch.float32, device=self.device)

        self.feet_contact = torch.zeros(T, E, 2, dtype=torch.float32, device=self.device)
        self.next_feet_contact = torch.zeros(T, E, 2, dtype=torch.float32, device=self.device)
        self.contact_time = torch.zeros(T, E, 2, dtype=torch.float32, device=self.device)
        self.air_time = torch.zeros(T, E, 2, dtype=torch.float32, device=self.device)
        self.impact_rew = torch.zeros(T, E, dtype=torch.float32, device=self.device)
        self.pushed = torch.zeros(T, E, dtype=torch.bool, device=self.device)

        self.cmd = torch.zeros(T, E, 7, dtype=torch.float32, device=self.device)
        self.heading_target = torch.zeros(T, E, dtype=torch.float32, device=self.device)
        self.is_heading = torch.zeros(T, E, dtype=torch.float32, device=self.device)
        self.is_standing = torch.zeros(T, E, dtype=torch.float32, device=self.device)

        self.ep_id = torch.full((T, E), -1, dtype=torch.long, device=self.device)
        self.clip_slot = torch.zeros(T, E, dtype=torch.long, device=self.device)
        self.clip_gen = torch.zeros(T, E, dtype=torch.long, device=self.device)
        self.amass_step = torch.zeros(T, E, dtype=torch.long, device=self.device)

        # Ring bank of full 1000-step AMASS arm clips in fp16 (~229 MB for 8192 slots)
        self.amass_bank = torch.zeros(self.clip_bank_size, self.amass_seq_len, arm_dim, dtype=torch.float16, device=self.device)
        self.bank_gen = torch.full((self.clip_bank_size,), -1, dtype=torch.long, device=self.device)
        self._bank_ptr = 0
        self._gen_counter = 1

        self.env_clip_slot = torch.zeros(E, dtype=torch.long, device=self.device)
        self.env_clip_gen = torch.zeros(E, dtype=torch.long, device=self.device)

        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def register_amass_clips(self, env_ids: torch.Tensor, clips: torch.Tensor) -> None:
        """Register newly sampled (M, 1000, 14) AMASS clips for reset `env_ids`."""
        m = int(env_ids.numel())
        if m == 0:
            return
        slots = (torch.arange(m, device=self.device, dtype=torch.long) + self._bank_ptr) % self.clip_bank_size
        gens = torch.arange(m, device=self.device, dtype=torch.long) + self._gen_counter
        self._bank_ptr = int((self._bank_ptr + m) % self.clip_bank_size)
        self._gen_counter += m

        self.amass_bank[slots] = clips.to(dtype=torch.float16)
        self.bank_gen[slots] = gens
        self.env_clip_slot[env_ids] = slots
        self.env_clip_gen[env_ids] = gens

    @torch.no_grad()
    def add_step(
        self,
        s: torch.Tensor,
        a_exec: torch.Tensor,
        prev_action: torch.Tensor,
        arm_tgt: torch.Tensor,
        s_next: torch.Tensor,
        feet_contact: torch.Tensor,
        next_feet_contact: torch.Tensor,
        contact_time: torch.Tensor,
        air_time: torch.Tensor,
        impact_rew: torch.Tensor,
        cmd: torch.Tensor,
        heading_target: torch.Tensor,
        is_heading: torch.Tensor,
        is_standing: torch.Tensor,
        ep_id: torch.Tensor,
        amass_step: torch.Tensor,
        pushed: torch.Tensor,
    ) -> None:
        p = self.ptr
        self.s[p].copy_(s)
        self.a_exec[p].copy_(a_exec)
        self.prev_action[p].copy_(prev_action)
        self.arm_tgt[p].copy_(arm_tgt)
        self.s_next[p].copy_(s_next)
        self.feet_contact[p].copy_(feet_contact)
        self.next_feet_contact[p].copy_(next_feet_contact)
        self.contact_time[p].copy_(contact_time)
        self.air_time[p].copy_(air_time)
        self.impact_rew[p].copy_(impact_rew)
        self.cmd[p].copy_(cmd)
        self.heading_target[p].copy_(heading_target)
        self.is_heading[p].copy_(is_heading)
        self.is_standing[p].copy_(is_standing)
        self.ep_id[p].copy_(ep_id)
        self.clip_slot[p].copy_(self.env_clip_slot)
        self.clip_gen[p].copy_(self.env_clip_gen)
        self.amass_step[p].copy_(amass_step)
        self.pushed[p].copy_(pushed)

        self.ptr = (p + 1) % self.T
        if self.size < self.T:
            self.size += 1

    def _age_to_idx(self, age: torch.Tensor) -> torch.Tensor:
        """Convert age (0 = newest row at ptr-1, size-1 = oldest row) to ring index."""
        return (self.ptr - 1 - age) % self.T

    @torch.no_grad()
    def sample_windows(self, batch_size: int, k: int, xi_per_env: torch.Tensor) -> WMWindowBatch:
        """Sample `batch_size` k-step windows for World Model autoregressive fitting.

        Because `s_next` stores the true pre-reset post-step state (even on terminal fall steps!),
        every prefix step `j` that shares `ep_id` with step 0 and has `!pushed[j]` is a 100%
        physically valid smooth transition—including the exact terminal fall transition!
        """
        if self.size < k:
            raise RuntimeError(f"Replay buffer size ({self.size}) < window length k ({k})")

        env_ids = torch.randint(0, self.E, (batch_size,), device=self.device)
        # Start age in [k - 1, self.size - 1] so ages - (0..k-1) are all >= 0 (within valid buffer)
        start_age = torch.randint(k - 1, self.size, (batch_size,), device=self.device)
        offs = torch.arange(k, device=self.device)[:, None]  # (k, 1)
        ages = start_age[None, :] - offs  # (k, B), strictly decreasing toward newest
        t_idx = self._age_to_idx(ages)  # (k, B)
        e_idx = env_ids[None, :].expand(k, batch_size)

        ep_seq = self.ep_id[t_idx, e_idx]  # (k, B)
        pushed_seq = self.pushed[t_idx, e_idx]  # (k, B)

        # A step j is valid for open-loop autoregressive rollout from states[0] iff all steps 0..j
        # belong to the same episode AND no external push teleportation happened on steps 0..j.
        same_ep = ep_seq == ep_seq[0:1, :]
        not_pushed = ~pushed_seq
        step_ok = same_ep & not_pushed
        # Cumulative AND along the k-step rollout horizon
        valid_mask = torch.cumprod(step_ok.to(torch.float32), dim=0)

        s0 = self.s[t_idx[0], env_ids].unsqueeze(0)  # (1, B, 71)
        s_following = self.s_next[t_idx, e_idx]  # (k, B, 71)
        states = torch.cat([s0, s_following], dim=0)  # (k+1, B, 71)

        return WMWindowBatch(
            states=states,
            policy_actions=self.a_exec[t_idx, e_idx],
            arm_targets=self.arm_tgt[t_idx, e_idx],
            xi=xi_per_env[env_ids],
            next_contacts=self.next_feet_contact[t_idx, e_idx],
            impact_rewards=self.impact_rew[t_idx, e_idx],
            valid_mask=valid_mask,
        )

    @torch.no_grad()
    def sample_starts(self, batch_size: int, horizon: int, xi_per_env: torch.Tensor) -> ImagStartBatch:
        """Sample `batch_size` initial states + exact `horizon`-step future AMASS arm trajectories."""
        if self.size < 1:
            raise RuntimeError("Replay buffer is empty")

        env_ids = torch.randint(0, self.E, (batch_size,), device=self.device)
        age = torch.randint(0, self.size, (batch_size,), device=self.device)
        t0 = self._age_to_idx(age)

        slots = self.clip_slot[t0, env_ids]  # (B,)
        gens = self.clip_gen[t0, env_ids]  # (B,)
        step0 = self.amass_step[t0, env_ids]  # (B,)

        h_offs = torch.arange(horizon, device=self.device)[:, None]  # (H, 1)
        seq_idx = (step0[None, :] + h_offs) % self.amass_seq_len  # (H, B)
        slot_exp = slots[None, :].expand(horizon, batch_size)
        bank_arms = self.amass_bank[slot_exp, seq_idx].float()  # (H, B, 14)

        # Fallback to held arm_tgt[t0] only if an ancient short episode's bank slot already wrapped
        bank_valid = (self.bank_gen[slots] == gens)[None, :, None]  # (1, B, 1)
        held_arm = self.arm_tgt[t0, env_ids].unsqueeze(0).expand(horizon, batch_size, self.arm_dim)
        arm_future = torch.where(bank_valid, bank_arms, held_arm)

        return ImagStartBatch(
            state=self.s[t0, env_ids],
            prev_action=self.prev_action[t0, env_ids],
            feet_contact=self.feet_contact[t0, env_ids],
            contact_time=self.contact_time[t0, env_ids],
            air_time=self.air_time[t0, env_ids],
            command=self.cmd[t0, env_ids],
            heading_target=self.heading_target[t0, env_ids],
            is_heading_env=self.is_heading[t0, env_ids],
            is_standing_env=self.is_standing[t0, env_ids],
            xi=xi_per_env[env_ids],
            arm_future=arm_future,
        )

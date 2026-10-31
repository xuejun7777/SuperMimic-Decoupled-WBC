"""World Model Multi-Step Fitting & Short-Horizon Analytic BPTT Actor-Critic Trainer.

Implements Reference-Free SuperMimic for Unitree G1 Decoupled Locomotion + Body Control:
1. `WorldModelTrainer`: fits `G1WorldModel` (`DynamicsMLP` + `FootContactHead` + `ImpactPenaltyHead`)
   on k-step replay windows with episode/push boundary masking.
2. `PolicyCriticTrainer`: rolls out `TeacherActorMLP` for `H` steps through the frozen
   `G1WorldModel` conditioned on exact future AMASS upper-body arm trajectories, evaluates
   the 31-term `DifferentiableG1Rewards` + terminal `PrivilegedCriticMLP` bootstrap,
   backpropagates analytic policy gradients (`BPTT`), and fits `PrivilegedCriticMLP` to
   survival-weighted GAE(gamma, lambda) targets computed via an EMA target critic.

Accelerates both k-step WM forward+backward and H-step Analytic BPTT forward+backward via
native `torch.cuda.CUDAGraph` capture—eliminating Python/autograd dispatch overhead while
remaining 100% free of Triton C-extension callbacks in multithreaded environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from typing import TYPE_CHECKING

import torch

from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards
from legged_lab.supermimic.nets import PrivilegedCriticMLP, TeacherActorMLP
from legged_lab.supermimic.replay_buffer import G1ReplayBuffer, ImagStartBatch, WMWindowBatch
from legged_lab.supermimic.world_model import G1WorldModel

if TYPE_CHECKING:
    from legged_lab.supermimic.symmetry import G1SagittalSymmetry


@dataclass
class WMTrainerCfg:
    window_k: int = 8
    batch_size: int = 2048
    batches_per_update: int = 8
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0
    use_mirror_augmentation: bool = True
    use_cudagraph: bool = True


@dataclass
class PolicyTrainerCfg:
    horizon: int = 24
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 2.0e-4
    critic_lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_grad_clip_norm: float = 1.0
    critic_epochs: int = 2
    critic_minibatches: int = 4
    critic_ema_tau: float = 0.02
    use_critic_bootstrap: bool = True
    obs_noise_in_imagination: bool = True
    # Zero-bias safety guardrail strictly outside all physical G1 joint ranges (|a| > 10.0 <=> > 2.5 rad offset)
    ood_action_limit: float = 10.0
    ood_action_weight: float = 0.1
    # Phase-asymmetry-gated sagittal mirror-symmetry equivariance & start-state reflection
    use_mirror_augmentation: bool = True
    symmetry_weight: float = 1.5
    sym_tau_low: float = 0.18
    sym_tau_high: float = 0.45
    sym_warmup_ratio: float = 0.45
    sym_ramp_ratio: float = 0.15
    use_cudagraph: bool = True


class WorldModelTrainer:
    """Autoregressive k-step World Model trainer with native CUDA Graph capture."""

    def __init__(
        self,
        wm: G1WorldModel,
        cfg: WMTrainerCfg,
        symmetry: G1SagittalSymmetry | None = None,
    ):
        self.wm = wm
        self.cfg = cfg
        self.symmetry = symmetry
        use_fused = next(wm.parameters()).is_cuda
        self.opt = torch.optim.AdamW(
            wm.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=use_fused,
        )
        self._wm_graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: WMWindowBatch | None = None
        self._static_loss: torch.Tensor | None = None
        self._static_metrics: dict[str, torch.Tensor] | None = None

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _capture_wm_graph(self, sample: WMWindowBatch) -> None:
        dev = sample.states.device
        self._static_batch = WMWindowBatch(
            states=sample.states.clone(),
            policy_actions=sample.policy_actions.clone(),
            arm_targets=sample.arm_targets.clone(),
            xi=sample.xi.clone(),
            next_contacts=sample.next_contacts.clone(),
            impact_rewards=sample.impact_rewards.clone(),
            valid_mask=sample.valid_mask.clone(),
        )
        sb = self._static_batch

        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(device=dev))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.opt.zero_grad(set_to_none=False)
                loss_w, _ = self.wm.window_loss(
                    states=sb.states,
                    policy_actions=sb.policy_actions,
                    arm_targets=sb.arm_targets,
                    xi=sb.xi,
                    true_contacts=sb.next_contacts,
                    true_impacts=sb.impact_rewards,
                    valid_mask=sb.valid_mask,
                )
                loss_w.backward()
        torch.cuda.current_stream(device=dev).wait_stream(stream)

        self._wm_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._wm_graph, stream=stream):
            self.opt.zero_grad(set_to_none=False)
            self._static_loss, self._static_metrics = self.wm.window_loss(
                states=sb.states,
                policy_actions=sb.policy_actions,
                arm_targets=sb.arm_targets,
                xi=sb.xi,
                true_contacts=sb.next_contacts,
                true_impacts=sb.impact_rewards,
                valid_mask=sb.valid_mask,
            )
            self._static_loss.backward()
        self.opt.zero_grad(set_to_none=False)

    def _copy_to_static(self, w: WMWindowBatch) -> None:
        assert self._static_batch is not None
        sb = self._static_batch
        sb.states.copy_(w.states)
        sb.policy_actions.copy_(w.policy_actions)
        sb.arm_targets.copy_(w.arm_targets)
        sb.xi.copy_(w.xi)
        sb.next_contacts.copy_(w.next_contacts)
        sb.impact_rewards.copy_(w.impact_rewards)
        sb.valid_mask.copy_(w.valid_mask)

    def update(self, buffer: G1ReplayBuffer, xi_per_env: torch.Tensor, num_batches: int | None = None) -> dict[str, float]:
        n_batches = int(num_batches if num_batches is not None else self.cfg.batches_per_update)
        agg: dict[str, torch.Tensor] = {}

        for _ in range(n_batches):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window_k, xi_per_env)
            if self.cfg.use_mirror_augmentation and self.symmetry is not None:
                self.symmetry.augment_wm_window_half_(w)
            if self.cfg.use_cudagraph and w.states.is_cuda:
                if self._wm_graph is None:
                    self._capture_wm_graph(w)
                self._copy_to_static(w)
                assert self._wm_graph is not None and self._static_metrics is not None
                self._wm_graph.replay()
                metrics = self._static_metrics
            else:
                loss, metrics = self.wm.window_loss(
                    states=w.states,
                    policy_actions=w.policy_actions,
                    arm_targets=w.arm_targets,
                    xi=w.xi,
                    true_contacts=w.next_contacts,
                    true_impacts=w.impact_rewards,
                    valid_mask=w.valid_mask,
                )
                self.opt.zero_grad(set_to_none=False)
                loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip_norm)
            self.opt.step()

            for k_name, val in metrics.items():
                agg[k_name] = agg[k_name] + val if k_name in agg else val.clone()
            agg["grad_norm"] = agg["grad_norm"] + gn.detach() if "grad_norm" in agg else gn.detach().clone()

        return {f"wm/{k_name}": float(val.item()) / n_batches for k_name, val in agg.items()}


class PolicyCriticTrainer:
    """Short-horizon Analytic BPTT Actor + Survival-Weighted GAE Critic trainer with CUDA Graph capture."""

    def __init__(
        self,
        wm: G1WorldModel,
        actor: TeacherActorMLP,
        critic: PrivilegedCriticMLP,
        diff_rewards: DifferentiableG1Rewards,
        cfg: PolicyTrainerCfg,
        symmetry: G1SagittalSymmetry | None = None,
    ):
        self.wm = wm
        self.actor = actor
        self.critic = critic
        self.critic_target = critic.make_ema_target()
        self.diff_rewards = diff_rewards
        self.cfg = cfg
        self.symmetry = symmetry

        use_fused = next(actor.parameters()).is_cuda
        self.actor_opt = torch.optim.Adam(
            actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
            fused=use_fused,
        )
        self.critic_opt = torch.optim.Adam(
            critic.net.parameters(),
            lr=cfg.critic_lr,
            fused=use_fused,
        )
        self._bptt_graph: torch.cuda.CUDAGraph | None = None
        self._static_start: ImagStartBatch | None = None
        self._static_ro: dict[str, torch.Tensor] | None = None
        self._static_actor_loss: torch.Tensor | None = None
        actor_dev = next(actor.parameters()).device
        self._cur_sym_weight: float = float(cfg.symmetry_weight)
        self._sym_weight_buf = torch.tensor(float(cfg.symmetry_weight), dtype=torch.float32, device=actor_dev)

    def set_lr(self, actor_lr: float, critic_lr: float | None = None) -> None:
        for g in self.actor_opt.param_groups:
            g["lr"] = actor_lr
        c_lr = critic_lr if critic_lr is not None else actor_lr
        for g in self.critic_opt.param_groups:
            g["lr"] = c_lr

    def set_symmetry_weight(self, w: float) -> None:
        """Update effective L_sym weight in-place on GPU so captured CUDA Graph uses the live scheduled weight."""
        self._cur_sym_weight = float(w)
        self._sym_weight_buf.fill_(float(w))

    def _single_imag_step(
        self,
        s_tau: torch.Tensor,
        feat_tau: torch.Tensor,
        prev_act: torch.Tensor,
        p_foot_tau: torch.Tensor,
        ct_tau: torch.Tensor,
        at_tau: torch.Tensor,
        cmd_tau: torch.Tensor,
        arm_tau: torch.Tensor,
        base_command: torch.Tensor,
        heading_target: torch.Tensor,
        is_heading_env: torch.Tensor,
        is_standing_env: torch.Tensor,
        xi: torch.Tensor,
        xi_norm: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        dt = self.diff_rewards.dt
        obs_tau = self.diff_rewards.build_actor_obs(
            s=s_tau,
            command=cmd_tau,
            prev_action=prev_act,
            foot_contact_prob=p_foot_tau,
            add_noise=self.cfg.obs_noise_in_imagination,
            ste_contacts=True,
        )
        a_tau = self.actor(obs_tau)

        s_next, feat_next, c_logits_next, r_imp = self.wm.step_with_features(
            s=s_tau,
            policy_action=a_tau,
            arm_target=arm_tau,
            xi_norm=xi_norm,
            feat_s=feat_tau,
        )
        p_foot_next = torch.sigmoid(c_logits_next)
        ct_next = p_foot_next * (ct_tau + dt)
        at_next = (1.0 - p_foot_next) * (at_tau + dt)

        cmd_next = self.diff_rewards.update_command_in_imagination(
            s_next,
            base_command,
            heading_target,
            is_heading_env,
            is_standing_env,
        )

        r_next, d_next, _ = self.diff_rewards(
            s_prev=s_tau,
            s_next=s_next,
            action=a_tau,
            prev_action=prev_act,
            arm_target=arm_tau,
            command=cmd_next,
            is_standing_env=is_standing_env,
            xi=xi,
            foot_contact_prob=p_foot_next,
            contact_time_next=ct_next,
            air_time_next=at_next,
            r_impact_pred=r_imp,
            return_diag=False,
        )
        return obs_tau.detach(), a_tau, s_next, feat_next, p_foot_next, ct_next, at_next, cmd_next, r_next, d_next

    def _rollout_imagination(
        self,
        start: ImagStartBatch,
        need_grad: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Unroll H steps in the Differentiable World Model."""
        cfg = self.cfg
        H = cfg.horizon

        xi_norm = self.wm.spec.normalize_xi(start.xi)
        s_tau = start.state
        feat_tau = self.wm.spec.canonical(s_tau)
        prev_act = start.prev_action
        p_foot_tau = start.feet_contact
        ct_tau = start.contact_time
        at_tau = start.air_time
        cmd_tau = self.diff_rewards.update_command_in_imagination(
            s_tau,
            start.command,
            start.heading_target,
            start.is_heading_env,
            start.is_standing_env,
        )

        B = s_tau.shape[0]
        survival = s_tau.new_ones(B)
        J = s_tau.new_zeros(B)
        ood_pen = s_tau.new_zeros(())
        sym_pen = s_tau.new_zeros(())
        sym_gate_frac = s_tau.new_zeros(())

        critic_xs: list[torch.Tensor] = []
        obs_det_list: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []
        dones_list: list[torch.Tensor] = []
        survivals_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []

        grad_ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with grad_ctx:
            for tau in range(H):
                critic_xs.append(
                    PrivilegedCriticMLP.build_features(
                        feat_tau.detach(),
                        cmd_tau.detach(),
                        xi_norm,
                        p_foot_tau.detach(),
                        ct_tau.detach(),
                        at_tau.detach(),
                    )
                )

                (
                    obs_det,
                    a_tau,
                    s_next,
                    feat_next,
                    p_foot_next,
                    ct_next,
                    at_next,
                    cmd_next,
                    r_next,
                    d_next,
                ) = self._single_imag_step(
                    s_tau,
                    feat_tau,
                    prev_act,
                    p_foot_tau,
                    ct_tau,
                    at_tau,
                    cmd_tau,
                    start.arm_future[tau],
                    start.command,
                    start.heading_target,
                    start.is_heading_env,
                    start.is_standing_env,
                    start.xi,
                    xi_norm,
                )
                obs_det_list.append(obs_det)

                if cfg.ood_action_weight > 0.0:
                    excess = torch.relu(a_tau.abs() - cfg.ood_action_limit)
                    ood_pen = ood_pen + (excess * excess).sum(dim=-1).mean() / H

                J = J + (cfg.gamma**tau) * survival * r_next

                rewards_list.append(r_next.detach())
                dones_list.append(d_next.detach())
                survivals_list.append(survival.detach())
                actions_list.append(a_tau.detach())

                survival = survival * (1.0 - d_next)
                s_tau = s_next
                feat_tau = feat_next
                prev_act = a_tau
                p_foot_tau = p_foot_next
                ct_tau = ct_next
                at_tau = at_next
                cmd_tau = cmd_next

            # Terminal step H features (keep graph connected for analytic dV(s_H)/ds_H!)
            x_H = PrivilegedCriticMLP.build_features(
                feat_tau,
                cmd_tau,
                xi_norm,
                p_foot_tau,
                ct_tau,
                at_tau,
            )
            if cfg.use_critic_bootstrap:
                v_H = self.critic(x_H)
                J = J + (cfg.gamma**H) * survival * v_H

            # Phase-asymmetry-gated sagittal mirror-equivariance regularization over all H*B imagined observations
            # (evaluated on detached obs_all so gradients strictly regularize pi_theta without steering dynamics,
            # and gated to 0 on symmetric double-support standing so step initiation is never locked)
            if self.symmetry is not None and cfg.symmetry_weight > 0.0:
                obs_all = torch.cat(obs_det_list, dim=0)  # (H*B, 60)
                obs_mir = self.symmetry.mirror_obs(obs_all)
                sym_gate = self.symmetry.obs_asymmetry_gate(
                    obs_all, obs_mir, tau_low=cfg.sym_tau_low, tau_high=cfg.sym_tau_high
                )
                a_both = self.actor(torch.cat([obs_all, obs_mir], dim=0))  # (2*H*B, 15)
                hb = obs_all.shape[0]
                a_orig = a_both[:hb]
                a_mir = self.symmetry.mirror_act(a_both[hb:])
                per_obs_mse = ((a_orig - a_mir) ** 2).mean(dim=-1)
                sym_pen = (sym_gate * per_obs_mse).mean()
                sym_gate_frac = sym_gate.mean().detach()

        X_all = torch.stack(critic_xs + [x_H.detach()], dim=0)  # (H+1, B, D)
        return {
            "J": J,
            "ood_pen": ood_pen,
            "sym_pen": sym_pen,
            "sym_gate_frac": sym_gate_frac,
            "X_all": X_all,
            "rewards": torch.stack(rewards_list, dim=0),  # (H, B)
            "dones": torch.stack(dones_list, dim=0),  # (H, B)
            "survivals": torch.stack(survivals_list, dim=0),  # (H, B)
            "final_survival": survival.detach(),
            "actions": torch.stack(actions_list, dim=0),  # (H, B, 15)
        }

    def _actor_objective_loss(self, ro: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        loss = -ro["J"].mean() / cfg.horizon + cfg.ood_action_weight * ro["ood_pen"]
        if self.symmetry is not None and cfg.symmetry_weight > 0.0:
            loss = loss + self._sym_weight_buf * ro["sym_pen"]
        return loss

    def _capture_bptt_graph(self, sample: ImagStartBatch) -> None:
        dev = sample.state.device
        self._static_start = ImagStartBatch(
            state=sample.state.clone(),
            prev_action=sample.prev_action.clone(),
            feet_contact=sample.feet_contact.clone(),
            contact_time=sample.contact_time.clone(),
            air_time=sample.air_time.clone(),
            command=sample.command.clone(),
            heading_target=sample.heading_target.clone(),
            is_heading_env=sample.is_heading_env.clone(),
            is_standing_env=sample.is_standing_env.clone(),
            xi=sample.xi.clone(),
            arm_future=sample.arm_future.clone(),
        )
        ss = self._static_start

        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            stream = torch.cuda.Stream(device=dev)
            stream.wait_stream(torch.cuda.current_stream(device=dev))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.actor_opt.zero_grad(set_to_none=False)
                    ro_w = self._rollout_imagination(ss, need_grad=True)
                    loss_w = self._actor_objective_loss(ro_w)
                    loss_w.backward()
            torch.cuda.current_stream(device=dev).wait_stream(stream)

            self._bptt_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._bptt_graph, stream=stream):
                self.actor_opt.zero_grad(set_to_none=False)
                ro_g = self._rollout_imagination(ss, need_grad=True)
                loss_g = self._actor_objective_loss(ro_g)
                loss_g.backward()
                self._static_ro = ro_g
                self._static_actor_loss = loss_g.detach()
            self.actor_opt.zero_grad(set_to_none=False)
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)

    def _copy_start_to_static(self, start: ImagStartBatch) -> None:
        assert self._static_start is not None
        ss = self._static_start
        ss.state.copy_(start.state)
        ss.prev_action.copy_(start.prev_action)
        ss.feet_contact.copy_(start.feet_contact)
        ss.contact_time.copy_(start.contact_time)
        ss.air_time.copy_(start.air_time)
        ss.command.copy_(start.command)
        ss.heading_target.copy_(start.heading_target)
        ss.is_heading_env.copy_(start.is_heading_env)
        ss.is_standing_env.copy_(start.is_standing_env)
        ss.xi.copy_(start.xi)
        ss.arm_future.copy_(start.arm_future)

    def _fit_critic(
        self,
        X_all: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        survivals: torch.Tensor,
    ) -> dict[str, float]:
        """Fit PrivilegedCriticMLP to survival-weighted GAE targets from EMA target critic."""
        cfg = self.cfg
        H, B = rewards.shape

        self.critic.norm.update(X_all)
        self.critic_target.norm.mean.copy_(self.critic.norm.mean)
        self.critic_target.norm.var.copy_(self.critic.norm.var)
        self.critic_target.norm.count.copy_(self.critic.norm.count)

        with torch.no_grad():
            V_tgt = self.critic_target(X_all.reshape((H + 1) * B, -1)).reshape(H + 1, B)
            returns = torch.zeros_like(V_tgt)
            returns[H] = V_tgt[H]
            for tau in range(H - 1, -1, -1):
                bootstrap = (1.0 - cfg.gae_lambda) * V_tgt[tau + 1] + cfg.gae_lambda * returns[tau + 1]
                returns[tau] = rewards[tau] + cfg.gamma * (1.0 - dones[tau]) * bootstrap

        X_flat = X_all[:H].reshape(H * B, -1)
        R_flat = returns[:H].reshape(H * B)
        W_flat = survivals.reshape(H * B)

        N = X_flat.shape[0]
        mb = max(N // cfg.critic_minibatches, 1)
        loss_sum_t = X_flat.new_zeros(())
        n_steps = 0

        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(N, device=X_flat.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                pred = self.critic(X_flat[idx])
                w_mb = W_flat[idx]
                sq_err = (pred - R_flat[idx]) ** 2
                c_loss = (sq_err * w_mb).sum() / w_mb.sum().clamp_min(1.0)

                self.critic_opt.zero_grad(set_to_none=False)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(), cfg.critic_grad_clip_norm)
                self.critic_opt.step()

                loss_sum_t = loss_sum_t + c_loss.detach()
                n_steps += 1

        if cfg.critic_ema_tau > 0.0:
            self.critic.update_ema_target(self.critic_target, cfg.critic_ema_tau)
        else:
            self.critic.update_ema_target(self.critic_target, 1.0)

        return {
            "critic/loss": float(loss_sum_t.item()) / max(n_steps, 1),
            "critic/value_mean": float(V_tgt[0].mean().item()),
            "critic/target_mean": float(returns[0].mean().item()),
        }

    def pretrain_critic_only(self, buffer: G1ReplayBuffer, xi_per_env: torch.Tensor, num_updates: int = 10) -> dict[str, float]:
        """Calibrate PrivilegedCriticMLP on imagined rollouts before the first Actor BPTT step."""
        last_stats: dict[str, float] = {}
        for _ in range(num_updates):
            start = buffer.sample_starts(self.cfg.batch_size, self.cfg.horizon, xi_per_env)
            if self.cfg.use_mirror_augmentation and self.symmetry is not None and self._cur_sym_weight > 0.0:
                self.symmetry.augment_imag_start_half_(start, min_asym_radius=self.cfg.sym_tau_high)
            ro = self._rollout_imagination(start, need_grad=False)
            last_stats = self._fit_critic(
                X_all=ro["X_all"],
                rewards=ro["rewards"],
                dones=ro["dones"],
                survivals=ro["survivals"],
            )
        # Sync target network fully after initial calibration
        self.critic.update_ema_target(self.critic_target, 1.0)
        return last_stats

    def update(self, buffer: G1ReplayBuffer, xi_per_env: torch.Tensor) -> dict[str, float]:
        """Run one H-step Analytic BPTT Actor update followed by GAE Critic update."""
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.horizon, xi_per_env)
        if cfg.use_mirror_augmentation and self.symmetry is not None and self._cur_sym_weight > 0.0:
            self.symmetry.augment_imag_start_half_(start, min_asym_radius=cfg.sym_tau_high)

        if cfg.use_cudagraph and start.state.is_cuda:
            if self._bptt_graph is None:
                self._capture_bptt_graph(start)
            self._copy_start_to_static(start)
            assert self._bptt_graph is not None and self._static_ro is not None and self._static_actor_loss is not None
            self._bptt_graph.replay()
            ro = self._static_ro
            actor_loss = self._static_actor_loss

            gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
            gn_val = float(gn.detach().item())
            has_nonfinite = not math.isfinite(gn_val)
            if has_nonfinite:
                self.actor_opt.zero_grad(set_to_none=False)
            else:
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), cfg.grad_clip_value)
                self.actor_opt.step()
        else:
            self.wm.requires_grad_(False)
            self.critic.requires_grad_(False)
            try:
                ro = self._rollout_imagination(start, need_grad=True)
                actor_loss = self._actor_objective_loss(ro)

                self.actor_opt.zero_grad(set_to_none=False)
                actor_loss.backward()

                gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
                gn_val = float(gn.detach().item())
                has_nonfinite = not math.isfinite(gn_val)
                if has_nonfinite:
                    self.actor_opt.zero_grad(set_to_none=False)
                else:
                    torch.nn.utils.clip_grad_value_(self.actor.parameters(), cfg.grad_clip_value)
                    self.actor_opt.step()
            finally:
                self.wm.requires_grad_(True)
                self.critic.requires_grad_(True)

        J: torch.Tensor = ro["J"]
        sym_pen: torch.Tensor = ro["sym_pen"]
        sym_gate_frac: torch.Tensor = ro["sym_gate_frac"]
        actions: torch.Tensor = ro["actions"]
        rewards: torch.Tensor = ro["rewards"]
        dones: torch.Tensor = ro["dones"]
        survivals: torch.Tensor = ro["survivals"]
        final_survival: torch.Tensor = ro["final_survival"]

        metrics: dict[str, float] = {
            "actor/objective_J": float(J.detach().mean().item()),
            "actor/loss": float(actor_loss.detach().item()),
            "actor/sym_mse": float(sym_pen.detach().item()),
            "actor/sym_gate_frac": float(sym_gate_frac.detach().item()),
            "actor/sym_weight": float(self._cur_sym_weight),
            "actor/grad_norm": gn_val,
            "actor/nonfinite_skipped": 1.0 if has_nonfinite else 0.0,
            "actor/action_abs_mean": float(actions.abs().mean().item()),
            "actor/action_abs_max": float(actions.abs().max().item()),
            "imag/step_reward_mean": float(rewards.mean().item()),
            "imag/undisc_return": float((survivals * rewards).sum(dim=0).mean().item()),
            "imag/survival_H": float(final_survival.mean().item()),
            "imag/soft_done_mean": float(dones.mean().item()),
        }

        critic_metrics = self._fit_critic(
            X_all=ro["X_all"],
            rewards=rewards,
            dones=dones,
            survivals=survivals,
        )
        metrics.update(critic_metrics)
        return metrics

"""Differentiable World Model for Reference-Free Unitree G1 Whole-Body Control.

Components:
1. `DynamicsMLP`: (canonical_state 71 + scaled_policy_act 15 + arm_rel_tgt 14 + xi_norm 4 = 104)
   -> MLP(2048, 2048, 2048, SiLU, zero-init output) -> 70-D manifold residual delta.
   Note: Command `c_t` is strictly excluded from dynamics to preserve pure physical Markovianity.
2. `FootContactHead`: canonical_state(71) -> MLP(128, 128) -> 2 foot contact logits,
   supervised via BCE on true `feet_contact` labels. Supplies both STE `feet_contact` in
   imagined 60-D Teacher Actor observations and multilinear soft gait-timer transitions.
3. `ImpactPenaltyHead`: (feat_t 71 + act 15 + arm 14 + feat_{t+1} 71 = 171) -> MLP(128, 128)
   -> non-positive `-0.01 * Softplus(z - 3.0) <= 0`, supervised via MSE on the true simulator
   contact-force penalties (`undesired_contacts + feet_force + feet_stumble`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_lab.supermimic.state_spec import DELTA_DIM, FEAT_DIM, STATE_DIM, XI_DIM, StateSpec, StateSpecCfg


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        act_cls=nn.SiLU,
        zero_last: bool = False,
        last_gain: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act_cls())
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    aux_hidden: tuple[int, ...] = (128, 128)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    contact_loss_weight: float = 0.5
    impact_loss_weight: float = 10.0
    use_bf16_dynamics: bool = True


class G1WorldModel(nn.Module):
    def __init__(
        self,
        cfg: WorldModelCfg,
        action_scale: torch.Tensor,
        default_arm_pos: torch.Tensor,
        clip_actions: float = 100.0,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__()
        self.cfg = cfg
        self.clip_actions = float(clip_actions)
        self.spec = StateSpec(cfg.state_spec, device=device)

        self.register_buffer("action_scale", action_scale.to(device=device, dtype=torch.float32).clone())
        self.register_buffer("default_arm_pos", default_arm_pos.to(device=device, dtype=torch.float32).clone())

        dyn_in = FEAT_DIM + 15 + 14 + XI_DIM  # 71 + 15 + 14 + 4 = 104
        self.dynamics = MLP(dyn_in, tuple(cfg.hidden), DELTA_DIM, act_cls=nn.SiLU, zero_last=True)
        self.contact_head = MLP(FEAT_DIM, tuple(cfg.aux_hidden), 2, act_cls=nn.SiLU)
        self.impact_head = MLP(FEAT_DIM + 15 + 14 + FEAT_DIM, tuple(cfg.aux_hidden), 1, act_cls=nn.SiLU, zero_last=True)

        self.to(device)

    def _encode_actions(self, policy_action: torch.Tensor, arm_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Express both policy 15-DoF actions and exogenous 14-DoF arm targets as radian offsets."""
        a_scaled = torch.clamp(policy_action, -self.clip_actions, self.clip_actions) * self.action_scale
        arm_rel = arm_target - self.default_arm_pos
        return a_scaled, arm_rel

    def _run_dynamics_mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_bf16_dynamics and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.dynamics(x).float()
        return self.dynamics(x)

    def step_with_features(
        self,
        s: torch.Tensor,
        policy_action: torch.Tensor,
        arm_target: torch.Tensor,
        xi_norm: torch.Tensor,
        feat_s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance state one 50Hz step and predict foot contact probabilities & non-positive impact penalty.

        Returns:
            s_next: (..., 71)
            feat_next: (..., 71) canonical features of s_next
            contact_logits: (..., 2)
            r_impact: (...,) strictly <= 0
        """
        feat_t = feat_s if feat_s is not None else self.spec.canonical(s)
        a_scaled, arm_rel = self._encode_actions(policy_action, arm_target)
        dyn_input = torch.cat([feat_t, a_scaled, arm_rel, xi_norm], dim=-1)
        delta = self._run_dynamics_mlp(dyn_input)
        s_next = self.spec.integrate(s, delta)

        feat_next = self.spec.canonical(s_next)
        contact_logits = self.contact_head(feat_next)
        imp_in = torch.cat([feat_t, a_scaled, arm_rel, feat_next], dim=-1)
        imp_raw = self.impact_head(imp_in).squeeze(-1)
        # Strictly non-positive output (~ -4.8e-4 at zero init) prevents reward hacking
        r_impact = -0.01 * F.softplus(imp_raw - 3.0)
        return s_next, feat_next, contact_logits, r_impact

    def step(self, s: torch.Tensor, policy_action: torch.Tensor, arm_target: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        xi_norm = self.spec.normalize_xi(xi)
        s_next, _, _, _ = self.step_with_features(s, policy_action, arm_target, xi_norm)
        return s_next

    def predict_contact_prob(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.contact_head(self.spec.canonical(s)))

    def window_loss(
        self,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
        arm_targets: torch.Tensor,
        xi: torch.Tensor,
        true_contacts: torch.Tensor,
        true_impacts: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """k-step autoregressive World Model loss with episode/push boundary masking.

        Auxiliary heads (`contact_head`, `impact_head`) are trained on both ground-truth features
        and detached rollout features so they stay calibrated on multi-step imagined trajectories
        without distorting `DynamicsMLP`'s pure physical state MSE objective.

        Shapes:
            states: (k+1, B, 71)
            policy_actions: (k, B, 15)
            arm_targets: (k, B, 14)
            xi: (B, 4)
            true_contacts: (k, B, 2) binary contact labels of states[1..k]
            true_impacts: (k, B) non-positive impact reward sums of steps 0..k-1
            valid_mask: optional (k, B) float mask (1.0 = valid smooth same-episode prefix)
        """
        k, B = policy_actions.shape[:2]
        xi_norm = self.spec.normalize_xi(xi)

        # Precompute canonical features and encoded actions across all k steps in one vectorized pass
        feat_true_all = self.spec.canonical(states)  # (k+1, B, 71)
        a_scaled_all, arm_rel_all = self._encode_actions(policy_actions, arm_targets)  # (k, B, 15), (k, B, 14)

        s = states[0]
        feat_t = feat_true_all[0]

        total_state_loss = s.new_zeros(())
        err_1step = s.new_zeros(())
        err_kstep = s.new_zeros(())
        ones_mask = s.new_ones((k, B))
        mask = valid_mask if valid_mask is not None else ones_mask
        denoms = mask.sum(dim=-1).clamp_min(1.0)  # (k,)

        feat_pred_prev_list: list[torch.Tensor] = []
        feat_pred_next_list: list[torch.Tensor] = []

        for j in range(k):
            mask_j = mask[j]
            denom_j = denoms[j]

            feat_pred_prev_list.append(feat_t.detach())
            dyn_input = torch.cat([feat_t, a_scaled_all[j], arm_rel_all[j], xi_norm], dim=-1)
            delta = self._run_dynamics_mlp(dyn_input)
            s_next = self.spec.integrate(s, delta)

            err_vec = self.spec.error_vector(s_next, states[j + 1])
            sq_err = err_vec * err_vec
            total_state_loss = total_state_loss + (sq_err.sum(dim=-1) * mask_j).sum() / denom_j

            with torch.no_grad():
                nse = (sq_err.mean(dim=-1) * mask_j).sum() / denom_j
                if j == 0:
                    err_1step = nse
                err_kstep = nse

            s = s_next
            feat_t = self.spec.canonical(s)
            feat_pred_next_list.append(feat_t.detach())

        # Vectorized auxiliary heads across all k steps at once (2 GEMMs instead of 32!)
        feat_pred_prev = torch.stack(feat_pred_prev_list, dim=0)  # (k, B, 71)
        feat_pred_next = torch.stack(feat_pred_next_list, dim=0)  # (k, B, 71)
        feat_true_prev = feat_true_all[:-1]  # (k, B, 71)
        feat_true_next = feat_true_all[1:]  # (k, B, 71)

        c_logits_both = self.contact_head(torch.cat([feat_true_next, feat_pred_next], dim=0))  # (2k, B, 2)
        c_logits_true, c_logits_pred = c_logits_both[:k], c_logits_both[k:]
        c_bce_per_step_env = 0.5 * (
            F.binary_cross_entropy_with_logits(c_logits_true, true_contacts, reduction="none").mean(dim=-1)
            + F.binary_cross_entropy_with_logits(c_logits_pred, true_contacts, reduction="none").mean(dim=-1)
        )
        contact_loss = ((c_bce_per_step_env * mask).sum(dim=-1) / denoms).mean()

        a_sc_d = a_scaled_all.detach()
        arm_r_d = arm_rel_all.detach()
        imp_in_true = torch.cat([feat_true_prev, a_sc_d, arm_r_d, feat_true_next], dim=-1)
        imp_in_pred = torch.cat([feat_pred_prev, a_sc_d, arm_r_d, feat_pred_next], dim=-1)
        imp_both = -0.01 * F.softplus(
            self.impact_head(torch.cat([imp_in_true, imp_in_pred], dim=0)).squeeze(-1) - 3.0
        )
        imp_true, imp_pred = imp_both[:k], imp_both[k:]
        imp_mse_per_step_env = 0.5 * ((imp_true - true_impacts) ** 2 + (imp_pred - true_impacts) ** 2)
        impact_loss = ((imp_mse_per_step_env * mask).sum(dim=-1) / denoms).mean()

        with torch.no_grad():
            acc_per_step_env = ((c_logits_pred > 0.0).float() == (true_contacts > 0.5).float()).float().mean(dim=-1)
            contact_acc = ((acc_per_step_env * mask).sum(dim=-1) / denoms).mean()
            valid_frac = mask.mean()

        state_loss = total_state_loss / k
        loss = state_loss + self.cfg.contact_loss_weight * contact_loss + self.cfg.impact_loss_weight * impact_loss

        metrics = {
            "loss": loss.detach(),
            "state_loss": state_loss.detach(),
            "contact_bce": contact_loss.detach(),
            "impact_mse": impact_loss.detach(),
            "nse_1step": err_1step.detach(),
            "nse_kstep": err_kstep.detach(),
            "contact_acc": contact_acc.detach(),
            "valid_frac": valid_frac.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def rollout_nse(
        self,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
        arm_targets: torch.Tensor,
        xi: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Open-loop k-step normalized squared error per step (k,)."""
        k, B = policy_actions.shape[:2]
        xi_norm = self.spec.normalize_xi(xi)
        s = states[0]
        feat_t = self.spec.canonical(s)
        ones_mask = s.new_ones(B)
        errs = []
        for j in range(k):
            mask_j = valid_mask[j] if valid_mask is not None else ones_mask
            denom_j = mask_j.sum().clamp_min(1.0)
            s, feat_t, _, _ = self.step_with_features(s, policy_actions[j], arm_targets[j], xi_norm, feat_s=feat_t)
            err_j = (self.spec.normalized_sq_error(s, states[j + 1]) * mask_j).sum() / denom_j
            errs.append(err_j)
        return torch.stack(errs)

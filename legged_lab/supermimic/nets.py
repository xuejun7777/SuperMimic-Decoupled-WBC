"""Apples-to-Apples Actor & Asymmetric Privileged Critic networks for SuperMimic on G1.

`TeacherActorMLP` uses the exact same topology (`60 -> 512 -> 256 -> 128 -> 15`, ELU,
180,367 parameters) as RSL-RL's `ActorCritic.actor` in `BaseAgentCfg` (with small-gain
`orthogonal_(gain=0.01)` output initialization for stable BPTT start), ensuring:
  1. Zero network-capacity confounding in the PPO vs SuperMimic comparison.
  2. Direct checkpoint export to RSL-RL format so `play_teacher.py` and `train_student.py`
     work out-of-the-box without changing a single line of code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from legged_lab.supermimic.state_spec import FEAT_DIM, XI_DIM


@dataclass
class ActorCfg:
    obs_dim: int = 60
    act_dim: int = 15
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    last_gain: float = 0.01


@dataclass
class CriticCfg:
    # canonical_state(71) + command(7) + xi_norm(4) + foot_contact_prob(2) + contact_time(2) + air_time(2) = 88
    in_dim: int = FEAT_DIM + 7 + XI_DIM + 2 + 2 + 2
    hidden_dims: tuple[int, ...] = (512, 256, 128)


def _build_elu_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, last_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ELU())
        d = h
    last = nn.Linear(d, out_dim)
    if last_gain is not None:
        nn.init.orthogonal_(last.weight, gain=last_gain)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class RunningMeanStd(nn.Module):
    """Differentiable running mean/std input normalizer (buffers updated under no_grad)."""

    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1.0e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.shape[0] < 2:
            return
        b_mean = flat.mean(dim=0)
        b_var = flat.var(dim=0, unbiased=False)
        b_count = float(flat.shape[0])

        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + (delta * delta) * (self.count * b_count / tot)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + self.eps)
        return torch.clamp((x - self.mean) / std, -10.0, 10.0)


class TeacherActorMLP(nn.Module):
    """Exact RSL-RL-compatible Teacher Actor: `self.actor` is `nn.Sequential(60->512->256->128->15)`."""

    def __init__(self, cfg: ActorCfg):
        super().__init__()
        self.cfg = cfg
        self.actor = _build_elu_mlp(cfg.obs_dim, tuple(cfg.hidden_dims), cfg.act_dim, last_gain=cfg.last_gain)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def rsl_rl_state_dict(self, critic_hidden: tuple[int, ...] = (512, 256, 128), critic_obs_dim: int = 63, noise_std: float = 0.25) -> dict[str, torch.Tensor]:
        """Build an RSL-RL `ActorCritic` compatible `model_state_dict` for `play_teacher.py` / `train_student.py`."""
        sd: dict[str, torch.Tensor] = {}
        for k, v in self.actor.state_dict().items():
            sd[f"actor.{k}"] = v.detach().clone()
        dummy_critic = _build_elu_mlp(critic_obs_dim, critic_hidden, 1)
        for k, v in dummy_critic.state_dict().items():
            sd[f"critic.{k}"] = v.to(device=next(self.parameters()).device).detach().clone()
        # BaseAgentCfg uses noise_std_type="scalar" -> scalar `std` parameter of shape (15,) or (1,)
        # Note: RSL-RL 2.3 ActorCritic with noise_std_type="scalar" uses nn.Parameter(init_noise_std * torch.ones(num_actions))
        dev = next(self.parameters()).device
        sd["std"] = torch.full((self.cfg.act_dim,), float(noise_std), device=dev)
        return sd


class PrivilegedCriticMLP(nn.Module):
    """Asymmetric privileged value network V(phi(s), cmd, xi_norm, p_foot, t_cont, t_air) -> scalar."""

    def __init__(self, cfg: CriticCfg):
        super().__init__()
        self.cfg = cfg
        self.norm = RunningMeanStd(cfg.in_dim)
        self.net = _build_elu_mlp(cfg.in_dim, tuple(cfg.hidden_dims), 1, last_gain=1.0)

    @staticmethod
    def build_features(
        feat_s: torch.Tensor,
        command: torch.Tensor,
        xi_norm: torch.Tensor,
        foot_contact_prob: torch.Tensor,
        contact_time: torch.Tensor,
        air_time: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                feat_s,
                command,
                xi_norm,
                foot_contact_prob,
                torch.clamp(contact_time, 0.0, 2.0),
                torch.clamp(air_time, 0.0, 2.0),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)

    def make_ema_target(self) -> PrivilegedCriticMLP:
        tgt = copy.deepcopy(self)
        tgt.requires_grad_(False)
        return tgt

    @torch.no_grad()
    def update_ema_target(self, target: PrivilegedCriticMLP, tau: float) -> None:
        for p_t, p_o in zip(target.net.parameters(), self.net.parameters(), strict=True):
            p_t.lerp_(p_o, tau)
        for b_t, b_o in zip(target.buffers(), self.buffers(), strict=True):
            b_t.copy_(b_o)

"""71-D full-body Markov state representation and CUDAGraph-safe manifold integration.

State layout `s` (..., 71) in `robot.joint_names` (USD) order:
    0:3    root_link_pos_local   (pelvis link pos minus env_origin, meters; z == world z on flat terrain)
    3:7    root_link_quat_w      (pelvis link orientation in world frame, wxyz)
    7:36   joint_pos             (all 29 G1 hinge positions, rad, USD order)
    36:39  root_link_lin_vel_w   (pelvis link origin linear velocity in world frame, m/s)
    39:42  root_link_ang_vel_b   (pelvis angular velocity in pelvis body frame, rad/s)
    42:71  joint_vel             (all 29 G1 hinge angular rates, rad/s, USD order)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from legged_lab.supermimic.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NUM_JOINTS = 29
XI_DIM = 4


def _canonical_fn(s: torch.Tensor, in_scale: torch.Tensor) -> torch.Tensor:
    p = s[..., 0:3]
    q = quat_normalize(s[..., 3:7])
    qj = s[..., 7:36]
    v = s[..., 36:39]
    w = s[..., 39:42]
    qd = s[..., 42:71]
    hq = heading_quat(q)
    q_local = quat_canonical(quat_mul(quat_inv(hq), q))
    v_h = quat_apply_inverse(hq, v)
    feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
    return feat / in_scale


def _integrate_fn(s: torch.Tensor, delta: torch.Tensor, out_scale: torch.Tensor, dt: float) -> torch.Tensor:
    d = delta * out_scale
    dv_h = d[..., 0:3]
    dw_b = d[..., 3:6]
    dqd = d[..., 6:35]
    dp_h = d[..., 35:38]
    dth_b = d[..., 38:41]
    dq = d[..., 41:70]

    p = s[..., 0:3]
    q = quat_normalize(s[..., 3:7])
    qj = s[..., 7:36]
    v = s[..., 36:39]
    w = s[..., 39:42]
    qd = s[..., 42:71]
    hq = heading_quat(q)

    v_new = v + quat_apply(hq, dv_h)
    w_new = w + dw_b
    qd_new = qd + dqd

    p_new = p + dt * v_new + quat_apply(hq, dp_h)
    q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
    qj_new = qj + dt * qd_new + dq
    return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)


def _error_vector_fn(s_pred: torch.Tensor, s_true: torch.Tensor, err_scale: torch.Tensor) -> torch.Tensor:
    p1 = s_pred[..., 0:3]
    q1 = quat_normalize(s_pred[..., 3:7])
    qj1 = s_pred[..., 7:36]
    v1 = s_pred[..., 36:39]
    w1 = s_pred[..., 39:42]
    qd1 = s_pred[..., 42:71]

    p2 = s_true[..., 0:3]
    q2 = quat_normalize(s_true[..., 3:7])
    qj2 = s_true[..., 7:36]
    v2 = s_true[..., 36:39]
    w2 = s_true[..., 39:42]
    qd2 = s_true[..., 42:71]

    rot_err = rotvec_from_quat(quat_mul(quat_inv(q2), q1))
    err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
    return err / err_scale


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    xi_offset: tuple[float, ...] = (0.8, 0.6, 0.0025, 0.0)
    xi_scale: tuple[float, ...] = (0.2, 0.2, 0.0025, 5.0)


class StateSpec(nn.Module):
    def __init__(self, cfg: StateSpecCfg, device: str | torch.device = "cuda:0", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.dt = float(cfg.dt) if cfg.explicit_integration else 0.0

        def _rep(scales: tuple[float, ...], sizes: tuple[int, ...]) -> torch.Tensor:
            return torch.cat([torch.full((n,), float(sc), dtype=dtype, device=device) for sc, n in zip(scales, sizes)])

        self.register_buffer("err_scale", _rep(cfg.err_scale, (3, 3, 3, 3, NUM_JOINTS, NUM_JOINTS)))
        self.register_buffer("out_scale", _rep(cfg.out_scale, (3, 3, NUM_JOINTS, 3, 3, NUM_JOINTS)))
        self.register_buffer("in_scale", _rep(cfg.in_scale, (1, 6, 3, 3, NUM_JOINTS, NUM_JOINTS)))
        self.register_buffer("xi_offset", torch.tensor(cfg.xi_offset, dtype=dtype, device=device))
        self.register_buffer("xi_scale", torch.tensor(cfg.xi_scale, dtype=dtype, device=device))

    @staticmethod
    def split(s: torch.Tensor):
        """Split (..., 71) state into (p, q_wxyz, qj, v_link_w, w_b, qd)."""
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        return _canonical_fn(s, self.in_scale)

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return _integrate_fn(s, delta, self.out_scale, self.dt)

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        return _error_vector_fn(s_pred, s_true, self.err_scale)

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        e = _error_vector_fn(s_pred, s_true, self.err_scale)
        return (e * e).mean(dim=-1)

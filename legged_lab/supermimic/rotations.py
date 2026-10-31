"""Differentiable CUDAGraph-compatible batched rotation helpers in wxyz quaternion convention.

Matches IsaacLab (`isaaclab.utils.math`) wxyz conventions everywhere using pure stream-safe
ATen CUDA tensor operations so multi-step World Model fitting and H-step Analytic BPTT
capture cleanly into native `torch.cuda.CUDAGraph`s without Triton or legacy-stream overhead.
"""

from __future__ import annotations

import math

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so w >= 0 to remove quaternion double-cover ambiguity."""
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion inverse (conjugate)."""
    qn = quat_normalize(q)
    return torch.cat([qn[..., :1], -qn[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector `v` by unit wxyz quaternion `q` via Rodrigues cross-product formula."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector `v` by inverse (conjugate) of unit wxyz quaternion `q`."""
    w = q[..., :1]
    u = -q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Construct unit wxyz quaternion from angle (...,) and unit axis (..., 3)."""
    half = 0.5 * angle.unsqueeze(-1)
    axis_n = axis / axis.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.cat([torch.cos(half), axis_n * torch.sin(half)], dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential map: rotation vector (..., 3) -> unit quaternion wxyz (..., 4)."""
    angle_sq = (v * v).sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1.0e-30))
    half = 0.5 * angle
    small = angle < 1.0e-4
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(half) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(half))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithmic map: unit quaternion wxyz (..., 4) -> rotation vector (..., 3)."""
    qn = quat_canonical(quat_normalize(q))
    w = qn[..., :1]
    xyz = qn[..., 1:]
    s = xyz.norm(p=2, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1.0e-6
    safe_s = torch.where(small, torch.ones_like(s), s)
    k = torch.where(small, 2.0 + (angle * angle) / 12.0, angle / safe_s)
    return xyz * k


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of 3x3 rotation matrix flattened row-wise to match matrix_from_quat[..., :, :2].reshape(..., 6)."""
    qn = quat_normalize(q)
    w, x, y, z = qn.unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    return torch.stack([r00, r01, r10, r11, r20, r21], dim=-1)


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle around world Z from wxyz quaternion."""
    w, x, y, z = q.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Pure-yaw heading quaternion (differentiable via atan2, matching math_utils.yaw_quat)."""
    return quat_from_yaw(heading_yaw(q))


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def projected_gravity(q_body_w: torch.Tensor) -> torch.Tensor:
    """World gravity (0, 0, -1) expressed in body frame (..., 3)."""
    qn = quat_normalize(q_body_w)
    w, x, y, z = qn.unbind(dim=-1)
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return torch.stack([gx, gy, gz], dim=-1)


def smooth_euler_xyz_from_quat(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable XYZ Euler angles matching IsaacLab's euler_xyz_from_quat + wrap_to_pi."""
    qn = quat_normalize(q)
    w, x, y, z = qn.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = (2.0 * (w * y - z * x)).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

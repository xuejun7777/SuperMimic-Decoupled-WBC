"""Record an HD 50fps Side-by-Side Choreographed Loco-Manipulation WBC Demo Video.

Compares Reference-Free SuperMimic (`SM_3000`) vs Native RSL-RL PPO (`PPO_3000`) under
identical initial states, identical continuous AMASS bimanual arm-swing disturbances,
and an 18-second (900-step) 6-phase whole-body loco-manipulation command schedule:
  1. 0-3s  : Upright forward walking (`vx=0.45 m/s, h=0.74 m`)
  2. 3-6s  : Lateral sidestepping + yaw turning (`vy=0.35 m/s, wz=0.60 rad/s, h=0.72 m`)
  3. 6-9s  : Continuous deep squat & recovery (`h: 0.72 m -> 0.42 m -> 0.68 m`)
  4. 9-12s : Extreme forward torso bow (`pitch: 0.0 -> 1.15 rad (66 deg), h=0.63 m`)
  5. 12-15s: Torso roll + waist yaw twist (`roll=0.40 rad, yaw=0.80 rad, h=0.72 m`)
  6. 15-18s: Crouched forward-bowing walk (`vx=0.35 m/s, h=0.55 m, pitch=0.50 rad`)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record side-by-side G1 WBC showcase video.")
parser.add_argument("--task", type=str, default="g1_flat")
parser.add_argument("--sm_ckpt", type=str, required=True)
parser.add_argument("--ppo_ckpt", type=str, required=True)
parser.add_argument("--out_mp4", type=str, required=True)
parser.add_argument("--out_grid_png", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=900)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import cv2
import imageio
import isaaclab.utils.math as math_utils
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
import torch
from rsl_rl.runners import OnPolicyRunner

from legged_lab.envs import *  # noqa: F401, F403
import legged_lab.utils.amass as amass_module
from legged_lab.utils import task_registry
from legged_lab.utils.cli_args import apply_trained_actuator_gains

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


def _setup_studio_visuals() -> None:
    """Ensure bright, high-contrast studio lighting and non-black ground material offline."""
    stage = omni.usd.get_context().get_stage()

    # Bind a clean light-slate studio material with subtle contrast onto /World/ground
    mat_path = Sdf.Path("/World/Looks/StudioFloorMat")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendPath("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.68, 0.72, 0.77))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    ground_prim = stage.GetPrimAtPath("/World/ground")
    if ground_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
            mat, UsdShade.Tokens.strongerThanDescendants
        )

    # Spawn visual reference grid tiles around origin so walking/turning translation is unmistakable
    UsdGeom.Xform.Define(stage, "/World/StudioGrid")
    tile_dark = UsdShade.Material.Define(stage, "/World/Looks/TileDark")
    sh_d = UsdShade.Shader.Define(stage, "/World/Looks/TileDark/Shader")
    sh_d.CreateIdAttr("UsdPreviewSurface")
    sh_d.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.52, 0.56, 0.62))
    sh_d.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    tile_dark.CreateSurfaceOutput().ConnectToSource(sh_d.ConnectableAPI(), "surface")

    idx = 0
    for gx in range(-6, 12):
        for gy in range(-6, 7):
            if (gx + gy) % 2 == 0:
                cube = UsdGeom.Cube.Define(stage, f"/World/StudioGrid/t_{idx}")
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube)
                xf.AddTranslateOp().Set(Gf.Vec3d(gx * 0.5, gy * 0.5, -0.004))
                xf.AddScaleOp().Set(Gf.Vec3f(0.49, 0.49, 0.01))
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(tile_dark)
                idx += 1

    # Clean offline DomeLight (replace missing Nucleus HDR texture with bright studio dome)
    dome_prim = stage.GetPrimAtPath("/World/skyLight")
    if dome_prim.IsValid():
        dome = UsdLux.DomeLight(dome_prim)
        dome.CreateTextureFileAttr().Set("")
        dome.CreateColorAttr().Set(Gf.Vec3f(0.90, 0.93, 0.98))
        dome.CreateIntensityAttr().Set(1100.0)


def _command_schedule(step_i: int) -> tuple[str, tuple[float, float, float, float, float, float, float], bool]:
    """Return (phase_title, 7D_command, is_standing) for step `step_i` at 50 Hz."""
    t = step_i * 0.02
    if t < 3.0:
        return ("1/6 Upright Forward Walk (vx=0.45 m/s)", (0.45, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0), False)
    elif t < 6.0:
        return ("2/6 Lateral Walk + Yaw Turn (vy=0.35, wz=0.60)", (0.0, 0.35, 0.60, 0.72, 0.0, 0.0, 0.0), False)
    elif t < 9.0:
        # Smooth cosine squat 0.72 -> 0.42 -> 0.68
        u = (t - 6.0) / 3.0
        if u < 0.65:
            s = 0.5 * (1.0 - math.cos(math.pi * (u / 0.65)))
            h = 0.72 + (0.42 - 0.72) * s
        else:
            s = 0.5 * (1.0 - math.cos(math.pi * ((u - 0.65) / 0.35)))
            h = 0.42 + (0.68 - 0.42) * s
        return (f"3/6 Deep Pelvis Squat (h={h:.2f} m)", (0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0), True)
    elif t < 12.0:
        # Smooth deep forward bow up to 1.15 rad (66 deg) and back to 0.25 rad
        u = (t - 9.0) / 3.0
        if u < 0.70:
            s = 0.5 * (1.0 - math.cos(math.pi * (u / 0.70)))
            p = 1.15 * s
            h = 0.68 - 0.05 * s
        else:
            s = 0.5 * (1.0 - math.cos(math.pi * ((u - 0.70) / 0.30)))
            p = 1.15 * (1.0 - s)
            h = 0.63 + 0.09 * s
        return (f"4/6 Extreme Torso Bow (pitch={p:.2f} rad / {math.degrees(p):.0f} deg)", (0.0, 0.0, 0.0, h, 0.0, p, 0.0), True)
    elif t < 15.0:
        u = min((t - 12.0) / 0.8, 1.0)
        s = 0.5 * (1.0 - math.cos(math.pi * u))
        r = 0.40 * s
        y = 0.80 * s
        return (f"5/6 Torso Roll + Yaw Twist (roll={r:.2f}, yaw={y:.2f})", (0.0, 0.0, 0.0, 0.72, r, 0.0, y), True)
    else:
        return ("6/6 Crouched Bowing Walk (vx=0.35, h=0.55, pitch=0.50)", (0.35, 0.0, 0.0, 0.55, 0.0, 0.50, 0.0), False)


@torch.no_grad()
def _read_telemetry(env, torso_id: int, pelvis_id: int) -> dict[str, float]:
    robot = env.robot
    vel_yaw = quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    wz = float(robot.data.root_ang_vel_w[0, 2].item())
    h = float(robot.data.body_pos_w[0, torso_id, 2].item())

    q_pelvis = robot.data.body_quat_w[:, pelvis_id]
    q_torso = robot.data.body_quat_w[:, torso_id]
    q_pelvis_yaw = math_utils.yaw_quat(q_pelvis)
    q_torso_rel = math_utils.quat_mul(math_utils.quat_conjugate(q_pelvis_yaw), q_torso)
    roll_act, pitch_act, _ = math_utils.euler_xyz_from_quat(q_torso_rel)
    roll_act = float(math_utils.wrap_to_pi(roll_act)[0].item())
    pitch_act = float(math_utils.wrap_to_pi(pitch_act)[0].item())
    q_yaw_rel = math_utils.quat_mul(math_utils.quat_conjugate(q_pelvis_yaw), math_utils.yaw_quat(q_torso))
    yaw_act = float((2.0 * torch.atan2(q_yaw_rel[:, 3], q_yaw_rel[:, 0]))[0].item())

    return {
        "vx": float(vel_yaw[0, 0].item()),
        "vy": float(vel_yaw[0, 1].item()),
        "wz": wz,
        "h": h,
        "roll": roll_act,
        "pitch": pitch_act,
        "yaw": yaw_act,
    }


def _draw_panel_hud(
    frame_rgb: np.ndarray,
    algo_title: str,
    title_rgb: tuple[int, int, int],
    phase_str: str,
    t_sec: float,
    cmd: tuple[float, ...],
    tel: dict[str, float],
    falls: int,
) -> np.ndarray:
    img = frame_rgb.copy()
    # Semi-transparent top dark banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 112), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, algo_title, (14, 27), font, 0.56, title_rgb, 2, cv2.LINE_AA)
    fall_color = (90, 230, 120) if falls == 0 else (255, 95, 95)
    cv2.putText(img, f"t={t_sec:4.1f}s | Falls: {falls}", (img.shape[1] - 182, 27), font, 0.50, fall_color, 1, cv2.LINE_AA)
    cv2.putText(img, phase_str, (14, 54), font, 0.50, (235, 240, 248), 1, cv2.LINE_AA)

    line1 = (
        f"CMD : vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f} | "
        f"h={cmd[3]:.2f}m r={cmd[4]:+.2f} p={cmd[5]:+.2f} y={cmd[6]:+.2f}"
    )
    line2 = (
        f"MEAS: vx={tel['vx']:+.2f} vy={tel['vy']:+.2f} wz={tel['wz']:+.2f} | "
        f"h={tel['h']:.2f}m r={tel['roll']:+.2f} p={tel['pitch']:+.2f} y={tel['yaw']:+.2f}"
    )
    cv2.putText(img, line1, (14, 80), font, 0.42, (175, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(img, line2, (14, 102), font, 0.42, (165, 245, 185), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def _record_single_policy(
    env,
    policy,
    rgb_annotator,
    torso_id: int,
    pelvis_id: int,
    algo_title: str,
    title_rgb: tuple[int, int, int],
    steps: int,
    seed: int,
) -> list[np.ndarray]:
    env.seed(seed)
    if getattr(amass_module, "_global_amass_loader", None) is not None:
        amass_module._global_amass_loader._rng = np.random.default_rng(seed)

    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset(all_ids)
    # Place robot cleanly at world origin facing +X for identical camera framing
    root_state = env.robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins
    env.robot.write_root_pose_to_sim(root_state[:, :7])
    env.robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    env.robot.write_joint_state_to_sim(env.robot.data.default_joint_pos, env.robot.data.default_joint_vel)
    env.scene.write_data_to_sim()
    env.sim.forward()

    # Warm up RTX renderer for 6 frames
    pos0 = env.robot.data.root_pos_w[0].cpu().numpy()
    env.sim.set_camera_view(eye=(pos0[0] + 2.05, pos0[1] - 1.35, 1.26), target=(pos0[0], pos0[1], 0.66))
    for _ in range(6):
        env.sim.render()

    obs, _ = env.get_observations()
    frames: list[np.ndarray] = []
    falls = 0
    cam_xy = np.array([pos0[0], pos0[1]], dtype=np.float64)
    phase6_heading_ref: float | None = None

    for step_i in range(steps):
        t_sec = step_i * 0.02
        phase_str, cmd_tuple, is_standing = _command_schedule(step_i)
        cg = env.command_generator
        cmd_t = torch.tensor(cmd_tuple, dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, 7)
        cg.vel_command_b.copy_(cmd_t)
        # Use BaseEnv's native closed-loop heading hold during straight forward-walk phases (1/6 and 6/6)
        use_heading_hold = (t_sec < 3.0) or (t_sec >= 15.0)
        if t_sec < 3.0:
            cg.heading_target.zero_()
        elif t_sec >= 15.0:
            if phase6_heading_ref is None:
                phase6_heading_ref = float(env.robot.data.heading_w[0].item())
            cg.heading_target.fill_(phase6_heading_ref)
        cg.is_heading_env.fill_(use_heading_hold)
        cg.is_zero_vel_yaw_env.fill_(False)
        cg.is_standing_env.fill_(is_standing)
        cg.time_left.fill_(1.0e6)
        if use_heading_hold:
            cg._update_command()
        eff_cmd_tuple = tuple(float(x) for x in cg.command[0].tolist())
        obs[:, 6:13] = cg.command * env.obs_scales.commands

        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        if bool(((dones > 0) & (~env.time_out_buf))[0].item()):
            falls += 1

        cg.vel_command_b.copy_(torch.tensor(eff_cmd_tuple, dtype=torch.float32, device=env.device).unsqueeze(0))
        tel = _read_telemetry(env, torso_id, pelvis_id)

        # Smooth front-3/4 tracking camera following robot XY
        rpos = env.robot.data.root_pos_w[0].cpu().numpy()
        cam_xy = 0.90 * cam_xy + 0.10 * rpos[:2]
        env.sim.set_camera_view(
            eye=(float(cam_xy[0] + 2.05), float(cam_xy[1] - 1.35), 1.24),
            target=(float(cam_xy[0]), float(cam_xy[1]), 0.64),
        )
        env.sim.render()
        raw = rgb_annotator.get_data()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(*raw.shape)[:, :, :3]
        hud_frame = _draw_panel_hud(arr, algo_title, title_rgb, phase_str, t_sec, eff_cmd_tuple, tel, falls)
        frames.append(hud_frame)

    return frames


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    apply_trained_actuator_gains(env_cfg, os.path.dirname(os.path.abspath(args_cli.sm_ckpt)))
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.commands.curriculum.enable = False
    env_cfg.commands.debug_vis = True
    env_cfg.scene.max_episode_length_s = 30.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.seed = int(args_cli.seed)

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=True)
    _setup_studio_visuals()

    panel_w, panel_h = 640, 720
    render_product = rep.create.render_product("/OmniverseKit_Persp", (panel_w, panel_h))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    torso_id = env.robot.body_names.index("torso_link")
    pelvis_id = env.robot.body_names.index("pelvis")

    # 1. Record Reference-Free SuperMimic (3000 iters)
    sm_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(args_cli.sm_ckpt), device=agent_cfg.device)
    sm_runner.load(os.path.abspath(args_cli.sm_ckpt), load_optimizer=False)
    sm_policy = sm_runner.get_inference_policy(device=env.device)
    print("[VIDEO] Recording Reference-Free SuperMimic (3000 iters)...", flush=True)
    sm_frames = _record_single_policy(
        env, sm_policy, rgb_annotator, torso_id, pelvis_id,
        "SuperMimic (Ours, 3000 iters / 0.295B steps)", (100, 235, 165),
        steps=args_cli.steps, seed=args_cli.seed,
    )

    # 2. Record Native RSL-RL PPO (3000 iters) under exact same seed & AMASS arm motions
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(args_cli.ppo_ckpt), device=agent_cfg.device)
    ppo_runner.load(os.path.abspath(args_cli.ppo_ckpt), load_optimizer=False)
    ppo_policy = ppo_runner.get_inference_policy(device=env.device)
    print("[VIDEO] Recording Native RSL-RL PPO (3000 iters)...", flush=True)
    ppo_frames = _record_single_policy(
        env, ppo_policy, rgb_annotator, torso_id, pelvis_id,
        "Native RSL-RL PPO (3000 iters / 0.295B steps)", (255, 175, 95),
        steps=args_cli.steps, seed=args_cli.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_mp4)), exist_ok=True)
    divider = np.full((panel_h, 4, 3), 230, dtype=np.uint8)
    combined_frames: list[np.ndarray] = []
    writer = imageio.get_writer(
        args_cli.out_mp4,
        fps=50,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    for f_l, f_r in zip(sm_frames, ppo_frames, strict=True):
        merged = np.concatenate([f_l, divider, f_r], axis=1)
        # Pad width from 1284 to 1280 by cropping 2px each outer border for clean 1280x720 16:9
        merged = merged[:, 2:-2, :]
        combined_frames.append(merged)
        writer.append_data(merged)
    writer.close()
    print(f"[VIDEO] Saved 1280x720@50fps MP4 ({len(combined_frames)} frames) -> {args_cli.out_mp4}", flush=True)

    if args_cli.out_grid_png:
        # Save 6 representative phase keyframes (t = 1.5s, 4.5s, 7.8s, 11.0s, 13.8s, 16.5s) as a 3x2 summary sheet
        key_steps = [75, 225, 390, 550, 690, 825]
        rows = []
        for r_idx in range(0, 6, 2):
            f_a = cv2.resize(combined_frames[min(key_steps[r_idx], len(combined_frames) - 1)], (960, 540))
            f_b = cv2.resize(combined_frames[min(key_steps[r_idx + 1], len(combined_frames) - 1)], (960, 540))
            rows.append(np.concatenate([f_a, f_b], axis=1))
        grid = np.concatenate(rows, axis=0)
        imageio.imwrite(args_cli.out_grid_png, grid)
        print(f"[VIDEO] Saved 6-phase keyframe sheet -> {args_cli.out_grid_png}", flush=True)


if __name__ == "__main__":
    import sys
    import traceback

    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

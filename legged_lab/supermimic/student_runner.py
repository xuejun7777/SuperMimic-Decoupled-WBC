"""World-Model Guided Student Policy Trainer with Analytic BPTT, 31-Term Task Reward, and Sagittal Symmetry.

Trains a 116D Proprioceptive Student (`StudentMLP`: 116 -> 512 -> 256 -> 128 -> 15)
by combining:
1. Online DAgger / Teacher Action supervision from the trained SuperMimic Teacher (60D).
2. Continuous online fitting of the Differentiable World Model (`G1WorldModel`) and Privileged Critic (`PrivilegedCriticMLP`)
   on the Student's live PhysX rollouts.
3. H-step Differentiable Imagination Rollouts of the Student directly in `G1WorldModel`,
   backpropagating Analytic BPTT gradients of the full 31-term differentiable reward `DifferentiableG1Rewards`
   + `PrivilegedCriticMLP` terminal value bootstrap.
4. 116D Phase-Gated Sagittal Mirror Symmetry loss (L_sym_116) to eliminate drift under partial observability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.imagination import WMTrainerCfg, WorldModelTrainer
from legged_lab.supermimic.nets import ActorCfg, CriticCfg, PrivilegedCriticMLP, ProprioStudentMLP, TeacherActorMLP, export_student_jit
from legged_lab.supermimic.replay_buffer import G1ReplayBuffer
from legged_lab.supermimic.symmetry import G1SagittalSymmetry
from legged_lab.supermimic.usd_fk import UsdArticulationFK
from legged_lab.supermimic.world_model import G1WorldModel, WorldModelCfg


@dataclass
class SuperMimicStudentCfg:
    # Outer loop & rollout
    num_steps_per_env: int = 24
    utd: float = 0.4
    replay_capacity: int = 1000
    bc_warmup_iters: int = 50
    dagger_warmup_iters: int = 100

    # Student optimization
    student_lr: float = 3.0e-4
    lr_final_ratio: float = 0.15
    grad_clip_norm: float = 1.0

    # Teacher action supervision & annealing
    teacher_anchor_init: float = 2.0
    teacher_anchor_final: float = 0.25

    # Symmetry
    symmetry_weight: float = 1.5
    sym_tau_low: float = 0.18
    sym_tau_high: float = 0.45

    # World Model & Imagination
    horizon: int = 24
    imag_batch: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hindsight_cmd_frac: float = 0.25
    use_mirror_aug: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 250
    use_tensorboard: bool = True

    world_model: WorldModelCfg = None
    wm_trainer: WMTrainerCfg = None
    critic: CriticCfg = None


class SuperMimicStudentRunner:
    def __init__(
        self,
        env: SuperMimicG1Env,
        teacher_ckpt_path: str,
        cfg: SuperMimicStudentCfg,
        log_dir: str,
        resume_student_path: str | None = None,
        device: str | torch.device = "cuda:0",
    ):
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = torch.device(device)
        os.makedirs(self.log_dir, exist_ok=True)

        # 1. Differentiable Kinematics, Symmetry, and 31-term Rewards
        self.fk = UsdArticulationFK(env.robot, device=self.device)
        self.symmetry = G1SagittalSymmetry(env, self.fk, device=self.device)
        self.diff_rewards = DifferentiableG1Rewards(env, self.fk).to(self.device)

        default_arm_pos = env.custom_default_joint_pos[0, env.num_actions : env.num_actions + 14].clone()
        wm_cfg = cfg.world_model if cfg.world_model is not None else WorldModelCfg()
        self.wm = G1WorldModel(
            cfg=wm_cfg,
            action_scale=self.diff_rewards.action_scale,
            default_arm_pos=default_arm_pos,
            clip_actions=env.clip_actions,
            device=self.device,
        )

        # 2. Frozen Teacher Actor
        self.teacher_actor = TeacherActorMLP(ActorCfg(obs_dim=60, act_dim=env.num_actions)).to(self.device)
        self.teacher_actor.eval()
        for p in self.teacher_actor.parameters():
            p.requires_grad = False

        # 3. Privileged Critic & Target Critic
        critic_cfg = cfg.critic if cfg.critic is not None else CriticCfg()
        self.critic = PrivilegedCriticMLP(critic_cfg).to(self.device)
        self.critic_target = self.critic.make_ema_target()

        # 4. Student Policy (116 -> 512 -> 256 -> 128 -> 15)
        self.student = ProprioStudentMLP(obs_dim=116, act_dim=env.num_actions).to(self.device)
        use_fused = self.student.policy[0].weight.is_cuda
        self.student_opt = torch.optim.Adam(
            self.student.parameters(),
            lr=cfg.student_lr,
            fused=use_fused,
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.net.parameters(),
            lr=3.0e-4,
            fused=use_fused,
        )

        wm_tr_cfg = cfg.wm_trainer if cfg.wm_trainer is not None else WMTrainerCfg()
        self.wm_trainer = WorldModelTrainer(self.wm, wm_tr_cfg, symmetry=self.symmetry)

        # Load Teacher weights and warm-start World Model & Critic
        self._load_teacher_checkpoint(teacher_ckpt_path)
        if resume_student_path is not None:
            self._load_student_checkpoint(resume_student_path)

        # 5. Replay Buffer & Proprioceptive Observation History Buffer (2 x 58D)
        self.buffer = G1ReplayBuffer(
            num_envs=env.num_envs,
            capacity=cfg.replay_capacity,
            act_dim=env.num_actions,
            arm_dim=14,
            amass_seq_len=env.sequence_length,
            device=self.device,
        )
        env.attach_replay_buffer(self.buffer)

        # Live history buffer: (num_envs, 2, 58)
        self.obs_hist = torch.zeros(env.num_envs, 2, 58, dtype=torch.float32, device=self.device)

        self.current_learning_iteration: int = 0
        self.total_env_steps: int = 0
        self.total_updates: int = 0
        self.update_credit: float = 0.0
        self.wall_time_s: float = 0.0
        self.best_mean_reward: float = -float("inf")

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if cfg.use_tensorboard else None
        self.jsonl_file = open(os.path.join(self.log_dir, "student_metrics.jsonl"), "a", encoding="utf-8")

    def _load_teacher_checkpoint(self, path: str) -> None:
        target = path
        if os.path.basename(path).startswith("model_"):
            sm_cand = os.path.join(os.path.dirname(path), "supermimic_last.pt")
            if os.path.isfile(sm_cand):
                target = sm_cand
        print(f"[SuperMimicStudent] Loading Teacher checkpoint from {target}", flush=True)
        ck = torch.load(target, map_location=self.device, weights_only=False)
        if "actor" in ck:
            self.teacher_actor.load_state_dict(ck["actor"])
        elif "model_state_dict" in ck:
            sd = {k[len("actor.") :]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
            self.teacher_actor.actor.load_state_dict(sd)

        if "wm" in ck:
            self.wm.load_state_dict(ck["wm"])
            print("[SuperMimicStudent] Warm-started World Model from Teacher", flush=True)
        if "critic" in ck:
            self.critic.load_state_dict(ck["critic"])
            self.critic_target = self.critic.make_ema_target()
            print("[SuperMimicStudent] Warm-started Privileged Critic from Teacher", flush=True)

    def _load_student_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"[SuperMimicStudent] Student checkpoint {path} not found, starting fresh.", flush=True)
            return
        print(f"[SuperMimicStudent] Resuming student from checkpoint: {path}", flush=True)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in ck:
            self.student.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            self.student_opt.load_state_dict(ck["optimizer_state_dict"])
        if "iter" in ck:
            self.current_learning_iteration = int(ck["iter"])
            print(f"[SuperMimicStudent] Resumed at iteration {self.current_learning_iteration}", flush=True)

    def _cosine_factor(self, progress: float, final_ratio: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return final_ratio + (1.0 - final_ratio) * c

    def _teacher_anchor_at(self, progress: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        return self.cfg.teacher_anchor_init + (self.cfg.teacher_anchor_final - self.cfg.teacher_anchor_init) * p

    def update_history_buffer(self, obs_58: torch.Tensor, dones: torch.Tensor) -> None:
        """Update live (num_envs, 2, 58) history buffer."""
        self.obs_hist[:, 0, :] = self.obs_hist[:, 1, :].clone()
        self.obs_hist[:, 1, :] = obs_58
        reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self.obs_hist[reset_ids] = 0.0

    def _collect_one_step(
        self,
        obs: torch.Tensor,
        use_teacher: bool = False,
        train_dagger: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, float]:
        snap = self.env.current_step_inputs()
        obs_58 = obs[:, :58]

        # Shift history with current step's obs_58 before acting
        self.obs_hist[:, 0, :] = self.obs_hist[:, 1, :].clone()
        self.obs_hist[:, 1, :] = obs_58

        with torch.no_grad():
            actions_teacher = self.teacher_actor(obs)

        actions_student = self.student(self.obs_hist)
        step_rmse = 0.0

        if train_dagger:
            step_loss = torch.sqrt(nn.functional.mse_loss(actions_student, actions_teacher) + 1e-8)
            self.student_opt.zero_grad(set_to_none=False)
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_clip_norm)
            self.student_opt.step()
            step_rmse = float(step_loss.detach().item())

        if use_teacher:
            actions_exec = actions_teacher
        else:
            actions_exec = actions_student.detach()

        with torch.no_grad():
            next_obs, rewards, dones, extras = self.env.step(actions_exec)

            # Reset history on envs that completed
            reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                self.obs_hist[reset_ids] = 0.0

            self.buffer.add_step(
                s=snap["s"],
                a_exec=actions_exec,
                prev_action=snap["prev_action"],
                arm_tgt=snap["arm_tgt"],
                s_next=self.env.last_pre_reset_state,
                feet_contact=snap["feet_contact"],
                next_feet_contact=self.env.last_pre_reset_feet_contact,
                contact_time=snap["contact_time"],
                air_time=snap["air_time"],
                impact_rew=self.env.last_impact_reward,
                cmd=snap["cmd"],
                heading_target=snap["heading_target"],
                is_heading=snap["is_heading"],
                is_standing=snap["is_standing"],
                ep_id=snap["ep_id"],
                amass_step=snap["amass_step"],
                pushed=self.env.last_pushed,
            )
            self.total_env_steps += self.env.num_envs
        return next_obs, rewards, dones, extras, step_rmse

    def _fit_critic(
        self,
        X_all: torch.Tensor,
        returns: torch.Tensor,
        survivals: torch.Tensor,
    ) -> float:
        # X_all is (H+1, B, D), returns is (H+1, B), survivals is (H, B)
        H, B = survivals.shape[:2]
        X_flat = X_all[:H].reshape(H * B, -1)
        R_flat = returns[:H].reshape(H * B)
        W_flat = survivals[:H].reshape(H * B)


        self.critic.norm.update(X_all)
        self.critic_target.norm.mean.copy_(self.critic.norm.mean)
        self.critic_target.norm.var.copy_(self.critic.norm.var)

        pred = self.critic(X_flat)
        sq_err = (pred - R_flat) ** 2
        c_loss = (sq_err * W_flat).sum() / W_flat.sum().clamp_min(1.0)

        self.critic_opt.zero_grad(set_to_none=False)
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(), 1.0)
        self.critic_opt.step()
        self.critic.update_ema_target(self.critic_target, 0.02)
        return float(c_loss.detach().item())

    def update_student_bptt(
        self,
        buffer: G1ReplayBuffer,
        xi_per_env: torch.Tensor,
        teacher_anchor_w: float,
        sym_w: float,
    ) -> dict[str, float]:
        """H-step Analytic BPTT on StudentMLP in G1WorldModel."""
        cfg = self.cfg
        H = cfg.horizon
        start = buffer.sample_starts(cfg.imag_batch, H, xi_per_env)

        # Relabel coupled 7D commands on a fraction of batch (D3)
        if cfg.hindsight_cmd_frac > 0.0:
            B = start.command.shape[0]
            n_rel = int(round(cfg.hindsight_cmd_frac * B))
            if n_rel > 0:
                dev = start.command.device
                idx = torch.randperm(B, device=dev)[:n_rel]
                start.command[idx, 0] = torch.rand(n_rel, device=dev) * 1.6 - 0.6
                start.command[idx, 1] = torch.rand(n_rel, device=dev) * 1.0 - 0.5
                start.command[idx, 2] = torch.rand(n_rel, device=dev) * 2.0 - 1.0
                start.command[idx, 3] = torch.rand(n_rel, device=dev) * 0.32 + 0.42
                start.command[idx, 4] = torch.rand(n_rel, device=dev) * 1.0 - 0.5
                start.command[idx, 5] = torch.rand(n_rel, device=dev) * 1.4 - 0.3
                start.command[idx, 6] = torch.rand(n_rel, device=dev) * 1.8 - 0.9
                start.is_heading_env[idx] = 0.0
                start.is_standing_env[idx] = 0.0

        if cfg.use_mirror_aug and self.symmetry is not None and sym_w > 0.0:
            self.symmetry.augment_imag_start_half_(start, min_asym_radius=cfg.sym_tau_high)

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
        tea_anchor_loss = s_tau.new_zeros(())
        sym_pen = s_tau.new_zeros(())

        critic_xs: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []
        dones_list: list[torch.Tensor] = []
        survivals_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []

        # Construct initial observation o_0 and seed history [o_0, o_0] for step 0
        obs_tau = self.diff_rewards.build_actor_obs(
            s=s_tau,
            command=cmd_tau,
            prev_action=prev_act,
            foot_contact_prob=p_foot_tau,
            add_noise=False,
            ste_contacts=True,
        )
        obs_58_prev = obs_tau[:, :58].detach()  # frame t-1

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

            # Build 116D history: concat(obs_{tau-1}^{0:58}, obs_{tau}^{0:58})
            obs_58_curr = obs_tau[:, :58]
            h_tau = torch.cat([obs_58_prev, obs_58_curr], dim=-1)  # (B, 116)

            # Student predicts action from proprioceptive history
            a_tau = self.student(h_tau)

            # Teacher action anchor (supervised action cloning on imagined states)
            if teacher_anchor_w > 0.0:
                with torch.no_grad():
                    a_tea = self.teacher_actor(obs_tau)
                tea_anchor_loss = tea_anchor_loss + ((a_tau - a_tea) ** 2).mean() / H

            # 116D Sagittal mirror symmetry loss
            if sym_w > 0.0 and self.symmetry is not None:
                h_mir = self.symmetry.mirror_obs_history_116(h_tau.detach())
                a_mir = self.student(h_mir)
                a_mir_sym = self.symmetry.mirror_act(a_mir)
                gate = self.symmetry.obs_asymmetry_gate(obs_tau.detach(), tau_low=cfg.sym_tau_low, tau_high=cfg.sym_tau_high)
                step_sym_mse = ((a_tau - a_mir_sym) ** 2).mean(dim=-1)
                sym_pen = sym_pen + (gate * step_sym_mse).mean() / H

            # World Model transition
            s_next, feat_next, c_logits_next, r_imp = self.wm.step_with_features(
                s=s_tau,
                policy_action=a_tau,
                arm_target=start.arm_future[tau],
                xi_norm=xi_norm,
                feat_s=feat_tau,
            )
            p_foot_next = torch.sigmoid(c_logits_next)
            dt = self.diff_rewards.dt
            ct_next = p_foot_next * (ct_tau + dt)
            at_next = (1.0 - p_foot_next) * (at_tau + dt)

            cmd_next = self.diff_rewards.update_command_in_imagination(
                s_next,
                start.command,
                start.heading_target,
                start.is_heading_env,
                start.is_standing_env,
            )

            r_next, d_next, _ = self.diff_rewards(
                s_prev=s_tau,
                s_next=s_next,
                action=a_tau,
                prev_action=prev_act,
                arm_target=start.arm_future[tau],
                command=cmd_next,
                is_standing_env=start.is_standing_env,
                xi=start.xi,
                foot_contact_prob=p_foot_next,
                contact_time_next=ct_next,
                air_time_next=at_next,
                r_impact_pred=r_imp,
                return_diag=False,
            )

            J = J + (cfg.gamma**tau) * survival * r_next

            rewards_list.append(r_next.detach())
            dones_list.append(d_next.detach())
            survivals_list.append(survival.detach())
            actions_list.append(a_tau.detach())

            survival = survival * (1.0 - d_next)
            obs_58_prev = obs_58_curr.detach()
            s_tau = s_next
            feat_tau = feat_next
            prev_act = a_tau
            p_foot_tau = p_foot_next
            ct_tau = ct_next
            at_tau = at_next
            cmd_tau = cmd_next

            # Build next observation
            obs_tau = self.diff_rewards.build_actor_obs(
                s=s_tau,
                command=cmd_tau,
                prev_action=prev_act,
                foot_contact_prob=p_foot_tau,
                add_noise=False,
                ste_contacts=True,
            )

        # Terminal step H bootstrap
        x_H = PrivilegedCriticMLP.build_features(
            feat_tau,
            cmd_tau,
            xi_norm,
            p_foot_tau,
            ct_tau,
            at_tau,
        )
        v_H = self.critic(x_H)
        J = J + (cfg.gamma**H) * survival * v_H

        # Compute GAE returns for Critic update
        X_all = torch.stack(critic_xs + [x_H.detach()], dim=0)  # (H+1, B, D)
        rewards = torch.stack(rewards_list, dim=0)
        dones = torch.stack(dones_list, dim=0)
        survivals = torch.stack(survivals_list, dim=0)

        with torch.no_grad():
            V_tgt = self.critic_target(X_all.reshape((H + 1) * B, -1)).reshape(H + 1, B)
            returns = torch.zeros_like(V_tgt)
            returns[H] = V_tgt[H]
            for t_step in range(H - 1, -1, -1):
                boot = (1.0 - cfg.gae_lambda) * V_tgt[t_step + 1] + cfg.gae_lambda * returns[t_step + 1]
                returns[t_step] = rewards[t_step] + cfg.gamma * (1.0 - dones[t_step]) * boot

        # Joint Student loss
        actor_rl_loss = -J.mean() / H
        total_student_loss = actor_rl_loss + teacher_anchor_w * tea_anchor_loss + sym_w * sym_pen

        self.student_opt.zero_grad(set_to_none=False)
        total_student_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip_norm)
        self.student_opt.step()

        # Update critic
        c_loss = self._fit_critic(X_all, returns, survivals)

        return {
            "student/loss": float(total_student_loss.detach().item()),
            "student/rl_loss": float(actor_rl_loss.detach().item()),
            "student/tea_anchor_loss": float(tea_anchor_loss.detach().item()),
            "student/sym_mse": float(sym_pen.detach().item()),
            "student/grad_norm": float(gn.detach().item()),
            "critic/loss": c_loss,
            "imag/undisc_return": float((survivals * rewards).sum(dim=0).mean().item()),
            "imag/survival_H": float(survival.mean().item()),
        }

    def learn(self, num_learning_iterations: int) -> None:
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)

        tot_it = int(num_learning_iterations)
        bc_warmup = self.cfg.bc_warmup_iters
        dagger_warmup = bc_warmup + self.cfg.dagger_warmup_iters

        rewbuffer: deque[float] = deque(maxlen=100)
        lenbuffer: deque[float] = deque(maxlen=100)
        fallbuffer: deque[float] = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)

        ep_infos: list[dict] = []
        upd_agg: dict[str, float] = {}
        upd_count: int = 0
        t_block_start = time.time()
        steps_in_block = 0

        start_it = self.current_learning_iteration
        for it in range(start_it, tot_it):
            t_iter_start = time.time()
            progress = it / max(tot_it, 1)

            # Cosine LR decay
            lr_mult = self._cosine_factor(progress, self.cfg.lr_final_ratio)
            for g in self.student_opt.param_groups:
                g["lr"] = self.cfg.student_lr * lr_mult

            teacher_anchor_w = self._teacher_anchor_at(progress)
            sym_w = self.cfg.symmetry_weight

            use_teacher_in_sim = it < bc_warmup
            step_rmse_list: list[float] = []

            for _ in range(self.cfg.num_steps_per_env):
                obs, rewards, dones, extras, step_rmse = self._collect_one_step(
                    obs,
                    use_teacher=use_teacher_in_sim,
                    train_dagger=True,
                )
                step_rmse_list.append(step_rmse)
                steps_in_block += self.env.num_envs

                if "log" in extras and extras["log"]:
                    ep_infos.append(extras["log"])

                cur_reward_sum += rewards
                cur_episode_length += 1.0
                done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    rewbuffer.extend(cur_reward_sum[done_ids].cpu().tolist())
                    lenbuffer.extend(cur_episode_length[done_ids].cpu().tolist())
                    fallbuffer.extend(self.env.last_terminated[done_ids].cpu().tolist())
                    cur_reward_sum[done_ids] = 0.0
                    cur_episode_length[done_ids] = 0.0

                # World-Model & BPTT Updates start after initial BC warmup
                if it >= bc_warmup:
                    self.update_credit += self.cfg.utd
                    while self.update_credit >= 1.0:
                        self.update_credit -= 1.0
                        wm_stats = self.wm_trainer.update(self.buffer, self.env.xi)
                        stu_stats = self.update_student_bptt(
                            self.buffer,
                            self.env.xi,
                            teacher_anchor_w=teacher_anchor_w,
                            sym_w=sym_w,
                        )
                        self.total_updates += 1
                        upd_count += 1
                        for k, v in wm_stats.items():
                            upd_agg[k] = upd_agg.get(k, 0.0) + v
                        for k, v in stu_stats.items():
                            upd_agg[k] = upd_agg.get(k, 0.0) + v

            self.wall_time_s += time.time() - t_iter_start
            self.current_learning_iteration = it + 1

            if (it + 1) % self.cfg.log_interval == 0 or (it + 1) == tot_it:
                dt_block = max(time.time() - t_block_start, 1.0e-6)
                fps = steps_in_block / dt_block
                mean_rew = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0.0
                mean_len = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0
                mean_fall = statistics.mean(fallbuffer) if len(fallbuffer) > 0 else 0.0
                mean_rmse = statistics.mean(step_rmse_list) if len(step_rmse_list) > 0 else 0.0

                avg_upd = {k: v / max(upd_count, 1) for k, v in upd_agg.items()}
                mode_str = "BC_WARMUP" if it < bc_warmup else ("DAGGER_WM" if it < dagger_warmup else "WM_BPTT_RL")

                print(
                    f"[Student] it {it + 1}/{tot_it} ({mode_str}) | steps {self.total_env_steps / 1e6:.2f}M | "
                    f"fps {fps:,.0f} | rew {mean_rew:.2f} | ep_len {mean_len:.1f} | fall {mean_fall:.2f} | "
                    f"step_rmse {mean_rmse:.4f} | wm_nse {avg_upd.get('wm/nse_1step', 0.0):.4f} | "
                    f"stu_loss {avg_upd.get('student/loss', 0.0):.4f} tea_loss {avg_upd.get('student/tea_anchor_loss', 0.0):.4f} "
                    f"sym_mse {avg_upd.get('student/sym_mse', 0.0):.4f} survH {avg_upd.get('imag/survival_H', 0.0):.3f}",
                    flush=True,
                )

                if self.writer is not None:
                    self.writer.add_scalar("Train/mean_reward", mean_rew, it + 1)
                    self.writer.add_scalar("Train/mean_episode_length", mean_len, it + 1)
                    self.writer.add_scalar("Train/fall_rate", mean_fall, it + 1)
                    self.writer.add_scalar("Train/step_rmse", mean_rmse, it + 1)
                    self.writer.add_scalar("Perf/fps", fps, it + 1)
                    for k, v in avg_upd.items():
                        self.writer.add_scalar(k, v, it + 1)

                rec = {
                    "iteration": it + 1,
                    "total_env_steps": self.total_env_steps,
                    "fps": fps,
                    "mean_reward": mean_rew,
                    "mean_episode_length": mean_len,
                    "fall_rate": mean_fall,
                    "step_rmse": mean_rmse,
                    **avg_upd,
                }
                self.jsonl_file.write(json.dumps(rec) + "\n")
                self.jsonl_file.flush()

                ep_infos.clear()
                upd_agg.clear()
                upd_count = 0
                t_block_start = time.time()
                steps_in_block = 0

            # Periodic saving & TorchScript export
            if (self.cfg.save_interval > 0 and (it + 1) % self.cfg.save_interval == 0) or (it + 1) == tot_it:
                self.save(it + 1, mean_reward=statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.jsonl_file.close()

    def save(self, iteration: int, mean_reward: float | None = None) -> None:
        """Save checkpoint and export TorchScript student_policy_jit.pt."""
        ckpt_path = os.path.join(self.log_dir, f"student_iter_{iteration}.pt")
        payload = {
            "iter": iteration,
            "mean_reward": mean_reward,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.student_opt.state_dict(),
            "config": asdict(self.cfg),
        }
        torch.save(payload, ckpt_path)

        # Export standard TorchScript JIT into exported/
        exp_dir = os.path.join(self.log_dir, "exported")
        os.makedirs(exp_dir, exist_ok=True)
        jit_path = os.path.join(exp_dir, "student_policy_jit.pt")
        export_student_jit(self.student, jit_path)

        last_pt = os.path.join(self.log_dir, "student_last.pt")
        torch.save(payload, last_pt)

        if mean_reward is not None and math.isfinite(mean_reward) and mean_reward >= self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            best_jit = os.path.join(self.log_dir, "student_policy_best_jit.pt")
            export_student_jit(self.student, best_jit)
            best_pt = os.path.join(self.log_dir, "model_best.pt")
            torch.save(payload, best_pt)
            print(f"[SuperMimicStudent] New best Student model exported to {best_jit} (rew={mean_reward:.2f})", flush=True)

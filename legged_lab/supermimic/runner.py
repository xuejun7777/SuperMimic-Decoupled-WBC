"""End-to-End Reference-Free SuperMimic Runner for Unitree G1 Decoupled WBC.

Aligns 1:1 with RSL-RL's `OnPolicyRunner` outer iteration definition:
  1 learning iteration = `num_steps_per_env` (24) simulator steps across all `num_envs`
so that:
  - Total environment transitions (`24 * max_iterations * num_envs`) match PPO bit-for-bit.
  - `BaseEnv.step()`'s internal command curriculum advances at the exact same iteration milestones.
  - TensorBoard metrics (`Train/mean_reward`, `Train/mean_episode_length`, `Episode/*`) and
    exported RSL-RL checkpoints (`model_<iter>.pt`) plug directly into `play_teacher.py`,
    `train_student.py`, and `eval_compare.py`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import os
import statistics
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from legged_lab.supermimic.diff_rewards import DifferentiableG1Rewards
from legged_lab.supermimic.env_adapter import SuperMimicG1Env
from legged_lab.supermimic.imagination import (
    PolicyCriticTrainer,
    PolicyTrainerCfg,
    WMTrainerCfg,
    WorldModelTrainer,
)
from legged_lab.supermimic.nets import ActorCfg, CriticCfg, PrivilegedCriticMLP, TeacherActorMLP
from legged_lab.supermimic.replay_buffer import G1ReplayBuffer
from legged_lab.supermimic.symmetry import G1SagittalSymmetry
from legged_lab.supermimic.usd_fk import UsdArticulationFK
from legged_lab.supermimic.world_model import G1WorldModel, WorldModelCfg


@dataclass
class SuperMimicRunnerCfg:
    # Outer loop (1 iteration = num_steps_per_env simulator steps, matching RSL-RL PPO)
    num_steps_per_env: int = 24
    utd: float = 0.4  # WM + Policy updates per single simulator step (0.4 * 24 = 9.6 updates / iter)
    replay_capacity: int = 1000
    warmup_iters: int = 20  # 20 * 24 = 480 env steps counted INSIDE total max_iterations budget
    wm_pretrain_updates: int = 40
    critic_pretrain_updates: int = 10

    # Exploration in real simulator
    action_noise_init: float = 0.30
    action_noise_final: float = 0.15
    det_frac: float = 0.20

    # Cosine learning-rate schedule
    lr_final_ratio: float = 0.15

    # Logging & checkpointing
    log_interval: int = 10
    save_interval: int = 500
    keep_only_best_and_last: bool = True
    use_tensorboard: bool = True

    # Component configs
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)


class SuperMimicRunner:
    def __init__(
        self,
        env: SuperMimicG1Env,
        cfg: SuperMimicRunnerCfg,
        log_dir: str,
        device: str | torch.device = "cuda:0",
    ):
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = torch.device(device)
        os.makedirs(self.log_dir, exist_ok=True)

        # Build USD-exact differentiable FK, sagittal symmetry operator, & 31-term differentiable rewards
        self.fk = UsdArticulationFK(env.robot, device=self.device)
        self.symmetry = G1SagittalSymmetry(env, self.fk, device=self.device)
        self.diff_rewards = DifferentiableG1Rewards(env, self.fk).to(self.device)

        default_arm_pos = env.custom_default_joint_pos[0, env.num_actions : env.num_actions + 14].clone()
        self.wm = G1WorldModel(
            cfg=cfg.world_model,
            action_scale=self.diff_rewards.action_scale,
            default_arm_pos=default_arm_pos,
            clip_actions=env.clip_actions,
            device=self.device,
        )
        self.actor = TeacherActorMLP(cfg.actor).to(self.device)
        self.critic = PrivilegedCriticMLP(cfg.critic).to(self.device)

        self.wm_trainer = WorldModelTrainer(self.wm, cfg.wm_trainer, symmetry=self.symmetry)
        self.policy_trainer = PolicyCriticTrainer(
            wm=self.wm,
            actor=self.actor,
            critic=self.critic,
            diff_rewards=self.diff_rewards,
            cfg=cfg.policy_trainer,
            symmetry=self.symmetry,
        )

        self.buffer = G1ReplayBuffer(
            num_envs=env.num_envs,
            capacity=cfg.replay_capacity,
            act_dim=env.num_actions,
            arm_dim=14,
            amass_seq_len=env.sequence_length,
            device=self.device,
        )
        env.attach_replay_buffer(self.buffer)

        self.current_learning_iteration: int = 0
        self.total_env_steps: int = 0
        self.total_updates: int = 0
        self.update_credit: float = 0.0
        self.wall_time_s: float = 0.0
        self.best_mean_reward: float = -float("inf")
        self._last_saved_rsl_ckpt: str | None = None

        self.writer: SummaryWriter | None = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if cfg.use_tensorboard else None
        self.jsonl_file = open(os.path.join(self.log_dir, "metrics.jsonl"), "a", encoding="utf-8")

    def _cosine_factor(self, progress: float, final_ratio: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return final_ratio + (1.0 - final_ratio) * c

    def _action_noise_at(self, progress: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        return self.cfg.action_noise_init + (self.cfg.action_noise_final - self.cfg.action_noise_init) * p

    def _symmetry_weight_at(self, global_progress: float) -> float:
        p_cfg = self.cfg.policy_trainer
        target_w = float(p_cfg.symmetry_weight)
        if target_w <= 0.0:
            return 0.0
        p = min(max(global_progress, 0.0), 1.0)
        w0 = float(p_cfg.sym_warmup_ratio)
        ramp = float(p_cfg.sym_ramp_ratio)
        if p < w0:
            return 0.0
        if ramp <= 0.0 or p >= w0 + ramp:
            return target_w
        u = (p - w0) / ramp
        return target_w * 0.5 * (1.0 - math.cos(math.pi * u))

    def save(
        self,
        iteration: int,
        mean_reward: float | None = None,
        include_optimizers: bool = True,
    ) -> str:
        """Save RSL-RL-compatible `model_<iter>.pt` (pruning older intermediate checkpoints) & compact `supermimic_last.pt`."""
        noise_std = self._action_noise_at(0.5)
        rsl_ckpt_path = os.path.join(self.log_dir, f"model_{iteration}.pt")
        rsl_payload = {
            "model_state_dict": self.actor.rsl_rl_state_dict(noise_std=noise_std),
            "iter": int(iteration),
            "infos": {"mean_reward": mean_reward},
        }
        if include_optimizers:
            rsl_payload["optimizer_state_dict"] = self.policy_trainer.actor_opt.state_dict()

        torch.save(rsl_payload, rsl_ckpt_path + ".tmp")
        os.replace(rsl_ckpt_path + ".tmp", rsl_ckpt_path)

        if self.cfg.keep_only_best_and_last:
            if self._last_saved_rsl_ckpt is not None and self._last_saved_rsl_ckpt != rsl_ckpt_path:
                if os.path.isfile(self._last_saved_rsl_ckpt):
                    os.remove(self._last_saved_rsl_ckpt)
            self._last_saved_rsl_ckpt = rsl_ckpt_path

        if mean_reward is not None and math.isfinite(mean_reward) and mean_reward >= self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            best_path = os.path.join(self.log_dir, "model_best.pt")
            torch.save(rsl_payload, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)

        sm_payload: dict = {
            "cfg": asdict(self.cfg),
            "iter": int(iteration),
            "total_env_steps": int(self.total_env_steps),
            "total_updates": int(self.total_updates),
            "wall_time_s": float(self.wall_time_s),
            "best_mean_reward": float(self.best_mean_reward),
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.policy_trainer.critic_target.state_dict(),
        }
        if include_optimizers:
            sm_payload["wm_opt"] = self.wm_trainer.opt.state_dict()
            sm_payload["actor_opt"] = self.policy_trainer.actor_opt.state_dict()
            sm_payload["critic_opt"] = self.policy_trainer.critic_opt.state_dict()

        sm_last = os.path.join(self.log_dir, "supermimic_last.pt")
        torch.save(sm_payload, sm_last + ".tmp")
        os.replace(sm_last + ".tmp", sm_last)

        return rsl_ckpt_path

    def load(self, ckpt_path: str, load_optimizer: bool = True) -> int:
        """Load from either `supermimic_*.pt` or an RSL-RL `model_*.pt` in the same folder."""
        sm_candidate = os.path.join(os.path.dirname(ckpt_path), "supermimic_last.pt")
        target = sm_candidate if (os.path.basename(ckpt_path).startswith("model_") and os.path.isfile(sm_candidate)) else ckpt_path
        ck = torch.load(target, map_location=self.device, weights_only=False)

        if "wm" in ck and "actor" in ck:
            self.wm.load_state_dict(ck["wm"])
            self.actor.load_state_dict(ck["actor"])
            self.critic.load_state_dict(ck["critic"])
            if "critic_target" in ck:
                self.policy_trainer.critic_target.load_state_dict(ck["critic_target"])
            if load_optimizer and "wm_opt" in ck and "actor_opt" in ck and "critic_opt" in ck:
                self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
                self.policy_trainer.actor_opt.load_state_dict(ck["actor_opt"])
                self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
            self.current_learning_iteration = int(ck.get("iter", 0))
            self.total_env_steps = int(ck.get("total_env_steps", 0))
            self.total_updates = int(ck.get("total_updates", 0))
            self.wall_time_s = float(ck.get("wall_time_s", 0.0))
            self.best_mean_reward = float(ck.get("best_mean_reward", -float("inf")))
        elif "model_state_dict" in ck:
            actor_sd = {k[len("actor.") :]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
            self.actor.actor.load_state_dict(actor_sd)
            self.current_learning_iteration = int(ck.get("iter", 0))
        else:
            raise ValueError(f"Unrecognized checkpoint format at {target}")

        return self.current_learning_iteration

    @torch.no_grad()
    def _collect_one_step(
        self,
        obs: torch.Tensor,
        noise_std: float,
        random_zero_centered: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all parallel environments once and push transition to `G1ReplayBuffer`."""
        snap = self.env.current_step_inputs()

        if random_zero_centered:
            actions = noise_std * torch.randn(self.env.num_envs, self.env.num_actions, device=self.device)
        else:
            mean_act = self.actor(obs)
            if noise_std > 0.0:
                noise = noise_std * torch.randn_like(mean_act)
                n_det = int(round(self.cfg.det_frac * self.env.num_envs))
                if n_det > 0:
                    noise[:n_det].zero_()
                actions = mean_act + noise
            else:
                actions = mean_act

        next_obs, rewards, dones, extras = self.env.step(actions)

        self.buffer.add_step(
            s=snap["s"],
            a_exec=actions,
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
        return next_obs, rewards, dones, extras

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)

        start_it = self.current_learning_iteration
        tot_it = start_it + int(num_learning_iterations)
        if num_learning_iterations <= 0:
            return

        # Clamp warmup iterations gracefully for short sanity checks while strictly counting inside tot_it
        eff_warmup_iters = min(self.cfg.warmup_iters, max(1, num_learning_iterations // 3))
        min_iters_for_window = math.ceil((self.cfg.wm_trainer.window_k + 2) / self.cfg.num_steps_per_env)
        eff_warmup_iters = max(eff_warmup_iters, min_iters_for_window)
        eff_warmup_iters = min(eff_warmup_iters, num_learning_iterations)
        half_warmup = max(1, eff_warmup_iters // 2)
        needs_warmup = self.buffer.size < (eff_warmup_iters * self.cfg.num_steps_per_env)

        rewbuffer: deque[float] = deque(maxlen=100)
        lenbuffer: deque[float] = deque(maxlen=100)
        fallbuffer: deque[float] = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)

        ep_infos: list[dict] = []
        upd_agg: dict[str, float] = {}
        upd_count: int = 0
        precalibrated = not needs_warmup

        self.save(start_it)
        t_block_start = time.time()
        steps_in_block = 0

        for it in range(start_it, tot_it):
            t_iter_start = time.time()
            local_it = it - start_it
            in_warmup = needs_warmup and (local_it < eff_warmup_iters)

            progress = (it - start_it) / max(tot_it - start_it, 1)
            global_progress = it / max(tot_it, 1)
            lr_mult = self._cosine_factor(progress, self.cfg.lr_final_ratio)
            self.wm_trainer.set_lr(self.cfg.wm_trainer.lr * lr_mult)
            self.policy_trainer.set_lr(
                actor_lr=self.cfg.policy_trainer.actor_lr * lr_mult,
                critic_lr=self.cfg.policy_trainer.critic_lr * lr_mult,
            )
            cur_sym_weight = self._symmetry_weight_at(global_progress)
            self.policy_trainer.set_symmetry_weight(cur_sym_weight)
            noise_std = self._action_noise_at(progress)

            t_collect = 0.0
            t_learn = 0.0

            for _ in range(self.cfg.num_steps_per_env):
                tc0 = time.time()
                rand_perturb = in_warmup and (start_it == 0) and (local_it < half_warmup)
                obs, rewards, dones, extras = self._collect_one_step(
                    obs,
                    noise_std=noise_std,
                    random_zero_centered=rand_perturb,
                )
                t_collect += time.time() - tc0
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

                if not in_warmup:
                    tl0 = time.time()
                    self.update_credit += self.cfg.utd
                    while self.update_credit >= 1.0:
                        self.update_credit -= 1.0
                        wm_stats = self.wm_trainer.update(self.buffer, self.env.xi)
                        pi_stats = self.policy_trainer.update(self.buffer, self.env.xi)
                        self.total_updates += 1
                        upd_count += 1
                        for k_name, v_val in wm_stats.items():
                            upd_agg[k_name] = upd_agg.get(k_name, 0.0) + v_val
                        for k_name, v_val in pi_stats.items():
                            upd_agg[k_name] = upd_agg.get(k_name, 0.0) + v_val
                    t_learn += time.time() - tl0

            # Pre-calibrate World Model & Privileged Critic right at the end of warmup
            if in_warmup and (local_it == eff_warmup_iters - 1) and not precalibrated:
                tl0 = time.time()
                n_wm_pre = self.cfg.wm_pretrain_updates if start_it == 0 else min(10, self.cfg.wm_pretrain_updates)
                n_cr_pre = self.cfg.critic_pretrain_updates if start_it == 0 else min(5, self.cfg.critic_pretrain_updates)
                print(
                    f"[SuperMimic] Warmup complete ({self.buffer.size}x{self.env.num_envs} transitions). "
                    f"Calibrating World Model ({n_wm_pre} updates) & "
                    f"Critic ({n_cr_pre} updates)...",
                    flush=True,
                )
                wm_pre = {}
                for _ in range(n_wm_pre):
                    wm_pre = self.wm_trainer.update(self.buffer, self.env.xi)
                    self.total_updates += 1
                cr_pre = self.policy_trainer.pretrain_critic_only(
                    self.buffer, self.env.xi, num_updates=n_cr_pre
                )
                t_learn += time.time() - tl0
                precalibrated = True
                print(
                    f"[SuperMimic] Calibration done in {t_learn:.2f}s | "
                    f"wm/nse_1step={wm_pre.get('wm/nse_1step', 0.0):.4f} "
                    f"wm/nse_kstep={wm_pre.get('wm/nse_kstep', 0.0):.4f} "
                    f"wm/contact_acc={wm_pre.get('wm/contact_acc', 0.0):.3f} "
                    f"critic/loss={cr_pre.get('critic/loss', 0.0):.4f}",
                    flush=True,
                )


            self.wall_time_s += time.time() - t_iter_start
            self.current_learning_iteration = it + 1

            # Periodic logging
            if (it + 1) % self.cfg.log_interval == 0 or (it + 1) == tot_it:
                dt_block = max(time.time() - t_block_start, 1.0e-6)
                fps = steps_in_block / dt_block
                mean_rew = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else 0.0
                mean_len = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else 0.0
                mean_fall = statistics.mean(fallbuffer) if len(fallbuffer) > 0 else 0.0

                avg_upd = {k_name: v_val / max(upd_count, 1) for k_name, v_val in upd_agg.items()}
                ep_means: dict[str, float] = {}
                if ep_infos:
                    keys = ep_infos[0].keys()
                    for k_ep in keys:
                        vals = [
                            float(inf[k_ep].mean().item()) if torch.is_tensor(inf[k_ep]) else float(inf[k_ep])
                            for inf in ep_infos
                            if k_ep in inf
                        ]
                        if vals:
                            ep_means[k_ep] = sum(vals) / len(vals)

                record = {
                    "iter": it + 1,
                    "env_steps": self.total_env_steps,
                    "updates": self.total_updates,
                    "wall_time_s": round(self.wall_time_s, 2),
                    "fps": round(fps, 1),
                    "noise_std": round(noise_std, 4),
                    "lr_actor": self.cfg.policy_trainer.actor_lr * lr_mult,
                    "Train/mean_reward": mean_rew,
                    "Train/mean_episode_length": mean_len,
                    "Train/fall_rate": mean_fall,
                    **avg_upd,
                    **ep_means,
                }
                self.jsonl_file.write(json.dumps(record) + "\n")
                self.jsonl_file.flush()

                if self.writer is not None:
                    self.writer.add_scalar("Train/mean_reward", mean_rew, it + 1)
                    self.writer.add_scalar("Train/mean_episode_length", mean_len, it + 1)
                    self.writer.add_scalar("Train/fall_rate", mean_fall, it + 1)
                    self.writer.add_scalar("Perf/total_fps", fps, it + 1)
                    self.writer.add_scalar("Policy/mean_noise_std", noise_std, it + 1)
                    for k_name, v_val in avg_upd.items():
                        self.writer.add_scalar(k_name, v_val, it + 1)
                    for k_ep, v_ep in ep_means.items():
                        tag = k_ep if "/" in k_ep else f"Episode/{k_ep}"
                        self.writer.add_scalar(tag, v_ep, it + 1)

                print(
                    f"[SuperMimic] it {it + 1}/{tot_it} | steps {self.total_env_steps / 1e6:.2f}M | "
                    f"upd {self.total_updates} | fps {fps:,.0f} | "
                    f"rew {mean_rew:.2f} | ep_len {mean_len:.1f} | fall {mean_fall:.2f} | "
                    f"wm1 {avg_upd.get('wm/nse_1step', 0.0):.4f} wm8 {avg_upd.get('wm/nse_kstep', 0.0):.4f} "
                    f"c_acc {avg_upd.get('wm/contact_acc', 0.0):.3f} | "
                    f"J {avg_upd.get('actor/objective_J', 0.0):.3f} survH {avg_upd.get('imag/survival_H', 0.0):.3f} "
                    f"gn {avg_upd.get('actor/grad_norm', 0.0):.3f} |a| {avg_upd.get('actor/action_abs_mean', 0.0):.2f} "
                    f"v_loss {avg_upd.get('critic/loss', 0.0):.4f}",
                    flush=True,
                )

                ep_infos.clear()
                upd_agg.clear()
                upd_count = 0
                t_block_start = time.time()
                steps_in_block = 0

            if self.cfg.save_interval > 0 and (it + 1) % self.cfg.save_interval == 0:
                cur_mean_rew = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None
                self.save(it + 1, mean_reward=cur_mean_rew, include_optimizers=True)

        final_mean_rew = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None
        self.save(tot_it, mean_reward=final_mean_rew, include_optimizers=False)
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.jsonl_file.close()

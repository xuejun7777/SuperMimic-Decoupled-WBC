# SuperMimic-Decoupled-WBC

### Reference-Free Differentiable World Model RL for Humanoid Loco-Manipulation Whole-Body Control

<p align="center">
  <img src="https://img.shields.io/badge/Isaac_Lab-2.2.0_(Isaac_Sim_5.x)-76B900?style=flat-square&logo=nvidia" alt="Isaac Lab 2.2">
  <img src="https://img.shields.io/badge/PyTorch-2.7_%7C_CUDA_Graph-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch 2.7">
  <img src="https://img.shields.io/badge/Robot-Unitree_G1_(29_DoF)-007EC6?style=flat-square" alt="Unitree G1">
  <img src="https://img.shields.io/badge/Single_RTX_4090-3.5_Hours_(295M_Steps)-2EA44F?style=flat-square" alt="Single GPU 3.5h">
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-blue?style=flat-square" alt="BSD-3-Clause">
</p>

<p align="center">
  <img src="docs/assets/supermimic_vs_ppo_teaser.gif" alt="Reference-Free SuperMimic (Phase-Gated Symmetry) vs Native RSL-RL PPO at Equal 295M Step Budget" width="100%">
</p>

> **Left**: **Reference-Free SuperMimic + Phase-Gated Mirror Symmetry (`3,000` iters / `295M` steps, `3.5h` on 1x RTX 4090)** &nbsp;|&nbsp; **Right**: **Native RSL-RL PPO (`3,000` iters / `295M` steps)**  
> *Both policies are trained completely from scratch without MoCap reference trajectories and evaluated in Isaac Sim PhysX under identical initial states, identical continuous 14-DoF AMASS bimanual arm-swing disturbances, and a 6-phase 7D loco-manipulation command schedule. Full 18s 720p@50fps video: [`docs/assets/supermimic_vs_ppo_demo.mp4`](docs/assets/supermimic_vs_ppo_demo.mp4).*

---

## Overview

Original **SuperMimic** (*Scaling Humanoid Whole-Body Control with One Differentiable World Model*) demonstrated the power of learned differentiable dynamics + analytic Backpropagation Through Time (BPTT) on **MoCap motion tracking**, where Reference State Initialization (RSI) and dense per-joint MoCap targets anchor exploration.

**SuperMimic-Decoupled-WBC** explores a fundamental question for general robotics RL:

> *Can Differentiable World Models + Analytic BPTT replace zero-order PPO **completely without MoCap reference trajectories (Reference-Free)** as a general-purpose training paradigm for **humanoid loco-manipulation foundation controllers**?*

Built cleanly as a **zero-intrusion modular extension** (`legged_lab/supermimic/`) on top of [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC) (*Humanoid Touch Dream*, CMU), this repository trains a decoupled lower-body + 3-DoF waist controller (`15` actuated joints) on **Unitree G1** to track arbitrary **7D locomotion and torso-pose commands** (`[vx, vy, wz, height, roll, pitch, yaw]`) while rejecting continuous exogenous **14-DoF bimanual upper-body manipulation disturbances**.

### Key Highlights

- **~100-240x Sample Efficiency Over Zero-Order PPO**: At `295M` environment transitions (`~3.5h` on a single RTX 4090), Reference-Free SuperMimic escapes PPO's standing-still local optimum (**`100.0%` 20s survival across all 8 evaluation suites**, **`113.23`** episode return vs `92.82` for equal-budget PPO) and matches the walking smoothness (`d_act = 1.032` vs `1.075`) of the upstream flagship PPO policy trained for **`240,000` iterations (`70.8B` transitions = `240x` more samples)**.
- **Phase-Gated Sagittal Mirror Symmetry (`4.6x` Straighter Open-Loop Walking)**: Solves the *Phase-Free Bipedal Standing Lock* (where naive pointwise equivariance `pi(M(o)) = M(pi(o))` freezes memoryless bipeds at symmetric double-support rest `o = M(o)`) via a proprioceptive fixed-point exclusion gate + 50/50 mirror-augmented world model, cutting 5s straight-walk lateral crab drift from `37.1-39.9 cm` down to **`8.1-11.3 cm`** (**`3.5-4.6x` lower than unsymmetrized SuperMimic and `4.1-5.1x` lower than `240k`-iter PPO**).
- **Sharp Multi-Joint Posture Synergy**: Exact USD differentiable kinematics (`< 3e-6 m` parity) routes analytic reward gradients across all 15 lower-body & waist joints simultaneously—achieving **`0.46 deg` (`0.008 rad`)** pitch MAE on `46 deg` forward bows (`torso_pitch_bow`) and **`0.74 deg` (`0.013 rad`)** on extreme `68.8 deg` (`1.20 rad`) bows past G1's `+-29.8 deg` waist mechanical hard stop (vs `23.9 deg` for `240k`-iter PPO).
- **23,100-24,350+ FPS Single-GPU Throughput**: Native side-stream `torch.cuda.CUDAGraph` capture unrolls `H=24` closed-loop imagination steps + 30-body differentiable USD FK + 31 differentiable reward terms + phase-gated symmetry loss + full analytic BPTT backward pass in `bfloat16`/`fp32`.
- **100% Drop-In RSL-RL & Physical G1 Deployment Compatibility**: Exports standard `180,367`-parameter RSL-RL checkpoints (`example/supermimic_g1_flat/model_3000.pt`, zero inference wrapper) compatible out of the box with `play_teacher.py`, `train_student.py` (BC -> DAgger history distillation), and Unitree SDK2 deployment.
- **Verifiable Zero Upstream Baseline Tampering**: Zero lines of upstream environment physics, PD gains, observations, Domain Randomization, or PPO baseline code were altered (`git diff upstream-main -- legged_lab/envs legged_lab/mdp legged_lab/assets legged_lab/scripts/train_teacher.py` is **0 bytes**).

---

## Training Curves, Benchmarks & Mechanistic Diagnostics

<p align="center">
  <img src="docs/assets/supermimic_vs_ppo_curves.png" alt="SuperMimic vs PPO Training Curves, 8-Suite Benchmark Returns, and 5s Straight-Walk Lateral Drift" width="100%">
</p>

<p align="center">
  <img src="docs/assets/supermimic_vs_ppo_keyframes.png" alt="6-Phase Whole-Body Loco-Manipulation Keyframes" width="100%">
</p>

### 1. Six-Way PhysX Evaluation (`512` Envs, Paired AMASS Arm Disturbances: Teachers & Proprioceptive Students)

| Policy | Stage / Input | Env Steps | `full_random` Surv ↑ | `full_random` Return ↑ | `full_random` `vxy_err` ↓ | `full_random` `h_mae` ↓ | `torso_pitch_bow` (`46°`) `pitch_mae` ↓ | `D4` `5s` Straight-Walk Lateral Drift (Neutral / AMASS) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PPO_3000` (Equal Budget) | Teacher (`60D`) | `0.295B` (1x) | `94.9%` | `92.82` | `0.271 m/s` | `8.73 cm` | `15.5°` (`0.271 rad`) | *frozen (`vx ≈ 0`)* |
| `SuperMimic_3000` *(No Sym)* | Teacher (`60D`) | `0.295B` (1x) | `99.6%` | `112.27` | `0.133 m/s` | `3.77 cm` | `1.20°` (`0.021 rad`) | `37.1 cm` / `39.9 cm` |
| **`SuperMimic_3000` *(Sym, Default)*** | **Teacher (`60D`)** | **`0.295B` (1x)** | **`100.0%`** | **`113.23`** | **`0.129 m/s`** | **`3.63 cm`** | **`0.46°` (`0.008 rad`)** | **`8.1 cm` / `11.3 cm`** |
| `CMU_PPO_240k` (Upstream Ceiling) | Teacher (`60D`) | `70.78B` (240x) | `100.0%` | `120.94` | `0.093 m/s` | `2.87 cm` | `0.63°` (`0.011 rad`) | `41.7 cm` / `46.2 cm` |
| **`SM_3000_Sym_Student` *(Ours, 4k steps)*** | **Student (`116D`, No Contact)** | **`+16.4M` (`5 min`)** | **`99.6%`** | **`108.98`** | **`0.154 m/s`** | **`4.21 cm`** | **`0.86°` (`0.015 rad`)** | **`24.8 cm` / `25.3 cm`** *(yaw `5.35°`)* |
| `CMU_Official_Student` *(Bundled Deploy)* | Student (`116D`, No Contact) | `+2.46B` (`600k` steps) | `100.0%` | `121.04` | `0.097 m/s` | `3.33 cm` | `0.57°` (`0.010 rad`) | `31.6 cm` / `34.0 cm` *(yaw `13.31°`)* |

> **Proprioceptive Student Highlights (`example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt`)**: Distilled in just `4,000` steps (`1,500` BC + `2,500` DAgger, `~5 min` on 1x RTX 4090, `RMSE = 0.0461`) without foot-contact sensors (`2 x 58 = 116D` history), `SM_3000_Sym_Student` retains **`96.3%`** of its Teacher's full-random return (`108.98`), achieves **`100.0%` survival** on 7/7 targeted locomotion/posture suites, outperforms `CMU_Official_Student` on `squat_deep` (`70.99` vs `69.75`) and `extreme_pitch_1p2` (`24.8°` vs `42.4°` pitch error), and maintains **`2.5x` lower open-loop heading drift** (`5.35°` vs `13.31°`) and **`22-26%` lower straight-walk lateral drift** (`24.8 / 25.3 cm` vs `31.6 / 34.0 cm`).

### 2. Why Does Reference-Free SuperMimic Beat PPO? (Mechanistic Summary)

Full diagnostic suite reproducible via `legged_lab/scripts/run_diagnostics.py` — see **[Technical & Mechanistic Research Report](docs/supermimic_research_report.md)** for complete tables:

1. **Millimeter/Sub-Degree Short-Horizon Digital Twin (`D1`)**: Over `H=24` open-loop steps (`0.48s`), `G1WorldModel` (with 50/50 sagittal mirror batch augmentation) maintains **`0.197 cm`** pelvis height MAE, **`0.222°`** 29-joint MAE, **`97.46%`** foot-contact accuracy, and **Pearson `r = 0.9554`** step-reward correlation against real PhysX.
2. **High Directional Gradient SNR Without Action-Dimension Curse (`D2`)**: Across `180,367` policy parameters (`N=256`, `sigma=0.20`), zero-order score-function PG suffers severe variance (`cos = -0.1226` at `1x` budget; `0.5102` at **`16x`** rollout budget), whereas single-draw Analytic BPTT achieves **`cos = 0.7721`** and routes **`68.1%`** of extreme forward-bow action sensitivity directly into coordinated lower-body counter-balancing joints (`hip_pitch + knee + ankle_pitch`).
3. **Continuous Zero-Cost Hindsight Command Relabeling (`D3`)**: Because `G1WorldModel` is strictly **command-free**, relabeling old physical states with unseen counterfactual 7D commands correlates with branched real PhysX returns at **Pearson `r = 0.9917`** (`per-step MAE = 0.0101`).
4. **Eliminating Single-Seed Gait Drift Without Phase Clocks (`D4`)**: Phase-asymmetry-gated sagittal equivariance reduces 5s straight-walk (`vx = 0.45 m/s`) open-loop lateral drift to **`8.1 cm`** (neutral arms) / **`11.3 cm`** (AMASS bimanual disturbances) and closed-loop heading error to **`1.88°`** (`3.4x` tighter than `CMU_PPO_240k`'s `6.47°`).

---

## Quick Start

### 1. Installation

Follow the standard Isaac Lab 2.2 environment setup (Ubuntu 20.04/22.04+, Python 3.10/3.11, CUDA 12.x):

```bash
bash scripts/setup_conda.sh        # Isaac Sim 5.0/5.1 + Isaac Lab 2.2.0 (python 3.11)
conda activate isaac_wbc
```

*(Headless Docker setup is also supported via `bash scripts/setup_docker.sh`; see [Upstream Guide](docs/UPSTREAM_README.md).)*

### 2. Play the Bundled Reference-Free SuperMimic Policy Out of the Box

A pre-trained `3,000`-iteration Reference-Free SuperMimic checkpoint (`1.6 MB`) is bundled at [`example/supermimic_g1_flat/model_3000.pt`](example/supermimic_g1_flat/model_3000.pt):

```bash
# Interactive simulation window with keyboard 7D velocity + torso pose control
bash scripts/play_supermimic.sh

# Compare interactively against the upstream 240k-iter PPO teacher
bash scripts/play_teacher.sh
```

> **Keyboard Controls**: `W`/`S` (`vx`), `A`/`D` (`vy`), `Q`/`E` (`wz`), `H`/`J` (pelvis height), `Z`/`X` (torso roll), `C`/`V` (torso pitch), `B`/`N` (torso yaw), `R` (reset).

### 3. Train Reference-Free SuperMimic From Scratch (Single GPU)

```bash
# Train Reference-Free SuperMimic (4096 envs x 3000 iters = 295M steps, ~3.5h on 1x RTX 4090)
bash scripts/train_supermimic.sh

# Equal-budget Native RSL-RL PPO baseline (exact same 4096 envs, 3000 iters & pose curriculum)
python legged_lab/scripts/train_teacher.py \
    --task=g1_flat --run_name=ppo_baseline_a2a --logger=tensorboard \
    --num_envs=4096 --max_iterations=3000 --save_interval=500 --seed=42 --headless \
    --set commands.curriculum.body_height.start_iter=600 \
    --set commands.curriculum.body_height.end_iter=2000 \
    --set commands.curriculum.body_roll.start_iter=600 \
    --set commands.curriculum.body_roll.end_iter=2000 \
    --set commands.curriculum.body_pitch.start_iter=600 \
    --set commands.curriculum.body_pitch.end_iter=2000 \
    --set commands.curriculum.body_yaw.start_iter=600 \
    --set commands.curriculum.body_yaw.end_iter=2000
```

### 4. Standardized Evaluation, Parity Tests & Mechanistic Diagnostics

```bash
# 1. Run 8-suite deterministic PhysX benchmark (512 envs, paired AMASS disturbances)
bash scripts/eval_supermimic.sh

# 2. Verify USD Differentiable FK (< 3e-6 m) & 31-term Reward Parity (< 1.4e-7) + D1-D4 Diagnostics
SM_DIR=logs/g1_flat/<your_supermimic_run> bash scripts/run_diagnostics.sh

# 3. Render 1280x720@50fps side-by-side comparison video
python legged_lab/scripts/record_demo_video.py \
    --task=g1_flat \
    --sm_ckpt=example/supermimic_g1_flat/model_3000.pt \
    --ppo_ckpt=example/model_000.pt \
    --out_mp4=logs/g1_flat/demo_comparison.mp4
```

### 5. Student Distillation & Physical Unitree G1 Deployment

Because `SuperMimicRunner` exports standard RSL-RL checkpoints, distilling the 60D Teacher into a purely proprioceptive history-based Student JIT policy and deploying to a real Unitree G1 requires zero changes to the upstream pipeline:

```bash
# Distill SuperMimic teacher -> proprioceptive history student (BC -> DAgger)
HTD_TEACHER_RUN=supermimic_example HTD_TEACHER_CKPT=model_3000.pt bash scripts/train_student.sh
```

See [`docs/deployment.md`](docs/deployment.md) for physical Unitree G1 SDK2 deployment instructions.

---

## Repository Structure

```text
├── legged_lab/
│   ├── supermimic/                  # Modular Reference-Free SuperMimic implementation
│   │   ├── rotations.py             # CUDAGraph-safe wxyz SO(3) & smooth Euler operators
│   │   ├── state_spec.py            # 71D Markov state, SE(2) canonicalization & manifold integration
│   │   ├── usd_fk.py                # Zero-clone differentiable 30-body / 29-hinge USD kinematics
│   │   ├── symmetry.py              # Sagittal mirror equivariance & phase-asymmetry standing gate
│   │   ├── diff_rewards.py          # Differentiable 31-term G1 WBC rewards & 60D actor obs
│   │   ├── world_model.py           # bfloat16 2048x3 command-free dynamics + contact + impact heads
│   │   ├── nets.py                  # 180,367-param TeacherActorMLP & 88D PrivilegedCriticMLP
│   │   ├── replay_buffer.py         # [1000, 4096] GPU ring buffer + 8192-slot fp16 AMASS clip bank
│   │   ├── env_adapter.py           # Read-only BaseEnv subclass capturing pre-reset fall transitions
│   │   ├── imagination.py           # WorldModelTrainer & side-stream CUDAGraph PolicyCriticTrainer
│   │   ├── runner.py                # End-to-end runner with auto checkpoint pruning & ONNX export
│   │   └── test_parity.py           # Bitwise/eps parity & sagittal equivariance verification
│   ├── scripts/
│   │   ├── train_supermimic.py      # SuperMimic CLI entrypoint (mirrors train_teacher.py flags)
│   │   ├── eval_compare.py          # Deterministic 8-suite multi-checkpoint PhysX evaluator
│   │   ├── run_diagnostics.py       # Mechanistic diagnostic experiments D1-D4
│   │   └── record_demo_video.py     # Headless 720p@50fps side-by-side PhysX video renderer
│   └── envs/, mdp/, assets/         # 100% untouched upstream IsaacLab-Decoupled-WBC environment
├── example/
│   ├── supermimic_g1_flat/          # Bundled 1.6 MB Reference-Free SuperMimic teacher checkpoint
│   └── model_000.pt                 # Upstream official 240k-iter PPO teacher checkpoint
├── docs/
│   ├── supermimic_research_report.md # Full benchmarks, D1-D4 diagnostics & white-box disclosures
│   ├── UPSTREAM_README.md           # Original IsaacLab-Decoupled-WBC README
│   ├── training.md                  # Upstream training & keyboard control reference
│   └── deployment.md                # Physical Unitree G1 deployment guide
└── scripts/                         # One-command bash launchers (train/play/eval/diagnostics)
```

---

## Documentation

- **[Technical & Mechanistic Research Report](docs/supermimic_research_report.md)** — Complete 4-way benchmarks, `D1–D4` mechanistic & symmetry ablation studies, and white-box training disclosures.
- **[Upstream Training & Play Guide](docs/training.md)** & **[Physical G1 Deployment Guide](docs/deployment.md)** — Original HTD environment and Unitree SDK2 deployment manuals.
- **[Contributing & Reproducibility Protocol](CONTRIBUTING.md)** — Baseline integrity checks and code standards.

---

## Acknowledgements & License

Released under the **[BSD 3-Clause License](LICENSE.txt)**.
- Environment, decoupled G1 controller formulation, and `240k`-iter PPO reference checkpoint originate from **[IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC)** (*Humanoid Touch Dream*, CMU) and **[LeggedLab](https://github.com/Hellod035/LeggedLab)**.
- Differentiable world-model training paradigm is inspired by **SuperMimic** and community reproductions curated in **[awesome-supermimic](https://github.com/miao-org/awesome-supermimic/)**.

## Citation

If you find this repository or our mechanistic findings useful in your research, please cite this project alongside *SuperMimic* and *Humanoid Touch Dream*:

```bibtex
@software{supermimic_decoupled_wbc_2026,
  title   = {SuperMimic-Decoupled-WBC: Reference-Free Differentiable World Model RL for Humanoid Loco-Manipulation Whole-Body Control},
  author  = {{SuperMimic-Decoupled-WBC Contributors}},
  year    = {2026},
  url     = {https://github.com/xuejun7777/SuperMimic-Decoupled-WBC},
  license = {BSD-3-Clause}
}

@misc{niu2026htd,
  title         = {Learning Versatile Humanoid Manipulation with Touch Dreaming},
  author        = {Yaru Niu and Zhenlong Fang and Binghong Chen and Shuai Zhou and Revanth Senthilkumaran and Hao Zhang and Bingqing Chen and Chen Qiu and H. Eric Tseng and Jonathan Francis and Ding Zhao},
  year          = {2026},
  eprint        = {2604.13015},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

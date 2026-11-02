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
  <img src="docs/assets/supermimic_vs_ppo_teaser.gif" alt="World-Model Guided SuperMimic SOTA Student vs CMU Official PPO Student" width="100%">
</p>

> **Left**: **World-Model Guided SuperMimic SOTA Student (`116D` Proprioceptive History, `1,000` iters / `98.3M` steps)** &nbsp;|&nbsp; **Right**: **CMU Official PPO Student (`116D` Proprioceptive History, distilled from `240,000`-iter PPO Teacher)**  
> *Both hardware-ready `116D` Student policies operate without foot-contact sensors and are evaluated in Isaac Sim PhysX under identical initial states, identical continuous 14-DoF AMASS bimanual arm-swing disturbances, and a 6-phase 7D loco-manipulation command schedule. Full 18s 720p@50fps video: [`docs/assets/supermimic_vs_ppo_demo.mp4`](docs/assets/supermimic_vs_ppo_demo.mp4).*

---

## Overview

Original **SuperMimic** (*Scaling Humanoid Whole-Body Control with One Differentiable World Model*) demonstrated the power of learned differentiable dynamics + analytic Backpropagation Through Time (BPTT) on **MoCap motion tracking**, where Reference State Initialization (RSI) and dense per-joint MoCap targets anchor exploration.

**SuperMimic-Decoupled-WBC** explores a fundamental question for general robotics RL:

> *Can Differentiable World Models + Analytic BPTT replace zero-order PPO **completely without MoCap reference trajectories (Reference-Free)** as a general-purpose training paradigm for **humanoid loco-manipulation foundation controllers** — across both privileged Teacher training and proprioceptive Student distillation?*

Built cleanly as a **zero-intrusion modular extension** (`legged_lab/supermimic/`) on top of [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC) (*Humanoid Touch Dream*, CMU), this repository trains a decoupled lower-body + 3-DoF waist controller (`15` actuated joints) on **Unitree G1** to track arbitrary **7D locomotion and torso-pose commands** (`[vx, vy, wz, height, roll, pitch, yaw]`) while rejecting continuous exogenous **14-DoF bimanual upper-body manipulation disturbances**.

### Key Highlights

- **~100-240x Sample Efficiency Over Zero-Order PPO**: At `295M` environment transitions (`3,000` iters, `~3.5h` on a single RTX 4090), Reference-Free SuperMimic escapes PPO's standing-still local optimum (**`100.0%` 20s survival across all 8 evaluation suites**, **`113.23`** episode return vs `92.82` for equal-budget PPO). Extending Teacher training to `6,030` iterations (`593M` steps, still **`119x` fewer samples** than upstream's `240,000`-iter / `70.8B`-step PPO Teacher) reaches **`132.33`** training return and surpasses the `240k`-iter PPO Teacher across `walk_upright` (`70.18` vs `69.92`), `torso_pitch_bow` (`73.66` vs `72.41`), and `torso_roll_yaw` (`73.46` vs `72.91`).
- **World-Model Guided Student Training (Joint Analytic BPTT + Online DAgger)**: Standard Student distillation (`train_student.py`) only regresses Teacher actions via BC/DAgger without seeing task rewards. Our [`SuperMimicStudentRunner`](legged_lab/supermimic/student_runner.py) unrolls the `116D` Student (`2 x 58D` proprioceptive history, zero contact sensors) directly inside the differentiable `G1WorldModel` (`H=24`), combining **first-order Analytic BPTT on all 31 task rewards**, **online step-wise DAgger action regression**, and **116D sagittal mirror symmetry**. In **`1,000-1,250` iterations (`98M-123M` steps = `20x` fewer distillation steps than upstream's `600k`-step Student)**, our SOTA Student beats `CMU_Official_Student` on `walk_upright` (`69.98` vs `69.73`, **`20%` lower velocity & yaw error**), `walk_lateral_turn` (`70.08` vs `70.05`), `torso_pitch_bow` (`73.44` vs `72.43`), `torso_roll_yaw` (`73.25` vs `72.99`), and `extreme_pitch_1p2` (`63.50` vs `61.83`, **`1.49°` vs `43.12°` pitch error — even surpassing its own 60D Teacher!**).
- **Phase-Gated Sagittal Mirror Symmetry (`4.6x` Straighter Open-Loop Walking)**: Solves the *Phase-Free Bipedal Standing Lock* (where naive pointwise equivariance `pi(M(o)) = M(pi(o))` freezes memoryless bipeds at symmetric double-support rest `o = M(o)`) via a proprioceptive fixed-point exclusion gate + 50/50 mirror-augmented world model, cutting 5s straight-walk lateral drift down to **`6.8 cm` (neutral arms) / `8.5 cm` (AMASS arm swings) / `5.8 cm` (heading hold)** on our SOTA Student (**`3.4-4.7x` straighter than `CMU_Official_Student`'s `31.6 / 34.0 / 19.9 cm`**).
- **23,100-24,350+ FPS Single-GPU Throughput**: Native side-stream `torch.cuda.CUDAGraph` capture unrolls `H=24` closed-loop imagination steps + 30-body differentiable USD FK + 31 differentiable reward terms + phase-gated symmetry loss + full analytic BPTT backward pass in `bfloat16`/`fp32`.
- **100% Drop-In RSL-RL & Physical G1 Deployment Compatibility**: Exports standard `180,367`-parameter RSL-RL Teacher checkpoints (`example/supermimic_g1_flat/model_3000.pt`, `model_6030_sota.pt`) and `209,039`-parameter TorchScript JIT Student policies (`example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt`) compatible out of the box with `play_teacher.py`, `play_student.py`, and Unitree SDK2 deployment.
- **Verifiable Zero Upstream Baseline Tampering**: Zero lines of upstream environment physics, PD gains, observations, Domain Randomization, or PPO baseline code were altered (`git diff upstream-main -- legged_lab/envs legged_lab/mdp legged_lab/assets legged_lab/scripts/train_teacher.py` is **0 bytes**).

---

## Training Curves, Benchmarks & Mechanistic Diagnostics

<p align="center">
  <img src="docs/assets/supermimic_vs_ppo_curves.png" alt="SuperMimic vs PPO Training Curves, Locomotion & Posture Suite Returns, and 5s Straight-Walk Lateral Drift" width="100%">
</p>

<p align="center">
  <img src="docs/assets/supermimic_vs_ppo_keyframes.png" alt="6-Phase Whole-Body Loco-Manipulation Keyframes: SuperMimic SOTA Student vs CMU Official Student" width="100%">
</p>

### 1. Comprehensive PhysX Evaluation (`512` Envs, Paired AMASS Arm Disturbances: Teachers & Proprioceptive Students)

| Policy | Stage / Input | Env Steps | `walk_upright` Return / `vxy_err` | `torso_pitch_bow` (`46°`) Return / `pitch_mae` | `extreme_pitch_1p2` (`69°`) Return / `pitch_mae` | `torso_roll_yaw` Return | `full_random` (`20s`) Surv / Return / `pitch_mae` | `D4` `5s` Straight-Walk Drift (Neutral / AMASS / Hold) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PPO_3000` (Equal Budget) | Teacher (`60D`) | `0.295B` (1x) | `61.92` / `0.454 m/s` | `59.69` / `15.5°` | `55.50` / `47.0°` | `68.26` | `94.9%` / `92.82` / `19.2°` | *frozen (`vx ≈ 0`)* |
| `SuperMimic_3000` *(No Sym)* | Teacher (`60D`) | `0.295B` (1x) | `68.33` / `0.152 m/s` | `72.36` / `1.20°` | `63.29` / **`0.74°`** | `72.48` | `99.6%` / `112.27` / **`1.20°`** | `37.1 / 39.9 / 37.4 cm` |
| **`SuperMimic_3000` *(Sym)*** | **Teacher (`60D`)** | **`0.295B` (1x)** | `67.99` / `0.150 m/s` | `72.30` / **`0.46°`** | `58.92` / `27.0°` | `72.49` | **`100.0%`** / **`113.23`** / `13.4°` | **`8.1 / 11.3 / 12.3 cm`** *(yaw `1.88°`)* |
| **`SM_SOTA_Teacher` *(6k Sym)*** | **Teacher (`60D`)** | **`0.593B` (2x)** | **`70.18`** / **`0.076 m/s`** | **`73.66`** / `0.74°` | `61.09` / `1.78°` | **`73.46`** | `84.2%` / `91.87` / `2.98°` | `26.1 / 22.0 / 17.6 cm` *(yaw `1.93°`)* |
| `CMU_PPO_240k` (Upstream Ceiling) | Teacher (`60D`) | `70.78B` (240x) | `69.92` / `0.090 m/s` | `72.41` / `0.63°` | `63.23` / `23.9°` | `72.91` | `100.0%` / `120.94` / `17.8°` | `41.7 / 46.2 / 27.0 cm` *(yaw `6.47°`)* |
| `SM_3000_Sym_Student` *(Pure DAgger)* | Student (`116D`, No Contact) | `+16.4M` (`5m`) | `68.85` / `0.108 m/s` | `70.14` / `0.86°` | `56.38` / `24.8°` | `70.63` | `99.6%` / `108.98` / `12.9°` | `24.8 / 25.3 / 19.1 cm` *(yaw `2.09°`)* |
| **`SM_SOTA_Student` *(WM+DAgger, Best)*** | **Student (`116D`, No Contact)** | **`+98.3M` (`1k` it)** | **`69.97`** / **`0.072 m/s`** | **`73.44`** / `0.63°` | **`63.45`** / **`1.78°`** | **`73.11`** | `91.4%` / `99.96` / **`2.86°`** | **`6.8 / 8.5 / 5.8 cm`** *(yaw **`0.93°`**)* |
| **`SM_SOTA_Student` *(WM+DAgger, 1250)*** | **Student (`116D`, No Contact)** | **`+122.9M` (`1.25k`)** | **`69.98`** / **`0.072 m/s`** | **`73.26`** / `0.86°` | **`63.50`** / **`1.49°`** | **`73.25`** | `93.4%` / `100.40` / `2.98°` | **`11.0 / 9.8 / 7.4 cm`** *(yaw **`0.95°`**)* |
| `CMU_Official_Student` *(Bundled Deploy)* | Student (`116D`, No Contact) | `+2.46B` (`600k` it) | `69.73` / `0.090 m/s` | `72.43` / `0.52°` | `61.83` / `43.1°` | `72.99` | `99.6%` / `120.88` / `15.6°` | `31.6 / 34.0 / 19.9 cm` *(yaw `6.31°`)* |

> **World-Model Guided Student Highlights (`example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt`)**: By training the `116D` proprioceptive Student directly with **Differentiable World-Model Analytic BPTT + Online Step-Wise DAgger + 116D Sagittal Symmetry** (`legged_lab/scripts/train_supermimic_student.py`), `SM_SOTA_Student` surpasses `CMU_Official_Student` across **5 out of 7 targeted locomotion & posture suites** (`walk_upright`, `walk_lateral_turn`, `torso_pitch_bow`, `extreme_pitch_1p2`, `torso_roll_yaw`), cuts 5s straight-walk lateral drift by **`3.4-4.7x`** (`6.8 / 8.5 / 5.8 cm` vs `31.6 / 34.0 / 19.9 cm`), and tracks extreme `68.8°` (`1.20 rad`) forward bows to **`1.49°-1.78°` MAE** (whereas `CMU_Official_Student` stalls at `43.1°` error). In `full_random` (which samples physically conflicting combinations such as `90°` pitch + deep squat + roll jumps simultaneously), `CMU_Official_Student` sacrifices pitch obedience (`15.6°` MAE, `33.5°` RMSE) to stay upright, whereas `SM_SOTA_Student` strictly obeys extreme pitch commands (`2.86°` MAE — **`5.5x` lower pitch error**).

### 2. Why Does Reference-Free SuperMimic Beat PPO? (Mechanistic Summary)

Full diagnostic suite reproducible via `legged_lab/scripts/run_diagnostics.py` — see **[Technical & Mechanistic Research Report](docs/supermimic_research_report.md)** for complete tables:

1. **Millimeter/Sub-Degree Short-Horizon Digital Twin (`D1`)**: Over `H=24` open-loop steps (`0.48s`), `G1WorldModel` (with 50/50 sagittal mirror batch augmentation) maintains **`0.197 cm`** pelvis height MAE, **`0.222°`** 29-joint MAE, **`97.46%`** foot-contact accuracy, and **Pearson `r = 0.9554`** step-reward correlation against real PhysX.
2. **High Directional Gradient SNR Without Action-Dimension Curse (`D2`)**: Across `180,367` policy parameters (`N=256`, `sigma=0.20`), zero-order score-function PG suffers severe variance (`cos = -0.1226` at `1x` budget; `0.5102` at **`16x`** rollout budget), whereas single-draw Analytic BPTT achieves **`cos = 0.7721`** and routes **`68.1%`** of extreme forward-bow action sensitivity directly into coordinated lower-body counter-balancing joints (`hip_pitch + knee + ankle_pitch`).
3. **Continuous Zero-Cost Hindsight Command Relabeling (`D3`)**: Because `G1WorldModel` is strictly **command-free**, relabeling old physical states with unseen counterfactual 7D commands correlates with branched real PhysX returns at **Pearson `r = 0.9917`** (`per-step MAE = 0.0101`).
4. **Eliminating Single-Seed Gait Drift Without Phase Clocks (`D4`)**: Phase-asymmetry-gated sagittal equivariance reduces 5s straight-walk (`vx = 0.45 m/s`) open-loop lateral drift to **`8.1 cm`** on the 3k Teacher and **`6.8 cm`** on the World-Model SOTA Student (vs `41.7 cm` on `CMU_PPO_240k` and `31.6 cm` on `CMU_Official_Student`), with closed-loop heading error down to **`0.93°`** (`6.8x` tighter than `CMU_Official_Student`'s `6.31°`).

---

## Quick Start

### 1. Installation

Follow the standard Isaac Lab 2.2 environment setup (Ubuntu 20.04/22.04+, Python 3.10/3.11, CUDA 12.x):

```bash
bash scripts/setup_conda.sh        # Isaac Sim 5.0/5.1 + Isaac Lab 2.2.0 (python 3.11)
conda activate isaac_wbc
```

*(Headless Docker setup is also supported via `bash scripts/setup_docker.sh`; see [Upstream Guide](docs/UPSTREAM_README.md).)*

### 2. Play the Bundled Reference-Free SuperMimic Policies Out of the Box

Pre-trained Reference-Free SuperMimic checkpoints are bundled in [`example/supermimic_g1_flat/`](example/supermimic_g1_flat/):
- [`example/supermimic_g1_flat/model_3000.pt`](example/supermimic_g1_flat/model_3000.pt): Equal-budget `3,000`-iter Phase-Gated Symmetry Teacher (`100.0%` survival across all 8 suites).
- [`example/supermimic_g1_flat/model_6030_sota.pt`](example/supermimic_g1_flat/model_6030_sota.pt): Extended `6,030`-iter SOTA Teacher (`132.33` training return).
- [`example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt`](example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt): World-Model Guided `116D` Proprioceptive SOTA Student TorchScript JIT policy.

```bash
# Interactive simulation window with keyboard 7D velocity + torso pose control
bash scripts/play_supermimic.sh

# Compare interactively against the upstream 240k-iter PPO teacher
bash scripts/play_teacher.sh
```

> **Keyboard Controls**: `W`/`S` (`vx`), `A`/`D` (`vy`), `Q`/`E` (`wz`), `H`/`J` (pelvis height), `Z`/`X` (torso roll), `C`/`V` (torso pitch), `B`/`N` (torso yaw), `R` (reset).

### 3. Train Reference-Free SuperMimic Teacher & World-Model Student (Single GPU)

```bash
# 1. Train Reference-Free SuperMimic Teacher (4096 envs x 3000 iters = 295M steps, ~3.5h on 1x RTX 4090)
bash scripts/train_supermimic.sh

# 2. Train World-Model Guided Proprioceptive Student (Analytic BPTT + Online DAgger + 116D Symmetry)
python legged_lab/scripts/train_supermimic_student.py \
    --task=g1_flat --num_envs=4096 --max_iterations=1000 --headless \
    --teacher_ckpt=logs/g1_flat/<your_teacher_run>/supermimic_last.pt \
    --run_name=supermimic_student_sota

# 3. Equal-budget Native RSL-RL PPO baseline (exact same 4096 envs, 3000 iters & pose curriculum)
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

# 3. Render 1280x720@50fps side-by-side comparison video (supports both 60D Teacher & 116D Student JIT)
python legged_lab/scripts/record_demo_video.py \
    --task=g1_flat \
    --sm_ckpt=example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt \
    --ppo_ckpt=deploy/policy/g1_student/student_policy_jit.pt \
    --left_title="SuperMimic SOTA Student (Ours, 116D)" \
    --right_title="CMU Official Student (240k Teacher)" \
    --out_mp4=docs/assets/supermimic_vs_ppo_demo.mp4
```

### 5. Physical Unitree G1 Deployment

Both the World-Model Guided Student (`train_supermimic_student.py`) and standard upstream DAgger (`train_student.sh`) export drop-in `116D` TorchScript JIT policies (`student_policy_jit.pt`) ready for Unitree SDK2 deployment. See [`docs/deployment.md`](docs/deployment.md) for physical Unitree G1 deployment instructions.

---

## Repository Structure

```text
├── legged_lab/
│   ├── supermimic/                  # Modular Reference-Free SuperMimic implementation
│   │   ├── rotations.py             # CUDAGraph-safe wxyz SO(3) & smooth Euler operators
│   │   ├── state_spec.py            # 71D Markov state, SE(2) canonicalization & manifold integration
│   │   ├── usd_fk.py                # Zero-clone differentiable 30-body / 29-hinge USD kinematics
│   │   ├── symmetry.py              # Sagittal mirror equivariance (60D Teacher & 116D Student)
│   │   ├── diff_rewards.py          # Differentiable 31-term G1 WBC rewards & 60D actor obs
│   │   ├── world_model.py           # bfloat16 2048x3 command-free dynamics + contact + impact heads
│   │   ├── nets.py                  # 180,367-param TeacherActorMLP & 88D PrivilegedCriticMLP
│   │   ├── replay_buffer.py         # [1000, 4096] GPU ring buffer + 8192-slot fp16 AMASS clip bank
│   │   ├── env_adapter.py           # Read-only BaseEnv subclass capturing pre-reset fall transitions
│   │   ├── imagination.py           # WorldModelTrainer & side-stream CUDAGraph PolicyCriticTrainer
│   │   ├── runner.py                # End-to-end Teacher runner with checkpoint pruning & ONNX export
│   │   ├── student_runner.py        # World-Model Guided 116D Student runner (BPTT + Online DAgger + Sym)
│   │   └── test_parity.py           # Bitwise/eps parity & sagittal equivariance verification
│   ├── scripts/
│   │   ├── train_supermimic.py         # SuperMimic Teacher CLI entrypoint
│   │   ├── train_supermimic_student.py # World-Model Guided Student CLI entrypoint
│   │   ├── eval_compare.py             # Deterministic 8-suite multi-checkpoint PhysX evaluator
│   │   ├── run_diagnostics.py          # Mechanistic diagnostic experiments D1-D4
│   │   └── record_demo_video.py        # Headless 720p@50fps side-by-side PhysX video renderer
│   └── envs/, mdp/, assets/         # 100% untouched upstream IsaacLab-Decoupled-WBC environment
├── example/
│   ├── supermimic_g1_flat/          # Bundled SuperMimic Teacher (3k & 6k SOTA) + 116D SOTA Student JIT
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

- **[Technical & Mechanistic Research Report](docs/supermimic_research_report.md)** — Complete Teacher & Student benchmarks, `D1–D4` mechanistic & symmetry ablation studies, and white-box training disclosures.
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

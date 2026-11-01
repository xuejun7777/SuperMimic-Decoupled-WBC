# Technical Report: Reference-Free SuperMimic for Humanoid Loco-Manipulation Whole-Body Control

This document presents the full experimental benchmarks, mechanistic diagnostic studies (`D1–D4`), sagittal mirror-symmetry ablation (including the *Phase-Free Bipedal Standing Lock* phenomenon and its fixed-point-excluded solution), and methodological scope disclosures for **SuperMimic-Decoupled-WBC** — an open-source migration of differentiable-world-model Analytic BPTT (`SuperMimic`) completely detached from MoCap reference tracking onto 7D decoupled Unitree G1 locomotion and torso-pose control.

<p align="center">
  <img src="assets/supermimic_vs_ppo_teaser.gif" alt="Reference-Free SuperMimic vs Native RSL-RL PPO at Equal 295M Step Budget" width="100%">
</p>

<p align="center">
  <img src="assets/supermimic_vs_ppo_curves.png" alt="Training Curves, 8-Suite Returns, and 5s Straight-Walk Drift Comparison" width="100%">
</p>

---

## 1. System Architecture

```mermaid
flowchart LR
    Env["PhysX G1 Env (4096 Envs)"] -->|"71D Markov State + Contact + AMASS Clip ID"| RB["G1ReplayBuffer (1000 x 4096 GPU Ring)"]
    RB -->|"K=8 Windows + 50/50 Sagittal Mirror Aug"| WM["G1WorldModel (bf16 2048x3 Command-Free Dynamics + Contact + Impact)"]
    RB -->|"Initial States + 24-Step Arm Lookahead"| Imag["H=24 Differentiable Imagination (CUDAGraph)"]
    WM -->|"SO(3) Manifold Rollout"| Imag
    Actor["TeacherActorMLP (60 -> 512-256-128 -> 15)"] -->|"Closed-Loop Obs & Action"| Imag
    Imag -->|"USD Exact FK (< 3e-6 m)"| Rew["DifferentiableG1Rewards (31 Active Terms)"]
    Rew -->|"Analytic BPTT + Phase-Gated L_sym + GAE Critic"| Actor
```

- **Zero Upstream Baseline Modification**: All 283 upstream files in `legged_lab/envs/`, `legged_lab/mdp/`, `legged_lab/assets/`, and `legged_lab/scripts/train_teacher.py` remain 100% untouched (`git diff upstream-main -- legged_lab/envs legged_lab/mdp legged_lab/assets legged_lab/scripts/train_teacher.py` is 0 bytes).
- **Drop-In Deployment Compatibility**: The trained `TeacherActorMLP` (`60 -> [512, 256, 128] (ELU) -> 15`, `180,367` parameters, zero inference wrapper) shares the exact inference topology and 60D observation vector of the upstream RSL-RL PPO teacher, plugging directly into `play_teacher.py`, `train_student.py` (BC -> DAgger history distillation), and Unitree SDK2 physical G1 deployment.

---

## 2. Four-Way Quantitative Benchmark (`PPO_3000` vs `SM_3000_NoSym` vs `SM_3000_Sym` vs `240x`-Budget `CMU_PPO_240k`)

All checkpoints are evaluated deterministically across `512` parallel PhysX environments under identical initial states and paired continuous 14-DoF AMASS bimanual arm-swing disturbances via [`legged_lab/scripts/eval_compare.py`](../legged_lab/scripts/eval_compare.py):
1. **`PPO_3000`**: Native RSL-RL PPO trained from scratch for `3,000` iterations (`4,096 x 24 x 3,000 = 294.9M` steps, `52.5 min` on 1x RTX 4090).
2. **`SM_3000_NoSym`**: Reference-Free SuperMimic trained from scratch for `3,000` iterations (`294.9M` steps) without sagittal mirror augmentation or symmetry loss (`201.6 min` on 1x RTX 4090 at `24,314 FPS`).
3. **`SM_3000_Sym` (Ours, Bundled Default [`example/supermimic_g1_flat/model_3000.pt`](../example/supermimic_g1_flat/model_3000.pt))**: Reference-Free SuperMimic trained from scratch for `3,000` iterations (`294.9M` steps) with 50/50 mirror-augmented World Model + phase-asymmetry-gated sagittal equivariance `L_sym` (`212.5 min` on 1x RTX 4090 at `23,132 FPS`).
4. **`CMU_PPO_240k` (Asymptotic Ceiling Reference)**: Official upstream flagship PPO checkpoint [`example/model_000.pt`](../example/model_000.pt) trained for `240,000` iterations across `12,288` envs (`70.78B` steps = **`240x` more environment transitions**).

### 2.1 Full Uncurriculated 7D Command Distribution (`full_random`, 20s / 1000-step Episode)

| Policy (Env Transitions) | 20s Survival ↑ | Ep Length ↑ | Ep Return ↑ | `vxy_err` (m/s) ↓ | `wz_mae` (rad/s) ↓ | `h_mae` (cm) ↓ | `roll_mae` (rad) ↓ | `pitch_mae` (rad) ↓ | `yaw_mae` (rad) ↓ | `d_act` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PPO_3000`** (`0.295B`, 1x) | `94.9%` | `966.0` | `92.82` | `0.271` | `0.396` | `8.73` | `0.032` | `0.335` (`19.2°`) | `0.025` | `0.179` *(frozen)* |
| **`SM_3000_NoSym`** (`0.295B`, 1x) | `99.6%` | `999.1` | `112.27` | `0.133` | `0.114` | `3.77` | `0.031` | **`0.021` (`1.2°`)** | `0.024` | `0.754` *(stepping)* |
| **`SM_3000_Sym` (Default)** (`0.295B`, 1x) | **`100.0%`** | **`1000.0`** | **`113.23`** | **`0.129`** | **`0.104`** | **`3.63`** | **`0.020`** | `0.234` (`13.4°`) | **`0.016`** | `0.684` *(stepping)* |
| `CMU_PPO_240k` (`70.78B`, **240x**) | `100.0%` | `1000.0` | `120.94` | `0.093` | `0.106` | `2.87` | `0.016` | `0.311` (`17.8°`) | `0.019` | `0.647` *(stepping)* |

### 2.2 Seven Targeted Locomotion & Body-Pose Suites (`512` Envs x `10s`)

| Suite | Policy | Survival ↑ | Return ↑ | `vxy_err` (m/s) ↓ | `wz_mae` (rad/s) ↓ | `h_mae` (cm) ↓ | `pitch_mae` (rad) ↓ | `roll / yaw` (rad) ↓ | `d_act` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`walk_upright`**<br>`vx=0.45, h=0.74` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `100.0%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `61.92`<br>`68.33`<br>**`67.99`**<br>`69.92` | `0.454`<br>`0.152`<br>**`0.150`**<br>`0.090` | `0.050`<br>`0.125`<br>**`0.099`**<br>`0.108` | `4.58`<br>`1.52`<br>`3.86`<br>`0.65` | `0.014`<br>`0.010`<br>`0.015`<br>`0.014` | `0.011 / 0.012`<br>`0.011 / 0.010`<br>**`0.006 / 0.012`**<br>`0.009 / 0.009` | `0.161`<br>`1.076`<br>**`1.032`**<br>`1.075` |
| **`walk_lateral_turn`**<br>`vy=0.35, wz=0.60, h=0.72` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `100.0%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `57.08`<br>`68.86`<br>**`67.66`**<br>`70.22` | `0.355`<br>**`0.070`**<br>`0.131`<br>`0.086` | `0.605`<br>`0.115`<br>`0.144`<br>`0.089` | `3.95`<br>`1.71`<br>`2.82`<br>`0.69` | `0.013`<br>`0.010`<br>`0.016`<br>`0.010` | `0.008 / 0.012`<br>`0.015 / 0.008`<br>**`0.012 / 0.009`**<br>`0.010 / 0.006` | `0.154`<br>`1.055`<br>**`0.938`**<br>`1.050` |
| **`squat_deep`**<br>`v=0, h=0.42` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `95.3%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `57.38`<br>`69.74`<br>**`72.47`**<br>`70.01` | `0.123`<br>`0.054`<br>**`0.044`**<br>`0.021` | `0.150`<br>`0.051`<br>**`0.044`**<br>`0.031` | `16.30`<br>`5.60`<br>**`4.93`**<br>`1.41` | `0.029`<br>`0.019`<br>**`0.012`**<br>`0.012` | `0.014 / 0.009`<br>`0.008 / 0.006`<br>**`0.008 / 0.004`**<br>`0.007 / 0.008` | `0.354`<br>`0.669`<br>`0.541`<br>`0.171` |
| **`torso_pitch_bow`**<br>`v=0, h=0.68, pitch=0.80` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `99.8%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `59.69`<br>`72.36`<br>**`72.30`**<br>`72.41` | `0.046`<br>`0.024`<br>`0.026`<br>`0.024` | `0.056`<br>`0.019`<br>**`0.018`**<br>`0.023` | `10.62`<br>`1.05`<br>`3.07`<br>`3.10` | `0.271`<br>`0.021`<br>**`0.008` (`0.46°`)**<br>`0.011` | `0.017 / 0.013`<br>`0.006 / 0.007`<br>**`0.008 / 0.007`**<br>`0.006 / 0.008` | `0.162`<br>`0.139`<br>**`0.133`**<br>`0.105` |
| **`extreme_pitch_1p2`**<br>`v=0, h=0.62, pitch=1.20` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `99.6%`<br>`99.2%`<br>**`100.0%`**<br>`100.0%` | `55.50`<br>**`63.29`**<br>`58.92`<br>`63.23` | `0.042`<br>`0.025`<br>**`0.018`**<br>`0.024` | `0.054`<br>`0.019`<br>**`0.018`**<br>`0.028` | `6.16`<br>`1.88`<br>**`1.97`**<br>`2.28` | `0.821` (`47.0°`)<br>**`0.013` (`0.74°`)**<br>`0.472` (`27.0°`)<br>`0.417` (`23.9°`) | `0.013 / 0.012`<br>`0.014 / 0.010`<br>**`0.007 / 0.007`**<br>`0.009 / 0.010` | `0.153`<br>`0.180`<br>**`0.135`**<br>`0.106` |
| **`walk_squat_pitch`**<br>`vx=0.35, h=0.55, pitch=0.50` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `99.8%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `59.34`<br>`66.98`<br>**`66.40`**<br>`68.32` | `0.366`<br>`0.123`<br>`0.133`<br>`0.074` | `0.050`<br>`0.100`<br>**`0.074`**<br>`0.069` | `4.32`<br>`1.15`<br>**`1.07`**<br>`0.79` | `0.018`<br>`0.010`<br>`0.022`<br>`0.014` | `0.017 / 0.013`<br>`0.010 / 0.013`<br>**`0.011 / 0.008`**<br>`0.010 / 0.009` | `0.153`<br>`1.184`<br>**`1.079`**<br>`1.253` |
| **`torso_roll_yaw`**<br>`v=0, h=0.72, roll=0.40, yaw=0.80` | `PPO_3000`<br>`SM_3000_NoSym`<br>**`SM_3000_Sym`**<br>`CMU_PPO_240k` | `100.0%`<br>`100.0%`<br>**`100.0%`**<br>`100.0%` | `68.26`<br>`72.48`<br>**`72.49`**<br>`72.91` | `0.041`<br>`0.027`<br>**`0.026`**<br>`0.025` | `0.050`<br>`0.023`<br>**`0.021`**<br>`0.025` | `6.11`<br>`2.90`<br>**`2.55`**<br>`2.94` | `0.013`<br>`0.009`<br>`0.011`<br>`0.009` | `0.017 / 0.011`<br>`0.012 / 0.005`<br>**`0.011 / 0.009`**<br>`0.010 / 0.003` | `0.144`<br>`0.142`<br>**`0.114`**<br>`0.105` |

### 2.3 Hardware-Ready Proprioceptive Student Benchmark (`116D` 2-Step History, Zero Contact Sensor)

Because physical Unitree G1 robots do not have binary foot-contact sensors, [`legged_lab/scripts/train_student.py`](../legged_lab/scripts/train_student.py) distills the 60D Teacher into a `116 -> [512, 256, 128] (ELU) -> 15` `StudentMLP` (`209,039` parameters) that observes 2 steps of 58D proprioception (`obs[:, :58]`). We compare:
- **`SM_3000_Sym_Student` (Ours, Bundled [`example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt`](../example/supermimic_g1_flat/student_checkpoints/student_policy_jit.pt))**: Distilled from `SM_3000_Sym` in just **`4,000` steps** (`1,500` BC + `2,500` DAgger across `4,096` envs = `16.4M` steps, `~5 min` on 1x RTX 4090, final action `RMSE = 0.0461`).
- **`CMU_Official_Student` ([`deploy/policy/g1_student/student_policy_jit.pt`](../deploy/policy/g1_student/student_policy_jit.pt))**: Upstream official hardware deployment Student distilled for `600,000` steps (`250k` BC + `350k` DAgger = `2.46B` steps) from `CMU_PPO_240k`.

| Suite | Student Policy | Survival ↑ | Return ↑ | `vxy_err` (m/s) ↓ | `wz_mae` (rad/s) ↓ | `h_mae` (cm) ↓ | `pitch_mae` (rad) ↓ | `roll / yaw` (rad) ↓ | `d_act` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`full_random`** (`20s`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | `99.6%`<br>`100.0%` | `108.98` *(`96.3%` of Teacher)*<br>`121.04` | `0.154`<br>`0.097` | `0.138`<br>`0.108` | `4.21`<br>`3.33` | **`0.225` (`12.9°`)**<br>`0.298` (`17.1°`) | `0.029 / 0.019`<br>`0.018 / 0.019` | `0.743`<br>`0.659` |
| **`walk_upright`** (`vx=0.45`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | **`68.85`**<br>`69.85` | **`0.108`** *(beats Teacher!)*<br>`0.089` | `0.124`<br>`0.122` | **`1.61`**<br>`0.70` | `0.014`<br>`0.013` | `0.009 / 0.011`<br>`0.009 / 0.009` | `1.031`<br>`1.057` |
| **`walk_lateral_turn`** (`vy=0.35, wz=0.60`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | **`67.32`**<br>`70.05` | `0.183`<br>`0.084` | `0.143`<br>`0.100` | `2.29`<br>`0.69` | `0.011`<br>`0.010` | `0.012 / 0.008`<br>`0.011 / 0.006` | `0.907`<br>`1.046` |
| **`squat_deep`** (`h=0.42`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | `99.8%`<br>`100.0%` | **`70.99` *(beats CMU!)* **<br>`69.75` | `0.054`<br>`0.018` | `0.070`<br>`0.028` | `1.86`<br>`1.45` | `0.032`<br>`0.014` | `0.009 / 0.012`<br>`0.009 / 0.007` | `0.679`<br>`0.148` |
| **`torso_pitch_bow`** (`pitch=0.80`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | **`70.14`**<br>`72.40` | `0.030`<br>`0.024` | **`0.020`**<br>`0.022` | `4.50`<br>`3.21` | `0.015` (`0.86°`)<br>`0.010` (`0.57°`) | `0.014 / 0.009`<br>`0.006 / 0.006` | `0.163`<br>`0.104` |
| **`extreme_pitch_1p2`** (`pitch=1.20`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | `56.38`<br>`62.11` | `0.024`<br>`0.019` | **`0.022`**<br>`0.024` | `4.89`<br>`1.59` | **`0.433` (`24.8°`, `1.7x` lower!)**<br>`0.740` (`42.4°`) | `0.009 / 0.006`<br>`0.008 / 0.013` | `0.215`<br>`0.104` |
| **`walk_squat_pitch`** (`vx=0.35, h=0.55, p=0.50`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | **`64.80`**<br>`68.33` | `0.129`<br>`0.062` | `0.102`<br>`0.074` | `2.92`<br>`0.75` | `0.015`<br>`0.012` | `0.012 / 0.009`<br>`0.011 / 0.011` | `1.076`<br>`1.289` |
| **`torso_roll_yaw`** (`r=0.40, y=0.80`) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | **`70.63`**<br>`72.94` | `0.035`<br>`0.025` | `0.027`<br>`0.024` | `3.49`<br>`2.90` | `0.014`<br>`0.009` | `0.012 / 0.024`<br>`0.009 / 0.005` | `0.145`<br>`0.104` |
| **`D4` 5s Straight Walk** (`vx=0.45`, Open-Loop Neutral / AMASS / Heading-Hold) | **`SM_3000_Sym_Student`**<br>`CMU_Official_Student` | **`100.0%`**<br>`100.0%` | — | — | — | — | **Yaw Drift:**<br>**`5.35° / 6.88° / 2.09°`**<br>`13.31° / 15.32° / 6.31°` | **Lateral Drift `abs(y)`:**<br>**`24.8 / 25.3 / 19.1 cm`**<br>`31.6 / 34.0 / 19.9 cm` | — |

---

## 3. Mechanistic Diagnostics (`D1–D4`): Why Does Reference-Free SuperMimic Outperform PPO?

Reproducible via [`legged_lab/scripts/run_diagnostics.py`](../legged_lab/scripts/run_diagnostics.py) (`N = 256` parallel PhysX environments, evaluated on `SM_3000_Sym`):

### D1. Short-Horizon (`0.48s`) World Model Physical Fidelity vs Real PhysX

| Open-Loop Horizon `h` (Time) | Pelvis Height MAE (cm) ↓ | Root Geodesic Angle Err (deg) ↓ | Lin Vel Err (m/s) ↓ | 29-Joint Angle MAE (deg) ↓ | Foot Contact Acc (%) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`h = 1`** (`0.02 s`) | **`0.023 cm`** (`0.23 mm`) | **`0.055°`** | **`0.0064`** | **`0.040°`** | **`98.83%`** |
| **`h = 4`** (`0.08 s`) | `0.048 cm` | `0.166°` | `0.0114` | `0.087°` | `98.44%` |
| **`h = 8`** (`0.16 s`) | `0.068 cm` | `0.370°` | `0.0171` | `0.117°` | `97.66%` |
| **`h = 16`** (`0.32 s`) | `0.109 cm` | `0.917°` | `0.0303` | `0.160°` | `98.44%` |
| **`h = 24`** (`0.48 s`, BPTT Cutoff) | **`0.197 cm`** (`1.97 mm`) | **`1.685°`** | **`0.0488`** | **`0.222°`** | **`97.46%`** |

- **Step-Reward Correlation vs Real PhysX**: **Pearson `r = 0.9554`**. Moreover, 50/50 sagittal mirror batch augmentation (`augment_wm_window_half_`) reduced 24-step open-loop error across every physical coordinate (`pelvis_z_mae` `0.245 -> 0.197 cm`, `orient_err` `2.043° -> 1.685°`, `joint_pos_mae` `0.285° -> 0.222°`).

### D2. First-Order Analytic BPTT vs Zero-Order Score-Function PG Gradient SNR & Multi-Joint Jacobian Synergy

- **Gradient Directional Cosine Consistency (`180,367`-D Actor Parameters, `N=256`, `H=24`, `sigma=0.20`)**:
  - **Zero-Order PG (`1x` budget, GAE `gamma=0.99, lambda=0.95`)**: **`-0.1226`** (near-orthogonal noise across independent draws).
  - **Zero-Order PG (`16x` rollout budget)**: **`0.5102`**.
  - **First-Order Analytic BPTT (`1x` budget)**: **`0.7721`**.
- **Multi-Joint Action Jacobian Allocation under Extreme Bow (`pitch = 1.20 rad / 68.8°`, past `waist_pitch +-0.52 rad` limit)**:
  - Analytic sensitivity `|dJ_24 / da_0|` allocates **`68.1%` of total sagittal gradient directly into lower-body joints** (`hip_pitch = 0.1692`, `knee = 0.1081`, `ankle_pitch = 0.0497` per leg vs `waist_pitch = 0.3064`).

### D3. Implicit Hindsight Counterfactual Command Relabeling

- Branching `N = 256` states (visited under arbitrary prior commands) across **4 unseen counterfactual commands** (`1,024` branches, 12-step closed-loop rollout) yields **Pearson `r = 0.9917`** (`per-step MAE = 0.0101`) between imagined returns and real branched PhysX returns. Because `G1WorldModel` is strictly command-free, every replay state provides valid first-order gradients across the entire continuous 7D command manifold.

### D4. Eliminating Single-Seed Crab-Walk Drift Without Phase Clocks (`5.0s` Straight Walk `vx = 0.45 m/s`)

| Policy | Straight-Walk Control Mode (`5.0s`, `256` Envs) | Forward `x` (m) ↑ | Mean Heading Drift (deg) ↓ | Mean Lateral Drift `abs(y)` (m) *(signed)* ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **`CMU_PPO_240k`** *(240x budget, no sym)* | Open-loop `wz = 0` + Neutral arms<br>Open-loop `wz = 0` + AMASS arms<br>Closed-loop Heading Hold + AMASS arms | `1.671`<br>`1.754`<br>`1.798` | `15.06°`<br>`18.67°`<br>`6.47°` | `0.417 m` *(`-0.417 m`)*<br>`0.462 m` *(`-0.459 m`)*<br>`0.270 m` *(`-0.269 m`)* |
| **`SM_3000_NoSym`** *(1x budget, no sym)* | Open-loop `wz = 0` + Neutral arms<br>Open-loop `wz = 0` + AMASS arms<br>Closed-loop Heading Hold + AMASS arms | `1.580`<br>`1.503`<br>`1.516` | `6.29°`<br>`5.89°`<br>`2.10°` | `0.371 m` *(`+0.371 m`)*<br>`0.399 m` *(`+0.398 m`)*<br>`0.374 m` *(`+0.374 m`)* |
| **`SM_3000_UncondSym`** *(Naive `L_sym` from `it=0`)* | Open-loop `wz = 0` + AMASS arms | `0.020` *(locked!)* | `1.12°` | `0.021 m` *(standing trap)* |
| **`SM_3000_Sym` (Ours, Phase-Gated)** | **Open-loop `wz = 0` + Neutral arms**<br>**Open-loop `wz = 0` + AMASS arms**<br>**Closed-loop Heading Hold + AMASS arms** | **`1.451`**<br>**`1.493`**<br>**`1.495`** | **`4.60°`**<br>**`4.71°`**<br>**`1.88°`** | **`0.081 m`** *(`-0.081 m`, **`4.6-5.1x` lower!**)*<br>**`0.113 m`** *(`-0.103 m`, **`3.5-4.1x` lower!**)*<br>**`0.123 m`** *(`-0.120 m`, **`2.2-3.0x` lower!**)* |

#### Why Naive Pointwise Symmetry Fails on Phase-Free Bipeds ("Bipedal Buridan's Ass") & How Phase-Gated Equivariance Solves It

1. **Root Cause of Drift Without Symmetry**: Because `BaseEnv`'s `track_lin_vel_xy_exp` uses a wide Gaussian kernel (`sigma = 0.5 m/s`) with no world `(x, y)` position observation and no symmetry regularization (`RslRlSymmetryCfg = None`), a small single-seed lateral gait asymmetry `|dv_y| ≈ 0.07 m/s` retains `exp(-0.07^2 / 0.5^2) = 98.1%` of the velocity reward, producing `~37-46 cm` of crab slip over `5s` in both `SM_3000_NoSym` and official `CMU_PPO_240k`.
2. **Fixed-Point Singularity of Naive Pointwise Equivariance**: Unlike clock-driven locomotion policies that observe a periodic gait phase `phi in [0, 2*pi)` with `M(phi) = (phi + pi) mod 2*pi`, `BaseEnv`'s 60D observation is **strictly phase-free**. At symmetric double-support standing under a straight-walk command, `o_0 = M_obs(o_0)` is a **fixed point** of sagittal reflection. Enforcing unconditional equivariance `pi(M_obs(o_0)) = M_act(pi(o_0))` collapses `pi(o_0)` onto `M_act(pi(o_0))`, forcing **left leg actions to equal right leg actions** (`feet_air_time = 0.00016`, `forward_progress_x = 0.02 m` standing lock) and over-damping narrow-Gaussian (`std = 0.1`) posture skill unzipping.
3. **Proprioceptive Fixed-Point-Excluded Equivariance (`legged_lab/supermimic/symmetry.py`)**: Once a biped initiates a step, its state enters a 2D walking limit cycle in `(q_L - q_R, dq_L - dq_R, c_L - c_R)` phase space that **never intersects the origin** (`r_asym(o) >= 0.70` throughout both single-support swing and double-support stride transfer, whereas static double-support standing has `r_asym(o) <= 0.12`). By gating `L_sym` via `g(o) = clip((r_asym(o) - 0.18) / (0.45 - 0.18), 0, 1)` and ramping `w_sym` from `0 -> 1.5` after the 7D skill-unzipping phase (`iter 1350..1800`), the single exported `TeacherActorMLP` freely breaks symmetry to step out from rest while matching left-lead and right-lead strides to `sym_mse = 8.8e-4`, cutting straight-walk lateral drift by **`72-78%`** (`8.1 cm` neutral / `11.3 cm` AMASS).

---

## 4. Methodological Scope & White-Box Training Disclosure

- **Inference Parity vs White-Box Digital-Twin Training**: At deployment/inference time, `SM_3000_Sym` uses the exact same 60D observation vector and `180,367`-parameter MLP as PPO (zero privileged inputs, zero inference wrapper). During **training**, Reference-Free SuperMimic operates as a **White-Box Differentiable-Reward MBRL** teacher trainer — utilizing exact USD forward kinematics, analytic task reward expressions (`28/31` terms analytic, `3/31` impact penalties learned), privileged simulator Domain Randomization context `xi` (`4D`), an 88D privileged Critic, and 24-step exogenous arm disturbance lookahead in imagination.
- **Wall-Clock Compute Trade-Off**: At equal `294.9M` environment transitions, `SuperMimic` takes `~3.8-4.0x` more wall-clock time on a single RTX 4090 (`201-212 min` vs `52.5 min` for PPO), though at equal wall-clock time (`52 min`, `SM iter ~760`), `SuperMimic` already achieves higher training return (`~84-85`) than `PPO iter 3000` (`74.44`).

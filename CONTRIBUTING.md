# Contributing to SuperMimic-Decoupled-WBC

Thank you for your interest in extending **Reference-Free Differentiable World Model RL** to new robotic platforms and loco-manipulation tasks!

## Core Engineering & Scientific Principles

1. **Zero Upstream Baseline Tampering**:
   - Do **not** silently alter physics parameters, PD gains, observation dimensions, Domain Randomization ranges, or PPO baseline reward weights in `legged_lab/envs/` or `legged_lab/mdp/` when benchmarking algorithmic improvements.
   - All `SuperMimic` extensions live modularly under `legged_lab/supermimic/` and `legged_lab/scripts/`.
   - You can verify zero upstream environment drift at any time via:
     ```bash
     git diff upstream-main -- legged_lab/envs legged_lab/mdp legged_lab/assets legged_lab/scripts/train_teacher.py
     ```

2. **Exact Kinematic & Differentiable Reward Parity**:
   - Whenever adding a new robot URDF/USD or reward term, run the PhysX parity verification suite and ensure forward-kinematics error stays `< 1e-5 m` and analytic reward error stays `< 1e-6`:
     ```bash
     bash scripts/run_diagnostics.sh
     ```

3. **No Pretrained Weight Leakage & Compact Checkpointing**:
   - All benchmarked policies must be trained **from scratch** (`orthogonal_` random initialization) without warm-starting from PPO or MoCap tracking checkpoints.
   - Do not commit large intermediate checkpoints or `logs/` artifacts; `SuperMimicRunner` automatically prunes intermediate checkpoints (`keep_only_best_and_last=True`) and strips optimizer momentum tensors upon completion.

4. **Code Style**:
   - Follow PEP 8 / Ruff formatting (`line-length = 120`, configured in `pyproject.toml`):
     ```bash
     ruff check legged_lab/supermimic legged_lab/scripts/train_supermimic.py legged_lab/scripts/eval_compare.py legged_lab/scripts/run_diagnostics.py
     ```

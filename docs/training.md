# Training & Play Guide

> **See also:** [README](../README.md) · [Deployment](deployment.md)

Detailed guide for training the teacher and student policies and playing them in simulation. The
[Quick Start](../README.md#quick-start) wraps the most common commands in
`scripts/*.sh`; this document covers the options, overrides, and advanced workflows
behind them.

All commands assume the `isaac_wbc` conda env is active (`conda activate isaac_wbc`)
and are run from the repository root.

## Contents
- [Quick verification](#quick-verification)
- [Train teacher policy](#train-teacher-policy)
- [Play the teacher policy](#play-the-teacher-policy)
- [Train student policy](#train-student-policy)
- [Play the student policy](#play-the-student-policy)
- [Simulation keyboard controls](#simulation-keyboard-controls)
- [Training configuration](#training-configuration)
- [Reference: observation space](#reference)

---

## Quick verification

After installation, use small environment counts for a first smoke test:

```bash
python legged_lab/scripts/train_teacher.py \
    --task=g1_flat \
    --logger=tensorboard \
    --num_envs=8 \
    --max_iterations=5 \
    --headless
```

This five-iteration command only verifies that training starts successfully; all
full teacher-training examples below use the 250000-iteration checkpoint target.

For a trained student checkpoint, keep the first simulation check small:

```bash
python legged_lab/scripts/play_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --student_checkpoint=<path/to/student_policy_jit.pt> \
    --use_jit \
    --num_envs=8 \
    --max_steps=100 \
    --headless
```

---

## Train teacher policy

Train the teacher policy with the default training configuration (wrapped by
`scripts/train_teacher.sh`):

```bash
python legged_lab/scripts/train_teacher.py \
    --task=g1_flat \
    --run_name=htd_wbc \
    --logger=wandb \
    --num_envs=12288 \
    --max_iterations=250000 \
    --save_interval=2500 \
    --headless
```

### Overriding control gains and curriculum

Override any config field with `--set KEY=VALUE` (repeatable; see
[Training Configuration](#training-configuration) for how the config layers fit together). `KEY` is a
dotted attribute path; it targets the environment config by default, or the agent/PPO
config when prefixed with `agent.`. `VALUE` is parsed as Python, so numbers, tuples, and
dicts work (e.g. `--set agent.algorithm.learning_rate=1.0e-3`). For example, to set
custom waist PD gains and a body-pose curriculum:

```bash
python legged_lab/scripts/train_teacher.py \
    --task=g1_flat \
    --run_name=htd_wbc_kpw100_kdw4 \
    --logger=wandb \
    --num_envs=12288 \
    --max_iterations=250000 \
    --save_interval=2500 \
    --headless \
    --set "scene.robot.actuators.legs.stiffness={'.*_hip_yaw_joint':100.0,'.*_hip_roll_joint':100.0,'.*_hip_pitch_joint':100.0,'.*_knee_joint':150.0,'.*waist.*':100.0}" \
    --set "scene.robot.actuators.legs.damping={'.*_hip_yaw_joint':2.0,'.*_hip_roll_joint':2.0,'.*_hip_pitch_joint':2.0,'.*_knee_joint':4.0,'.*waist.*':4.0}" \
    --set "commands.curriculum.enable=True" \
    --set "commands.curriculum.body_yaw.start_iter=20000" \
    --set "commands.curriculum.body_yaw.end_iter=60000"
```

Notes:
- **Put quotes around each `--set` value.** The gain settings contain `*` characters,
  and without quotes your terminal can mess them up. Quotes keep them intact.
- **Pass the full gain dict** — assignment replaces it wholesale, so any omitted joint
  loses its gain. Start from the `G1_CFG_HTD` baseline and edit the entries you want.
- Curriculum: `start_iter`/`end_iter` set the schedule per axis (the range bounds come
  from `g1_config.py`); use `commands.curriculum.enable=False` to disable it.

### Resuming training

To resume training from a checkpoint, point `--load_run` at the run folder under
`logs/g1_flat/` and `--checkpoint` at the checkpoint file to load:

```bash
python legged_lab/scripts/train_teacher.py \
    --task=g1_flat \
    --logger=wandb \
    --num_envs=12288 \
    --max_iterations=250000 \
    --resume \
    --load_run=<run_folder> \
    --checkpoint=<model_checkpoint.pt> \
    --headless
```

`--max_iterations` is added to the resumed checkpoint's iteration count (e.g. a
checkpoint at 100000 trained with `--max_iterations=50000` runs to 150000). For
preemptible cluster jobs, prefer `--auto_resume` instead, which auto-discovers the
latest checkpoint and treats `--max_iterations` as an absolute target.
`--auto_resume` requires `--run_name` (the run folder is derived from it so a
requeued job lands in the same place).

> **Curriculum on resume:** the command curriculum continues from the checkpoint's
> iteration rather than restarting at 0 (the resumed iteration is used to offset the
> schedule). The curriculum bounds (`start_iter`/`end_iter`/ranges) are read fresh
> from the config at launch, not stored in the checkpoint, so edits to them take
> effect on resume, evaluated at the resumed absolute iteration. This applies to
> both `--resume` and `--auto_resume`.

---

## Play the teacher policy

Run the trained teacher policy in simulation (wrapped by `scripts/play_teacher.sh`).
Point `--load_run` at the run folder under `logs/g1_flat/` and `--checkpoint` at the
checkpoint to load:

```bash
python legged_lab/scripts/play_teacher.py \
    --task=g1_flat \
    --num_envs=1 \
    --load_run=<run_folder> \
    --checkpoint=<model_checkpoint.pt>
```

> **Note:** the play script automatically loads the actuator PD gains (kp/kd) the
> checkpoint was trained with from `logs/g1_flat/<run_folder>/params/env.yaml`, so a
> checkpoint is always played with its own gains (e.g. when different sweep runs
> used different waist/leg stiffness). If that file is missing it falls back to the
> default config gains and prints a warning.

---

## Train student policy

Train a student policy using imitation learning from the teacher. Point
`--load_run`/`--checkpoint` at the teacher run and checkpoint to distill from; the
student is distilled in an environment that automatically matches that checkpoint's
actuator gains (kp/kd), loaded from `logs/g1_flat/<run_folder>/params/env.yaml`.

The two-stage workflow is Behavior Cloning (BC) followed by DAgger. `--load_run`/`--checkpoint`
always point at the **teacher** run/checkpoint (they load the teacher policy and its gains);
DAgger additionally resumes the **student** checkpoint from BC via `--resume_checkpoint`.

### One-command two-stage (BC → DAgger)

`--bc_steps` runs the whole pipeline in a single command: Behavior Cloning for the first
`--bc_steps` steps, then DAgger automatically up to `--max_steps` (`--IL_type` is ignored in
this mode). Add `--auto_resume` to make it survive preemption (it continues the latest student
checkpoint, recovering the BC/DAgger phase from the step count). This is what
`scripts/train_student.sh` runs.

```bash
python legged_lab/scripts/train_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --checkpoint=<model_checkpoint.pt> \
    --num_envs=4096 \
    --bc_steps=250000 \
    --max_steps=600000 \
    --student_lr=5e-4 \
    --save_interval=5000 \
    --keep_checkpoints=3 \
    --logger=wandb \
    --headless
```

### Stage 1 — Behavior Cloning (BC)

Clone the teacher's actions for 250000 steps:

```bash
python legged_lab/scripts/train_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --checkpoint=<model_checkpoint.pt> \
    --num_envs=4096 \
    --max_steps=250000 \
    --student_lr=5e-4 \
    --save_interval=5000 \
    --keep_checkpoints=3 \
    --IL_type=bc \
    --logger=wandb \
    --headless
```

### Stage 2 — DAgger (resume from BC)

Resume the BC student and continue with DAgger. `--max_steps` is an **absolute** target,
so set it to `600000` to add 350000 DAgger steps on top of the 250000 BC steps:

```bash
python legged_lab/scripts/train_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --checkpoint=<model_checkpoint.pt> \
    --resume_checkpoint=logs/g1_flat/<run_folder>/student_checkpoints/student_step_250000.pt \
    --num_envs=4096 \
    --max_steps=600000 \
    --student_lr=5e-4 \
    --save_interval=5000 \
    --keep_checkpoints=3 \
    --IL_type=dagger \
    --logger=wandb \
    --headless
```

`--IL_type=bc` rolls out the **teacher's** actions in the environment (stable, off-policy
cloning); `--IL_type=dagger` rolls out the **student's** own actions (on-policy, corrects
compounding errors). To run DAgger from scratch instead, omit `--resume_checkpoint` and use
`--max_steps=250000`.

### Student training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--student_lr` | 5e-4 | Learning rate for student policy |
| `--max_steps` | 250000 | Maximum training steps (absolute target) |
| `--bc_steps` | 0 | If >0, run the two-stage pipeline: BC for the first N steps, then DAgger up to `--max_steps` (overrides `--IL_type`) |
| `--save_interval` | 2000 | Save checkpoint every N steps |
| `--keep_checkpoints` | 3 | Number of checkpoints to keep |
| `--IL_type` | bc | Imitation learning type: `bc` or `dagger` |
| `--resume_checkpoint` | None | Path to student checkpoint to resume from |
| `--auto_resume` | off | Auto-discover and resume the latest student checkpoint in the run's `student_checkpoints/` |
| `--wandb_run_version` | v1 | W&B run name suffix; bump (e.g. v1 → v2) to start a fresh W&B run for the same teacher run |

---

## Play the student policy

Pass `--load_run=<run_folder>` (the teacher run the student was distilled from, where its
`student_checkpoints/` live) so the play run applies the same actuator gains (kp/kd) the student
was trained under. Without it, the default config gains are used, which may not match the
student's teacher and will change the closed-loop behavior. `scripts/play_student.sh` wraps
the JIT form below.

```bash
python legged_lab/scripts/play_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --student_checkpoint=logs/g1_flat/<run_folder>/student_checkpoints/<student_step.pt> \
    --num_envs=1 \
    --max_steps=20000
```

Add `--log_obs` to write a detailed per-step observation history log next to the checkpoint (disabled by default).

For the JIT model:

```bash
python legged_lab/scripts/play_student.py \
    --task=g1_flat \
    --load_run=<run_folder> \
    --student_checkpoint=logs/g1_flat/<run_folder>/student_checkpoints/student_policy_jit.pt \
    --use_jit \
    --num_envs=1 \
    --max_steps=20000
```

---

## Simulation keyboard controls

When running `play_teacher.py` or `play_student.py` without `--headless`, you can control the robot using the keyboard.

#### Movement Commands
| Key | Function | Step Size | Description |
|-----|----------|-----------|-------------|
| **W** | Move forward | +0.1 | Increase cmd_x (forward velocity) |
| **S** | Move backward | -0.1 | Decrease cmd_x (backward velocity) |
| **A** | Move left | +0.1 | Increase cmd_y (left velocity) |
| **D** | Move right | -0.1 | Decrease cmd_y (right velocity) |
| **Q** | Rotate left | +0.1 | Increase cmd_z (counter-clockwise) |
| **E** | Rotate right | -0.1 | Decrease cmd_z (clockwise) |

#### Height Control
| Key | Function | Step Size | Description |
|-----|----------|-----------|-------------|
| **H** | Increase height | +0.01m | Raise body height |
| **J** | Decrease height | -0.01m | Lower body height |

#### Body Orientation
| Key | Function | Step Size | Description |
|-----|----------|-----------|-------------|
| **Z** | Roll right | +0.1 rad | Increase body roll (~5.7°) |
| **X** | Roll left | -0.1 rad | Decrease body roll (~5.7°) |
| **C** | Pitch down | +0.1 rad | Increase body pitch (~5.7°) |
| **V** | Pitch up | -0.1 rad | Decrease body pitch (~5.7°) |
| **B** | Yaw left | +0.1 rad | Increase body yaw (~5.7°) |
| **N** | Yaw right | -0.1 rad | Decrease body yaw (~5.7°) |

#### Reset
| Key | Function | Description |
|-----|----------|-------------|
| **R** | Reset robot | Reset all commands to defaults (height=0.72m, all others=0.0) |

**Note**: Commands are cumulative — each key press adds/subtracts the step size. Press **R** to reset all commands. Add `--continuous_torso` to instead ramp the torso commands (H/J/Z/X/C/V/B/N) continuously while a key is held. For physical-robot controller mappings, see the [Deployment Guide](deployment.md).

---

## Training configuration

Training is driven by one env config and one agent (PPO) config. They are built by
layering, each layer overriding the previous:

1. **Base defaults** — `legged_lab/envs/base/base_config.py` (reusable building blocks)
   and `base_env_config.py` (`BaseEnvCfg` + `BaseAgentCfg`).
2. **Robot asset** — `legged_lab/assets/unitree/unitree.py` (USD path, actuator kp/kd).
3. **Task values** — `legged_lab/envs/g1/g1_config.py` sets the G1-specific numbers
   (command ranges, curriculum, reward weights). This is the main file to edit.
4. **CLI flags** — `--num_envs`, `--seed`, `--max_iterations`, etc.
5. **`--set` overrides** — highest priority; change any field without editing files
   (see [Overriding control gains and curriculum](#overriding-control-gains-and-curriculum)).

The fully resolved config is saved to `logs/<exp>/<run>/params/{env,agent}.yaml` for
each run. Quick guide: change a G1 value → `g1_config.py`; a gain → `unitree.py`; a
one-off experiment → `--set` on the command line. Deployment configuration is described
in the [Deployment Guide](deployment.md#deployment-configuration).

Logs and checkpoints are written under `logs/<exp>/<run>/` (student logs under
`student_tensorboard/`); view them with `tensorboard --logdir=logs/g1_flat` or use
`--logger=wandb`.

---

## Reference

Command ranges are set in the env config (`legged_lab/envs/g1/g1_config.py`); the
joint-to-motor mapping, PD gains, and default positions used at deploy time live in
`deploy/configs/g1_student_htd.yaml`. Keep these values aligned with the trained policy;
see [Deployment Configuration](deployment.md#deployment-configuration).

### Observation space

**Teacher observations (60-dim)**
- Angular velocity (3)
- Projected gravity (3)
- Commands (7): linear_x, linear_y, angular_z, height, body_roll, body_pitch, body_yaw
- Joint positions (15)
- Joint velocities (15)
- Last actions (15)
- Feet contact (2)

**Student observations**
- **Base observations (58-dim)**: Same as teacher but without feet contact
- **History buffer**: 2 timesteps × 58 dimensions = 116-dim input to MLP
- **Network**: MLP with hidden layers [512, 256, 128]

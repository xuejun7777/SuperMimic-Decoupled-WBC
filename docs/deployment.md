# Deployment Guide for Physical G1 Robot

> **See also:** [README](../README.md) · [Training & Play](training.md)

Deploy trained student policies on physical Unitree G1 robots.

---

## Installation

Install the required Unitree SDK2 package for robot communication:

```bash
conda activate isaac_wbc
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

---

## Network Setup

Find network interface name:
```bash
ifconfig
```

Example output showing interface `eno1`:
```
eno1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.123.162  netmask 255.255.255.0
```

Configure static IP in network settings:
- IP: `192.168.123.xxx` (any value except .10)
- Netmask: `255.255.255.0`
- Gateway: (leave empty)

---

## Run the Policy on the Robot

The bundled policy and default config are ready to run. If you are using your own
policy, complete [Student Policy Deployment](#student-policy-deployment) first.

### Preflight Checklist

Before running deployment on hardware, verify:

- The robot is a supported Unitree G1 variant for the selected config.
- The policy file referenced by `policy_path` exists relative to the `deploy/` directory.
- The selected config matches the policy that was exported from training.
- The robot-facing network interface is connected and `ping 192.168.123.10` succeeds.
- The robot has enough clear space and a person is ready to intervene.
- The robot's hardware emergency stop is within reach and the operator knows how to use it.

### 1. Robot Setup

- Connect the robot to the computer via Ethernet.
- Press ``L2 + R2`` on the controller to enter debug mode.

### 2. Start Program

From the repository root:

```bash
cd deploy
python deploy_student_htd.py --config_path configs/g1_student_htd.yaml
```

Expected output:

```text
Successfully connected to the robot.
Enter zero torque state.
Waiting for the start signal to move to default pos...
```

### 3. Move to Default Position

Press ``Start`` on the controller. The robot moves to its default standing position
over two seconds.

Expected output:

```text
Moving to default pos.
Enter default pos state.
Waiting for the Button A signal to Start Control...
```

### 4. Start Control

Press ``A`` on the controller.

Expected output: `Start Control!`

### 5. Control Robot

**Joysticks** (continuous):
- Left stick up/down: Adjust torso pitch
- Left stick left/right: Adjust torso roll
- Right stick left/right: Command the robot's yaw rate

**Buttons** (one step per press):
- `Y` / `A`: Raise / lower the commanded body height
- `X` / `B`: Rotate the commanded torso yaw left / right
- D-pad `↑` / `↓`: Increase forward / backward velocity
- D-pad `←` / `→`: Move sideways left / right

Each button press updates a stored command; the command remains active after the
button is released.

**Software stop**: Press ``Select`` to stop policy inference and command publishing.
The program sends a damping command to all motors, waits 0.2 seconds, and exits. This
does not replace the robot's hardware emergency stop.

---

## Student Policy Deployment

### Policy File Location

An example JIT policy is bundled at `deploy/policy/g1_student/student_policy_jit.pt`, and the
default `g1_student_htd.yaml` already points at it (`policy_path`), so you can deploy it
directly without training first.

To deploy your own policy instead: student training exports a JIT policy to the run's
checkpoint directory,

```
logs/g1_flat/<run_name>/student_checkpoints/student_policy_jit.pt
```

which you copy into the deploy policy folder from the repository root:

```bash
cp logs/g1_flat/<run_name>/student_checkpoints/student_policy_jit.pt \
   deploy/policy/g1_student/student_policy_jit.pt
```

Before hardware deployment, confirm that the policy file, config file, robot model, and training/export path all match.

### Deployment Configuration

Deployment configs live under `deploy/configs/` and are selected at runtime with
`--config_path`. Paths such as `policy_path` are interpreted relative to the `deploy/`
directory when following this guide.

`configs/g1_student_htd.yaml` is the maintained config for the G1 student policy. It
controls 15 lower-body and waist joints (`num_actions: 15`) with a two-step observation
history (`history_length: 2`, `num_obs: 58`).

The config must match the policy it accompanies:

- `policy_path` points to the JIT policy.
- `joint2motor_idx`, `kps`, `kds`, and `default_joint_pos` each contain
  `num_actions` entries and match the training setup.
- `history_length` and `num_obs` match the student network's observation layout.
- `command_scale` matches the number of entries in `command_range`; deployment clips
  controller input to those ranges.
- `waist_pitch_limits` adds a hardware safety clip for the waist pitch joint.

Each target is computed as
`q_target = default_pos + action × action_scale`. Arm and upper-torso motors not
controlled by the policy are held at their configured default positions.

---

## Acknowledgement

This deployment code builds on [LeggedLab](https://github.com/Hellod035/LeggedLab).

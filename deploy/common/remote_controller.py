# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]
        
        
class RemoteControllerStudent:
    """
    Extended remote controller for student policy with joystick and button-controlled commands.

    Joystick mapping (continuous control):
    - ly: body_pitch (pitch angle, -1=down, +1=up)
    - lx: body_roll (roll angle, -1=right, +1=left)
    - rx: ang_vel_z (yaw rotation velocity, preserved from original)

    Button mapping (discrete step control):
    - Y: Increase height
    - A: Decrease height
    - X: Turn left (body_yaw +, counter-clockwise)
    - B: Turn right (body_yaw -, clockwise)
    - Up: Move forward (lin_vel_x +)
    - Down: Move backward (lin_vel_x -)
    - Left: Move left (lin_vel_y +)
    - Right: Move right (lin_vel_y -)
    """

    def __init__(self, height_step=0.005, roll_step=0.1, pitch_step=0.15, yaw_step=0.1, vel_step=0.1):
        """
        Args:
            height_step: Height change per button press (meters), default 0.005m (0.5cm)
            roll_step: Body roll change per button press (radians), default 0.1 rad (~5.7°)
            pitch_step: Body pitch change per button press (radians), default 0.15 rad (~8.6°)
            yaw_step: Body yaw change per button press (radians), default 0.1 rad (~5.7°)
            vel_step: Linear velocity change per button press (m/s), default 0.1 m/s

        Training command ranges for reference (legged_lab/envs/g1/g1_config.py):
            height: (0.35, 0.8) m
            body_roll: (-0.7, 0.7) rad
            body_pitch: (-0.52, 1.57) rad
            body_yaw: (-1.57, 1.57) rad
            lin_vel_x: (-0.5, 0.5) m/s
            lin_vel_y: (-0.5, 0.5) m/s

        This class does not clip; the deployment controller clips the assembled
        command to the `command_range` of the active deploy config
        (deploy/configs/*.yaml) before it reaches the policy.
        """
        self.base_controller = RemoteController()

        # Button states for edge detection (detect button press, not hold)
        self.prev_button = [0] * 16
        # Startup also uses controller buttons. Ignore discrete command presses until
        # the deployment controller has consumed the A press that starts policy control.
        self.discrete_commands_enabled = False

        # Discrete command states - initialized to match training defaults
        self.height = 0.70          # Training default
        self.body_roll = 0.0       # Training default (will be overridden by lx joystick)
        self.body_pitch = 0.0      # Training default (will be overridden by ly joystick)
        self.body_yaw = 0.0        # Training default
        self.lin_vel_x = 0.0       # Forward/backward velocity (controlled by Up/Down buttons)
        self.lin_vel_y = 0.0       # Left/right velocity (controlled by Left/Right buttons)

        # Step sizes for each command (different for each dimension)
        self.height_step = height_step
        self.roll_step = roll_step
        self.pitch_step = pitch_step
        self.yaw_step = yaw_step
        self.vel_step = vel_step

        # Joystick mapping ranges
        # Pitch range for ly joystick (ly: -1 to 1 → pitch: pitch_min to pitch_max)
        self.pitch_min = -0.52
        self.pitch_max = 1.0
        # Roll range for lx joystick (lx: -1 to 1 → roll: roll_min to roll_max)
        self.roll_min = -0.7
        self.roll_max = 0.7

    def set(self, data):
        """Update controller state from wireless remote data."""
        # Update base controller
        self.base_controller.set(data)

        # Map joysticks to body orientation
        # ly joystick → body_pitch (ly: -1 to 1 → pitch: pitch_min to pitch_max)
        ly = self.base_controller.ly
        self.body_pitch = (ly + 1.0) / 2.0 * (self.pitch_max - self.pitch_min) + self.pitch_min

        # lx joystick → body_roll (lx: -1 to 1 → roll: roll_min to roll_max)
        lx = self.base_controller.lx
        self.body_roll = (lx + 1.0) / 2.0 * (self.roll_max - self.roll_min) + self.roll_min

        # Detect button presses (rising edge: 0 -> 1) only after startup is complete.
        if self.discrete_commands_enabled:
            for i in range(16):
                if self.base_controller.button[i] == 1 and self.prev_button[i] == 0:
                    self._handle_button_press(i)

        # Update previous button state
        self.prev_button = self.base_controller.button.copy()

    def enable_discrete_commands(self):
        """Enable command buttons after the startup button press has been consumed."""
        self.prev_button = self.base_controller.button.copy()
        self.discrete_commands_enabled = True

    def _handle_button_press(self, button_idx):
        """Handle button press events for discrete commands."""
        # Y button: increase height
        if button_idx == KeyMap.Y:
            self.height += self.height_step
        # A button: decrease height
        elif button_idx == KeyMap.A:
            self.height -= self.height_step
        # X button: turn left (body_yaw +) - counter-clockwise is positive (right-hand rule)
        elif button_idx == KeyMap.X:
            self.body_yaw += self.yaw_step
        # B button: turn right (body_yaw -) - clockwise is negative
        elif button_idx == KeyMap.B:
            self.body_yaw -= self.yaw_step
        # Up: move forward (lin_vel_x +)
        elif button_idx == KeyMap.up:
            self.lin_vel_x += self.vel_step
        # Down: move backward (lin_vel_x -)
        elif button_idx == KeyMap.down:
            self.lin_vel_x -= self.vel_step
        # Left: move left (lin_vel_y +)
        elif button_idx == KeyMap.left:
            self.lin_vel_y += self.vel_step
        # Right: move right (lin_vel_y -)
        elif button_idx == KeyMap.right:
            self.lin_vel_y -= self.vel_step

    def reset_discrete_commands(self):
        """Reset all discrete commands to training defaults."""
        self.height = 0.70       # Training default
        self.body_roll = 0.0     # Training default (will be overridden by lx)
        self.body_pitch = 0.0    # Training default (will be overridden by ly)
        self.body_yaw = 0.0      # Training default
        self.lin_vel_x = 0.0     # Training default
        self.lin_vel_y = 0.0     # Training default

    # Properties to access base controller's joystick values
    @property
    def lx(self):
        return self.base_controller.lx

    @property
    def ly(self):
        return self.base_controller.ly

    @property
    def rx(self):
        return self.base_controller.rx

    @property
    def ry(self):
        return self.base_controller.ry

    @property
    def button(self):
        return self.base_controller.button

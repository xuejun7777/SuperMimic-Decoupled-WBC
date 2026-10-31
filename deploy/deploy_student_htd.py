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

import sys
import time
from threading import Lock

import numpy as np
import torch
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteControllerStudent
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config import Config


class StudentController:
    def __init__(self, config: Config) -> None:

        ChannelFactoryInitialize(0)

        # self.first_run = True
        self.config = config
        self.remote_controller = RemoteControllerStudent()
        self.device = torch.device('cpu')
        self.policy = torch.jit.load(config.policy_path, map_location=torch.device('cpu')).eval()
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)  # 50Hz
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)
        self.current_obs_history = np.zeros((config.history_length, config.num_obs), dtype=np.float32)

        self.waist_pitch_limits = np.array(config.waist_pitch_limits, dtype=np.float32)

        # Command clipping ranges (7-dim for student)
        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
                self.config.command_range["height"][0],
                self.config.command_range["body_roll"][0],
                self.config.command_range["body_pitch"][0],
                self.config.command_range["body_yaw"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
                self.config.command_range["height"][1],
                self.config.command_range["body_roll"][1],
                self.config.command_range["body_pitch"][1],
                self.config.command_range["body_yaw"][1],
            ],
            dtype=np.float32,
        )

        for _ in range(50):
            with torch.inference_mode():
                obs = self.current_obs_history.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs).to(self.device))

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.wait_for_low_state()

        self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.publish_thread.Start()
        self.wait_for_start()

        self.move_to_default_pos()

        self.wait_for_control()
        self.remote_controller.enable_discrete_commands()

        print("Start Control!")
        self.run_thread.Start()


    def LowStateHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def wait_for_start(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint2motor_idx
        dof_size = len(dof_idx)

        arm_dof_idx = self.config.arm_joint2motor_idx
        arm_dof_size = len(arm_dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        arm_init_dof_pos = np.zeros(arm_dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        for i in range(arm_dof_size):
            arm_init_dof_pos[i] = self.low_state.motor_state[arm_dof_idx[i]].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                for k in range(arm_dof_size):
                    motor_idx = arm_dof_idx[k]
                    target_pos = self.config.default_arm_joint_pos[k]
                    self.low_cmd.motor_cmd[motor_idx].q = arm_init_dof_pos[k] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_kps[k]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_kds[k]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            time.sleep(self.config.control_dt)

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def run(self):
        for i in range(len(self.config.joint2motor_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        # Read IMU data
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel
            )

        # Compute gravity orientation
        gravity_orientation = get_gravity_orientation(quat)
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        command = np.array(
            [
                self.remote_controller.lin_vel_x,       # lin_vel_x (Up/Down buttons)
                self.remote_controller.lin_vel_y,       # lin_vel_y (Left/Right buttons)
                -self.remote_controller.rx,             # ang_vel_z (rx joystick, yaw rotation)
                self.remote_controller.height,          # height (Y/A buttons)
                self.remote_controller.body_roll,       # body_roll (lx joystick)
                self.remote_controller.body_pitch,      # body_pitch (ly joystick)
                self.remote_controller.body_yaw,        # body_yaw (X/B buttons)
            ],
            dtype=np.float32,
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)

        # Assemble 58-dim observation for student
        # [ang_vel(3), gravity(3), command(7), joint_pos(15), joint_vel(15), action(15)]
        num_actions = self.config.num_actions  # 15
        self.current_obs[:3] = ang_vel                                      # [0:3]
        self.current_obs[3:6] = gravity_orientation                         # [3:6]
        self.current_obs[6:13] = command                                    # [6:13]
        self.current_obs[13:13 + num_actions] = joint_pos                   # [13:28]
        self.current_obs[13 + num_actions:13 + num_actions * 2] = joint_vel # [28:43]
        self.current_obs[13 + num_actions * 2:13 + num_actions * 3] = self.action  # [43:58]

        # Update observation history buffer (circular buffer)
        # Shift left and add new observation (same as training)
        self.current_obs_history[:-1] = self.current_obs_history[1:].copy()  # Shift left
        self.current_obs_history[-1] = self.current_obs  # Add new obs at the end

        # Run student policy inference
        obs = self.current_obs_history.reshape(1, -1).astype(np.float32)  # [1, 116] (history_length=2: 2*58=116)
        self.action = self.policy(torch.from_numpy(obs).to(self.device).clip(-100, 100)).clip(-100, 100).detach().numpy().squeeze()

        # Apply student actions to the 15 controlled joints
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale


        waist_pitch_idx = 14  
        target_dof_pos[waist_pitch_idx] = np.clip(
            target_dof_pos[waist_pitch_idx],
            self.config.waist_pitch_limits[0],  # lower limit
            self.config.waist_pitch_limits[1]   # upper limit
        )

        with self.cmd_lock:
            for i in range(len(self.config.joint2motor_idx)):
                self.low_cmd.motor_cmd[self.config.joint2motor_idx[i]].q = target_dof_pos[i]
            for i in range(len(self.config.arm_joint2motor_idx)):
                self.low_cmd.motor_cmd[self.config.arm_joint2motor_idx[i]].q = self.config.default_arm_joint_pos[i]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/g1_student_htd.yaml", help="student configuration file")
    args = parser.parse_args()

    config = Config(args.config_path)
    controller = StudentController(config)

    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select Button detected, Exit!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("Exit")

"""
Logging file for inputs and measurements according to data-driven-dynamics naming scheme;
All quantities that are not mentioned are assumed to be zero

| Variable name | Quantity                         		 | Min  | Max | Unit           |
|---------------|----------------------------------------|------|-----|----------------|
| u4   			| throttle input  						 | -1 	| 1   | position (m)   |
| u7   			| elevator input    					 | -1 	| 1   | position (m)   |
| vx   			| horizontal velocity of the plane    	 | -Inf | Inf | velocity (m/s) |
| vz   			| vertical   velocity of the plane    	 | -Inf | Inf | velocity (m/s) |
| theta   		| pitch of the plane                  	 | -PI  | PI  | rad            |
| ang_vel_y   	| pitch rate of the plane              	 | -Inf | Inf | rad/s          |
| acc_b_x  	 	| horizontal acceleration in body frame  | -Inf | Inf | m/s^2          |
| acc_b_z   	| vertical acceleration in body frame    | -Inf | Inf | m/s^2          |
| ang_acc_b_y   | angular acceleration in body frame     | -Inf | Inf | rad/s^2        |
"""

import numpy as np
import pandas as pd
import csv


class Logger:
    def __init__(self, log_path):
        logfile = open(log_path, 'w')
        self.logger = csv.writer(logfile, delimiter=',')
        self.logger.writerow(['idx', 'timestamp', 'u2', 'u4', 'u5', 'u6', 'u7', 'vx', 'vy', 'vz',  'q0', 'q1', 'q2', 'q3',
                              'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'acc_b_x', 'acc_b_y', 'acc_b_z', 'ang_acc_b_x', 'ang_acc_b_y', 'ang_acc_b_z'])

    def log_data(self, idx, env, action, accelerations, time):
        """
        Log data to csv file

        Args:
        :param idx: index of the current step
        :param env: environment object
        :param action: action taken by the agent
        :param accelerations: accelerations of the plane
        :param time: time of the current step

        Returns:
        :return: None
        """

        # get the linear accelerations in NED world frame
        acc_w_x = accelerations['linX']
        acc_w_z = accelerations['linZ']

        # multiply the accelerations with the transormation matrix to body frame (aligned with FRL)
        acc_b_x, acc_b_z = self.vector_rotation(
            np.array([acc_w_x, acc_w_z]), env.state[4])
        acc_ang_b_y = accelerations['angY']

        q0, q1, q2, q3 = Logger.euler_to_quaternion(0, env.state[4], 0)

        # compute gravity in the body frame to subtract it from the accelerations (sensor fusion - consistency with ulogs)
        gravity = np.array([0, -9.81])
        gravity_b = self.vector_rotation(gravity, - env.state[4])

        self.logger.writerow([idx, time, 0, action[0], 0, 0, action[1], env.state[2], 0, - env.state[3], q0,
                             q1, q2, q3, 0, env.state[5], 0, acc_b_x + gravity_b[0], 0, acc_b_z + gravity_b[1], 0, acc_ang_b_y, 0])

    def vector_rotation(self, vector: np.array(2), angle: float):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.matmul(rotation_matrix, vector)

    def euler_to_quaternion(roll: float, pitch: float, yaw: float):
        """
        Convert Euler angles to quaternion

        Args:
        :param roll: roll angle in radians
        :param pitch: pitch angle in radians
        :param yaw: yaw angle in radians

        Returns:
        :return: quaternion in the form [w, x, y, z]
        """

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def quaternion_to_euler(w: float, x: float, y: float, z: float):
        """
        Convert quaternion to Euler angles

        Args:
        :param w: w component of the quaternion
        :param x: x component of the quaternion
        :param y: y component of the quaternion
        :param z: z component of the quaternion

        Returns:
        :return: Euler angles in the form [roll, pitch, yaw]
        """

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def get_data(path):
        data = pd.read_csv(path)
        return data

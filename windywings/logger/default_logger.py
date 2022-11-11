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
        self.logger.writerow(['idx', 'timestamp', 'u4', 'u7', 'vx', 'vz',  'theta',
                              'ang_vel_y', 'acc_b_x', 'acc_b_z', 'ang_acc_b_y'])

    def log_data(self, idx, env, action, accelerations, time):
        # get the linear accelerations in NED world frame
        acc_w_x = accelerations['linX']
        acc_w_z = accelerations['linZ']

        # multiply the accelerations with the transormation matrix to body frame (aligned with FRL)
        acc_b_x, acc_b_z = self.vector_rotation(
            np.array([acc_w_x, acc_w_z]), env.state[4])
        acc_ang_b_y = accelerations['angY']

        self.logger.writerow([idx, time, action[0], action[1], env.state[2], - env.state[3], env.state[4],
                              env.state[5], acc_b_x, acc_b_z, acc_ang_b_y])

    def vector_rotation(self, vector: np.array(2), angle: float):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.matmul(rotation_matrix, vector)

    def get_data(path):
        data = pd.read_csv(path)
        return data

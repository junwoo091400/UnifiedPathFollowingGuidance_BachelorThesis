"""
Logging file for inputs and measurements according to data-driven-dynamics naming scheme;
All quantities that are not mentioned are assumed to be zero

| Variable name | Quantity                         		 | Min  | Max | Unit           |
|---------------|----------------------------------------|------|-----|----------------|
| U4   			| throttle input  						 | -1 	| 1   | position (m)   |
| U7   			| elevator input    					 | -1 	| 1   | position (m)   |
| Vx   			| horizontal velocity of the plane    	 | -Inf | Inf | velocity (m/s) |
| Vz   			| vertical   velocity of the plane    	 | -Inf | Inf | velocity (m/s) |
| Gamma   		| pitch of the plane                  	 | -PI  | PI  | rad            |
| Ang_vel_y   	| pitch rate of the plane              	 | -Inf | Inf | rad/s          |
| Acc_b_x  	 	| horizontal acceleration in body frame  | -Inf | Inf | m/s^2          |
| Acc_b_z   	| vertical acceleration in body frame    | -Inf | Inf | m/s^2          |
| Acc_ang_b_y   | angular acceleration in body frame     | -Inf | Inf | rad/s^2        |
"""

import numpy as np
import pandas as pd
import csv


class Logger:
    def __init__(self, log_path):
        logfile = open(log_path, 'w')
        self.logger = csv.writer(logfile, delimiter=',')
        self.logger.writerow(['timestamp', 'U4', 'U7', 'Vx', 'Vz',  'Gamma',
                              'Ang_vel_y', 'Acc_b_x', 'Acc_b_z', 'Ang_acc_b_y'])

    def log_data(self, env, action, accelerations, time):
        acc_w_x = accelerations['linX']
        acc_w_z = accelerations['linZ']
        gamma = env.state[4]

        # multiply the accelerations with the transormation matrix to body frame
        # (rotation by gamma in mathematical positive direction around y)
        acc_b_x = acc_w_x * np.cos(gamma) - acc_w_z * np.sin(gamma)
        acc_b_z = acc_w_x * np.sin(gamma) + acc_w_z * np.cos(gamma)
        acc_ang_b_y = accelerations['angY']

        self.logger.writerow([time, action[0], action[1], env.state[2], - env.state[3], env.state[4],
                              env.state[5], acc_b_x, acc_b_z, acc_ang_b_y])

    def get_data(path):
        data = pd.read_csv(path)
        return data

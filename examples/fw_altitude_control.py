#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

# core modules
import unittest

# 3rd party modules
import gym
import numpy as np
import logging
import csv

# internal modules
import windywings
from timeit import default_timer as timer

class Environments(unittest.TestCase):
	def test_env(self):
		env = gym.make('fixedwing-longitudinal', render_mode='human')
		env.reset()
		start_t=timer()

		logger = self.initializeLogger(env, 'data.csv')

		for i,_ in enumerate(range(400)): #dt=0.03, 400*0.03=12s
			# action = env.control()
			action = [0.0, 0.0]
			_, reward, done, _, accelerations = env.step(action)
			env.render()

			self.logData(env, logger, action, accelerations, i * env.dt)

			if(done):
				env.reset()
		end_t=timer()
		print("simulation time=",end_t-start_t)
		# env.plot_state()
	

	def initializeLogger(self, env, log_path):
		"""
		logging file for inputs and measurements according to data-driven-dynamics naming;
		all remaining quantities are assumed to be zero

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

		logfile = open(log_path, 'w')
		logger = csv.writer(logfile, delimiter=',')
		logger.writerow(['timestamp', 'U4', 'U7', 'Vx', 'Vz', 'Gamma', 'Ang_vel_y', 'Acc_b_x', 'Acc_b_z', 'Ang_acc_b_y'])

		return logger


	def logData(self, env, logger, action, accelerations, time):
		acc_w_x = accelerations['linX']
		acc_w_z = accelerations['linZ']
		gamma = env.state[4]

		# multiply the accelerations with the transormation matrix to body frame
		# (rotation by gamma in mathematical positive direction around y)
		acc_b_x = acc_w_x * np.cos(gamma) - acc_w_z * np.sin(gamma)
		acc_b_z = acc_w_x * np.sin(gamma) + acc_w_z * np.cos(gamma)
		acc_ang_b_y = accelerations['angY']

		logger.writerow([time, action[0], action[1], env.state[2], - env.state[3], env.state[4], \
			env.state[5], acc_b_x, acc_b_z, acc_ang_b_y])



if __name__ == "__main__":
	env=Environments()
	env.test_env()

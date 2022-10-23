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
import csv

# internal modules
from timeit import default_timer as timer
from windywings.logger.default_logger import Logger
from windywings.envs.fixedwing_env import FWLongitudinal


class Environments(unittest.TestCase):
    def test_env(self):
        env = gym.make('fixedwing-longitudinal', render_mode='human')
        env.reset()
        start_t = timer()

        logger = Logger('results/data.csv')

        for i, _ in enumerate(range(400)):  # dt=0.03, 400*0.03=12s
            action = [0.0, 0.0]
            _, reward, done, _, accelerations = env.step(action)
            env.render()

            logger.log_data(env, action, accelerations, i * env.dt)

            if (done):
                env.reset()
        end_t = timer()
        print("simulation time=", end_t-start_t)

    def ramp_input(self, control, start_value, transition_step, end_value, fixed_value, logfile, steps=500):
        env = gym.make('fixedwing-longitudinal')  # render_mode = 'human'
        env.reset(seed=22)
        start_t = timer()

        logger = Logger(logfile)

        for i, _ in enumerate(range(steps)):
            action = env.control(
                control, start_value, transition_step, end_value, steps, fixed_value, i)

            _, reward, done, _, accelerations = env.step(action)
            env.render()

            # only include steady state data in the log for system identification
            if (i >= transition_step):
                logger.log_data(env, action, accelerations, i * env.dt)

            if (done):
                env.reset()

        end_t = timer()
        print("simulation time=", end_t-start_t)


if __name__ == "__main__":
    env = Environments()

    # env.test_env()
    env.ramp_input('ramp_elevator', -1.0, 200, 1.0, 0.0,
                   'results/elevator_ramp_zero_thrust.csv', 2000)
    env.ramp_input('ramp_elevator', -1.0, 200, 1.0, 1.0,
                   'results/elevator_ramp_full_thrust.csv', 2000)
    env.ramp_input('ramp_throttle', 0.0, 200, 1.0, 0.0,
                   'results/throttle_ramp_slow.csv', 2000)
    env.ramp_input('ramp_throttle', 0.0, 100, 1.0, 0.0,
                   'results/throttle_ramp_fast.csv', 700)

    FWLongitudinal.visualize_results(paths=['results/elevator_ramp_zero_thrust.csv', 'results/elevator_ramp_full_thrust.csv'],
                                     variablesX=['Vx', 'Vx'], variablesY=['Vz', 'Vz'], invertedY=[True, True])

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
from pathlib import Path

# 3rd party modules
import gym

# internal modules
from timeit import default_timer as timer
from windywings.logger.default_logger import Logger
from windywings.envs.fixedwing_env import FWLongitudinal


class Environments(unittest.TestCase):
    def control_input(self, control, start_value, transition_step, end_value, fixed_value, logfile, steps=500, initial_state=[0.0, 0, 15.0, 0.0, 0.0, 0.0]):
        env = gym.make('fixedwing-longitudinal')  # render_mode = 'human'
        env.reset(seed=22, initial_state=initial_state)
        start_t = timer()

        logger = Logger(logfile)

        for i, _ in enumerate(range(steps)):
            action = env.control(
                control, start_value, transition_step, end_value, steps, fixed_value, i)

            _, reward, done, _, accelerations = env.step(action)
            env.render()

            # only include steady state data in the log for system identification
            if (i >= transition_step):
                logger.log_data(i - transition_step, env,
                                action, accelerations, i * env.dt)

        end_t = timer()
        print("simulation time=", end_t-start_t)


if __name__ == "__main__":
    env = Environments()

    Path("results").mkdir(parents=True, exist_ok=True)

    env.control_input('ramp_elevator', -1.0, 1000, 0.3, 0.0,
                      'results/elevator_ramp_zero_thrust.csv', 3000, [0.0, 0, 10.0, 5.0, 0.5, 0.0])
    env.control_input('ramp_elevator', -1.0, 1000, 0.3, 1.0,
                      'results/elevator_ramp_full_thrust.csv', 3000, [0.0, 0, 10.0, 5.0, 0.5, 0.0])
    env.control_input('sine_elevator', -1.0, 0, 1.0, 0.0,
                      'results/elevator_sine_zero_thrust.csv', 1500)
    env.control_input('sine_elevator', -1.0, 0, 1.0, 1.0,
                      'results/elevator_sine_full_thrust.csv', 1500)
    env.control_input('ramp_throttle', 0.0, 200, 1.0, 0.0,
                      'results/throttle_ramp_slow.csv', 2000)
    env.control_input('ramp_throttle', 0.0, 100, 1.0, 0.0,
                      'results/throttle_ramp_fast.csv', 700)
    env.control_input('ramp_throttle', 0.0, 100, 1.0, 0.0,
                      'results/throttle_ramp_very_fast.csv', 300)

    fixedwing_env = FWLongitudinal()

    # generate flight data for different lengths / speeds of elevator ramp inputs
    # for k in range(0,10):
    #     env.control_input('ramp_elevator', -1.0, 1000, 0.3, 0.0,
    #                   'results/elevator_ramp_speed' + str(k) + '.csv', 1050 + k * 50, [0.0, 0, 10.0, 5.0, 0.5, 0.0])

    # illustrate the non-linear lift model with sigmoid fusion
    fixedwing_env.demo_sigmoid_nonlinear_cl_model()

    # TODO: add possibility to disable plots with corresponding argument in command line
    # TODO: refactor code so that the illustrations for different files are shown in the same plot
    # visualize the flight data from the log file in a V vs Vz plot (to identify min sink speed)
    fixedwing_env.visualize_results(plots=[{'path': 'results/elevator_ramp_zero_thrust.csv', 'name': 'Sink rate vs. velocity at zero thrust'},
                                           {'path': 'results/elevator_ramp_full_thrust.csv', 'name': 'Sink rate vs. velocity at full thrust'}],
                                    mode='V Vz')

    # visualize the lift vs. drag values against the velocity in a plot for zero thrust data
    fixedwing_env.visualize_results(plots=[{'path': 'results/elevator_ramp_zero_thrust.csv', 'name': 'Lift-drag ratio vs. velocity at zero thrust'}],
                                    mode='L/D V')

    # visualize range of angles of attack during simulation
    fixedwing_env.visualize_results(
        plots=[{'path': 'results/elevator_ramp_zero_thrust.csv', 'name': 'AoA over time for zero thrust'},
               {'path': 'results/elevator_ramp_full_thrust.csv', 'name': 'AoA over time for full thrust'}],
        mode='AoAs')

    # visualize the flight path angles during simulation
    fixedwing_env.visualize_results(
        plots=[{'path': 'results/elevator_ramp_zero_thrust.csv', 'name': 'Flight path angle over time for zero thrust'},
               {'path': 'results/elevator_ramp_full_thrust.csv', 'name': 'Flight path angle over time for full thrust'}],
        mode='Gammas')

    # TODO: add visualizations for sine inputs
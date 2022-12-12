"""
Simulation script for simulating NPFG of a multicopter
"""
import unittest
from typing import Optional
import pygame

import gym
import windywings

from timeit import default_timer as timer
import numpy as np

class Environments(unittest.TestCase):
    def __init__(self):
        # NPFG Test Environment
        screen_width, screen_height = 600, 600
        world_size = 400.0 # Size of the simulated world (e.g. Width) in meters (will be scaled to screen internally)
        DEBUG_ENABLE = True

        # Runtime user settings
        self._stop_every_1_sec = False

        # World size of 200 means that it's a 200 m x 200 m space that vehicle can fly in
        self.env = gym.make('multicopter-fixedwing-lateral-npfg', vehicle_type = 'multicopter', world_size = world_size, screen_size = [screen_width, screen_height], render_mode='human', DEBUG = DEBUG_ENABLE)

        # Initial state setting
        posX = -world_size/2.0
        posY = world_size/8.0
        vehicle_longitudinal_speed = 10.0
        vehicle_lateral_speed = 10.0
        vehicle_heading = 0.0

        pathX = 0.0
        pathY = 0.0
        path_heading = 0#np.pi/4
        path_curvature = 0.0#0.01

        initial_state = np.array([posX, posY, vehicle_longitudinal_speed, vehicle_heading, pathX, pathY, path_heading, path_curvature, vehicle_lateral_speed], dtype=np.float32)
        
        self.env.reset(initial_state=initial_state)

    def test_env(self):
        ''' Executes the simulation'''
        start_t=timer()

        # NOTE: Vehicle dynamics dt is 0.03 seconds, and FPS rendering in Pygame is 100 FPS == 0.01 seconds
        # Therefore, the vehicle will travel 3 times faster than it *actually is, this lowers simulation time.
        sim_time_secs = 7.0

        for i,_ in enumerate(range(int(sim_time_secs / 0.01))):

            # Calculate NPFG logic
            action = self.env.control()

            # Take a setp in simulation
            obs, reward, done, _ = self.env.step(action)

            # Take a snapshot & analyze
            if self._stop_every_1_sec:
                if i != 0 and i%100 == 0:
                    input('Input key to simulate 1 second further')

            if(done):
                env.reset()

        end_t=timer()
        print("Actual simulation time=",end_t-start_t)

        # Wait for user input (to give time for screen capture)
        input('Waiting for user ...')

if __name__ == "__main__":
    env=Environments()
    env.test_env()
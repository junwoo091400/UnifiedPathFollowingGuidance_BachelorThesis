"""
Simulation script for simulating NPFG of a fixed-wing
"""
import unittest
from typing import Optional
import pygame

import gym
import windywings

from timeit import default_timer as timer
import numpy as np

'''
class LinePathDrawWrapper(gym.Wrapper):
    """ Wrapper to draw single line Path formed by 2 waypoints in 2D plane, and logic running by NPFG on top of basic Environment """
    def __init__(self, env, waypoint_A: np.ndarray((2, 1)) = None, waypoint_B: np.ndarray((2, 1)) = None):
        super().__init__(env)
        
        # Path
        if waypoint_A is not None and waypoint_B is not None:
            self.path = np.append(waypoint_A, waypoint_B, axis=1)
            assert np.shape(self.path) == (2, 2)
        else:
            self.path = None

        print('Path:', self.path)

    def render(self):
        """ Over-ridden render function to visualize Path & NPFG calculations """
        print('Hi render')
        self.env.render()
        print('self.env.screen:', self.env.screen, 'self.env.surf:', self.env.surf)
        # Additional render
        if self.env.screen is not None:
            print('Drawing line to:', self.env.surf)
            pygame.draw.line(self.env.surf, color=(0, 0, 0), start_pos=(0,0), end_pos=(200,100), width=10)
'''

class Environments(unittest.TestCase):
    def __init__(self):
        # NPFG Test Environment
        screen_width, screen_height = 600, 600
        world_size = 400.0 # Size of the simulated world (e.g. Width) in meters (will be scaled to screen internally)
        DEBUG_ENABLE = True

        # Runtime user settings
        self._stop_every_1_sec = False

        # World size of 200 means that it's a 200 m x 200 m space that vehicle can fly in
        self.env = gym.make('fixedwing-lateral-npfg', world_size = world_size, screen_size = [screen_width, screen_height], render_mode='human', DEBUG = DEBUG_ENABLE)

        # Initial state setting
        posX = -world_size/2
        posY = world_size/4
        vehicle_speed = 20.0
        vehicle_heading = 0.0

        pathX = 0.0
        pathY = 0.0
        path_heading = 0#np.pi/4
        path_curvature = 0.0#0.01

        initial_state = np.array([posX, posY, vehicle_speed, vehicle_heading, pathX, pathY, path_heading, path_curvature], dtype=np.float32)
        
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
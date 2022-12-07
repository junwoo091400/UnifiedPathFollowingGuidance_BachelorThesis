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
        screen_width, screen_height = 1200, 1200
        DEBUG_ENABLE = True

        self.env = gym.make('fixedwing-lateral-npfg', screen_size = [screen_width, screen_height], render_mode='human', DEBUG = DEBUG_ENABLE)

        # Initial state setting
        # PosX: 0, PosY: 0, Speed: 15, Heading: 0
        # PathX: 0, PathY: 0, PathHeading: PI/4, PathCurvature: 0
        posX = screen_width/2
        posY = screen_height/2
        path_heading = np.pi/4
        path_curvature = 0

        initial_state = np.array([posX, posY, 15.0, 0.0, 0.0, 0.0, path_heading, path_curvature], dtype=np.float32)
        
        self.env.reset(initial_state=initial_state)

    def test_env(self):
        ''' Executes the simulation'''
        start_t=timer()

        for i,_ in enumerate(range(400)): #dt=0.01, 400*0.01=4s

            # Calculate NPFG logic
            action = self.env.control()

            # Take a setp in simulation
            obs, reward, done, _ = self.env.step(action)

            if(done):
                env.reset()

        end_t=timer()
        print("simulation time=",end_t-start_t)

if __name__ == "__main__":
    env=Environments()
    env.test_env()
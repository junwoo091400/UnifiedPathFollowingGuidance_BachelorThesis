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

import argparse

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
    def __init__(self, debug_enable=False, path_bearing_deg=0.0, path_curvature=0.0, stop_every_1sec=False, vehicle_speed=20.0, vehicle_bearing_deg=0.0, world_size=300.0, nominal_airspeed=15.0, reference_airspeed=15.0):
        # NPFG Test Environment
        screen_width, screen_height = 1000, 1000

        # Runtime user settings
        self._stop_every_1_sec = stop_every_1sec

        # World size of 200 means that it's a 200 m x 200 m space that vehicle can fly in
        self.env = gym.make('multicopter-fixedwing-lateral-npfg', vehicle_type = 'fixedwing', nominal_airspeed=nominal_airspeed, reference_airspeed=reference_airspeed, world_size = world_size, screen_size = [screen_width, screen_height], render_mode='human', DEBUG = debug_enable)

        # Initial state setting
        posX = -world_size/4.0 # 1/4 from left
        posY = world_size/8.0 # Slightly up
        vehicle_longitudinal_speed = vehicle_speed
        vehicle_lateral_speed = 0.0 # Always set initial lateral speed to 0
        vehicle_heading = np.deg2rad(vehicle_bearing_deg)

        pathX = 0.0
        pathY = 0.0
        path_heading = np.deg2rad(path_bearing_deg)

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
    parser = argparse.ArgumentParser('Multicopter NPFG Control Simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug printout in the Environment')
    parser.add_argument('--path_bearing', type=float, default=0.0, help='Path target bearing in degrees')
    parser.add_argument('--path_curvature', type=float, dest='path_curvature', default=0.0, help='Path curvature (signed) in m^-1')
    parser.add_argument('--steps', action='store_true', help='Stop simulation every second to evaluate vehicle state')
    parser.add_argument('--vehicle_speed', type=float, dest='vehicle_speed', default=20.0, help='Initial vehicle speed in m/s')
    parser.add_argument('--vehicle_bearing', type=float, dest='vehicle_bearing', default=0.0, help='Initial vehicle bearing in degrees')
    parser.add_argument('--world_size', dest='world_size', default=300.0, help='World size in meters (will be a square with Size x Size shape)')
    parser.add_argument('--nominal_airspeed', type=float, help='Nominal airspeed that vehicle should achieve when on path')
    parser.add_argument('--reference_airspeed', type=float, help='Reference airspeed in m/s')
    args = parser.parse_args()

    env=Environments(args.debug, args.path_bearing, args.path_curvature, args.steps, args.vehicle_speed, args.vehicle_bearing, args.world_size, args.nominal_airspeed, args.reference_airspeed)
    env.test_env()
"""
Simulation script for simulating NPFG contro of a point-mass multicopter
"""
import unittest
from typing import Optional
from timeit import default_timer as timer
import numpy as np
import argparse

import pygame
import gym
from windywings.libs.npfg import NPFG

# Rendering
SCREEN_SIZE_DEFAULT = (1200, 1200) # Default screen size for rendering
MULTICOPTER_CIRCLE_RADIUS = 10 # Radius of the circle representing vehicle in rendering

# Environment constraints
NOM_VELOCITY = 0.0
MAX_VELOCITY = 15.0
MAX_ACCELERATION = 10.0
WORLD_SIZE_DEFAULT = 300.0 # [m] Default simulated world size
PATH_BEARING_DEG_DEFAULT = 0.0
PATH_CURVATURE_DEFAULT = 0.0

SIM_DURATION_SEC = 20.0 # [s] How long the simulation will run (`step()` time)
SIM_TIME_DT = 0.03 # [s] Dt from each `step` (NOTE: Ideally, this should match 1/FPS of the environment, but since we don't render in the environment, this isn't necessary)

class MC_npfg_pointmass(unittest.TestCase):
    ''' NPFG Testing class on a Point-mass modeled multicopter environment '''
    def __init__(self, debug_enable=False, path_bearing_deg=PATH_BEARING_DEG_DEFAULT, path_curvature=PATH_CURVATURE_DEFAULT, stop_every_1sec=False, vehicle_speed=MAX_VELOCITY/2, nominal_airspeed=NOM_VELOCITY, world_size=WORLD_SIZE_DEFAULT):
        velocity_bound = np.array([-MAX_VELOCITY, MAX_VELOCITY])
        acceleration_bound = np.array([-MAX_ACCELERATION, MAX_ACCELERATION])
        self.env = gym.make('multicopter-pointmass', world_size=world_size, velocity_bounds=velocity_bound, acceleration_bounds=acceleration_bound)

        # Initial state setting
        pos = np.array([-world_size/2, world_size/8])
        vel = np.array([vehicle_speed, 0.0])
        acc = np.array([0.0, 0.0])
        initial_state = np.concatenate((pos, vel, acc), axis=None)
        self.env.reset(initial_state=initial_state)

        # NPFG
        self.npfg = NPFG(nominal_airspeed, MAX_VELOCITY) # NOTE: Max velocity is fixed to this constant, so if nominal velocity is set higher by user, it will be capped!! (Currently 0.0, so doesn't matter)
        self.path_bearing = np.deg2rad(path_bearing_deg)
        self.path_unit_tangent_vec = np.array([np.cos(self.path_bearing), np.sin(self.path_bearing)])
        self.path_curvature = path_curvature
        self.path_position = np.array([0.0, 0.0])

        # Runtime user settings
        self._stop_every_1_sec = stop_every_1sec
        self.debug_enable = debug_enable

        # Runtime variables
        self.observation = self.env.observation_space.sample() # Copy of the observation from the last `step` function call
        self.action = self.env.action_space.sample() # Last sent action, which is derived from NPFG
        self.air_vel_ref = np.array([0.0, 0.0])
        self.acc_ff_curvature = 0.0

        # Rendering
        self.screen = None
        self.clock = None
        self.world_to_screen_scaling = np.max(SCREEN_SIZE_DEFAULT)/world_size # NOTE: Assume our screen will be default.

    def test_env(self):
        ''' Executes the simulation'''
        start_t=timer()

        for i,_ in enumerate(range(int(SIM_DURATION_SEC/SIM_TIME_DT))):
            # Decode observation to get current vehicle state
            pos, vel, acc = self.env.decode_state(self.observation)

            # Calculate NPFG logic
            self.npfg.navigatePathTangent_nowind(pos, self.path_position, self.path_unit_tangent_vec, vel, self.path_curvature)
            self.air_vel_ref = self.npfg.getAirVelRef()
            self.acc_ff_curvature = self.npfg.getAccelFFCurvature()

            # Calculate the action
            self.action = np.concatenate((self.air_vel_ref, self.acc_ff_curvature), axis=None)

            # Take a step in simulation
            self.observation, reward, done, info = self.env.step(self.action)

            # Render
            self.render()

            # Debug
            print('Pos', pos, 'Vel', vel, 'Vel ref', self.air_vel_ref, 'Acc FF', self.acc_ff_curvature)

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

    def world2screen(self, coords: np.array):
        ''' Converts the world coordinate (in meters) to a pixel location in Rendering screen '''
        assert np.shape(coords) == (2, ), "Coordinates to transform in World2Screen must be of size (2, )!"
        # We want to place the (0.0, 0.0) in the middle of the screen, therefore the transform is:
        # (X): Screen-size/2 + coordinateX * scaling
        return np.array(self.screen.get_size())/2 + coords * self.world_to_screen_scaling

    def render(self):
        ''' Render the simulation '''
        import pygame
        from pygame import gfxdraw
        from pygame.locals import Rect

        # Construct screen
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                SCREEN_SIZE_DEFAULT
            )
            pygame.display.set_caption('Multicopter Point Mass NPFG Simulation')
        
        # Construct clock
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Pygame surface construction
        self.surf = pygame.Surface(self.screen.get_size())
        self.surf.fill((255, 255, 255)) # White background

        # Decode observation to get current vehicle state
        pos, vel, acc = self.env.decode_state(self.observation)

        # Draw the Vehicle
        pygame.draw.circle(self.surf, pygame.color.Color('black'), self.world2screen(pos), MULTICOPTER_CIRCLE_RADIUS)

        # Draw the path target
        PATH_LENGTH_HALF = 2000 # Arbitrary length to extend the path to draw the line in the frame
        path_start_pos = self.world2screen(self.path_position - self.path_unit_tangent_vec * PATH_LENGTH_HALF)
        path_end_pos = self.world2screen(self.path_position + self.path_unit_tangent_vec * PATH_LENGTH_HALF)
        
        if (np.isfinite(path_start_pos).all() and np.isfinite(path_end_pos).all()):
            # Only draw when the coordinates are finite (defined)
            try:
                pygame.draw.line(self.surf, (255, 0, 0), path_start_pos, path_end_pos)
            except Exception as e:
                print(e)
                print('Path start: {}, Path end: {}'.format(path_start_pos, path_end_pos))

        # Draw on the screen
        self.surf = pygame.transform.flip(self.surf, False, True) # Flips the surface drawing in Y-axis, so that frame coordinate wise, X is RIGHT, Y is UP in the visualization
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(1/SIM_TIME_DT)
        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multicopter NPFG Control Simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug printout in the Environment')
    parser.add_argument('--path_bearing', type=float, default=PATH_BEARING_DEG_DEFAULT, help='Path target bearing in degrees')
    parser.add_argument('--path_curvature', type=float, default=PATH_CURVATURE_DEFAULT, dest='path_curvature', help='Path curvature (signed) in m^-1')
    parser.add_argument('--steps', action='store_true', help='Stop simulation every second to evaluate vehicle state')
    parser.add_argument('--vehicle_speed', type=float, dest='vehicle_speed', default=MAX_VELOCITY/2, help='Initial vehicle speed in m/s')
    parser.add_argument('--world_size', dest='world_size', default=WORLD_SIZE_DEFAULT, help='World size in meters (will be a square with Size x Size shape)')
    parser.add_argument('--nominal_airspeed', type=float, default=NOM_VELOCITY, help='Nominal airspeed that vehicle should achieve when on path')
    args = parser.parse_args()

    env=MC_npfg_pointmass(args.debug, args.path_bearing, args.path_curvature, args.steps, args.vehicle_speed, args.nominal_airspeed, args.world_size)
    env.test_env()
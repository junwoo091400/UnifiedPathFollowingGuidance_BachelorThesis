"""
Simulation script for demonstrating constant airspeed, non-holonomic FW and waypoint following based holonomic MC path following capability difference

MC: Hybrid Unicyclic Uniform
FW: Unicyclic (But we will fake it via using MC Pointmass environment anyways)
"""
import unittest
from timeit import default_timer as timer
import numpy as np
import argparse

import pygame
import gym

import matplotlib.pyplot as plt

from windywings.envs import MCPointMass

from theories.velocity_reference_algorithms import *

# Rendering
SCREEN_SIZE_DEFAULT = (1000, 1000) # Default screen size for rendering
MULTICOPTER_CIRCLE_RADIUS = 10 # Radius of the circle representing vehicle in rendering

UNICYCLIC_COLOR = 'red'
HYBRID_UNICYCLIC_UNIFORM_COLOR = 'green'

# Vehicle constraints
MIN_VELOCITY = 0.0
NOM_VELOCITY = 15.0
MAX_VELOCITY = 20.0
VEL_RANGE = np.array([MIN_VELOCITY, NOM_VELOCITY, MAX_VELOCITY])
MAX_ACCELERATION = 7.0 # [m/s^2] Max Accel in both X and Y direction

# User settings
V_PATH = 1.0
TRACK_KEEPING_SPEED_DEFAULT = 0.0 # Disable track keeping
V_APPROACH_MIN_DEFAULT = 0.0 # Disable V_approach_min

# World
WORLD_WIDTH_DEFAULT = 200.0 # [m] Default simulated world width (also height)
PATH_BEARING_DEG_DEFAULT = 0.0
PATH_CURVATURE_DEFAULT = 0.0

# Simulation settings
SIM_DURATION_SEC = 50.0 # [s] How long the simulation will run (`step()` time)
SIM_TIME_DT = 0.03 # [s] Dt from each `step` (NOTE: Ideally, this should match 1/FPS of the environment, but since we don't render in the environment, this isn't necessary)

def world2screen(coords: np.array, world_width):
        '''
        Converts the world coordinate (in meters) to a pixel location in Rendering screen

        NOTE: This should be a global function to make sure we have consistent scaling of all simulation
        environments into a single screen
        '''
        world_to_screen_scaling = np.max(SCREEN_SIZE_DEFAULT)/world_width # NOTE: Assume our screen will be default.
        assert np.shape(coords) == (2, ), "Coordinates to transform in World2Screen must be of size (2, )!"
        # We want to place the (0.0, 0.0) in the middle of the screen, therefore the transform is:
        # (X): Screen-size/2 + coordinateX * scaling
        return np.array(SCREEN_SIZE_DEFAULT)/2 + coords * world_to_screen_scaling

class TrackRecord:
    '''
    Wrapper class that includes velCurve object & position history & gym environment for rendering
    '''
    def __init__(self, vel_curve_obj: VelocityReferenceCurves, name: str, path_bearing_deg, path_curvature, vehicle_speed, world_width, color):
        # History
        self.state_history = None

        # VelCurve Object
        self.velCurve = vel_curve_obj
        self.name = name # VelCurve Name

        # Create environment
        velocity_bound = np.array([-MAX_VELOCITY, MAX_VELOCITY])
        acceleration_bound = np.array([-MAX_ACCELERATION, MAX_ACCELERATION])
        self.env = gym.make('multicopter-pointmass', world_size=world_width, velocity_bounds=velocity_bound, acceleration_bounds=acceleration_bound)

        # Path setting
        self.path_bearing = np.deg2rad(path_bearing_deg)
        self.path_unit_tangent_vec = np.array([np.cos(self.path_bearing), np.sin(self.path_bearing)])
        self.path_curvature = path_curvature
        self.path_position = np.array([0.0, 0.0])

        # Initial state setting
        pos = np.array([-world_width/3, world_width/3]) # Semi-left semi-top corner
        vel = np.array([vehicle_speed, 0.0])
        acc = np.array([0.0, 0.0])
        initial_state = np.concatenate((pos, vel, acc), axis=None)
        self.env.reset(initial_state=initial_state)

        # Runtime variables
        self.observation = self.env.observation_space.sample() # Copy of the observation from the last `step` function call
        self.action = self.env.action_space.sample() # Last sent action, which is derived from NPFG
        self.air_vel_ref = np.array([0.0, 0.0])
        self.acc_ff_curvature = 0.0
        self.position_history = None

        # Cache runtime constants
        self.world_width = world_width
        self.color = color

    def setPath(self, path_bearing_deg, path_position):
        '''
        Sets the path to follow
        '''
        self.path_bearing = np.deg2rad(path_bearing_deg)
        self.path_unit_tangent_vec = np.array([np.cos(self.path_bearing), np.sin(self.path_bearing)])
        self.path_position = path_position

    def update(self):
        '''
        Update the simulation step

        Return whether sim is done
        '''
        # Decode observation to get current vehicle state
        pos, vel, acc = self.env.decode_state(self.observation)

        # Calculate input for Vel Curves
        unit_path_tangent = self.path_unit_tangent_vec
        position_error_vec = pos - self.path_position
        signed_track_error = np.cross(unit_path_tangent, position_error_vec) # If positive, vehicle is on left side of path

        self.air_vel_ref = self.velCurve.calculate_velRef(np.abs(signed_track_error), V_PATH) # Note, result is in path-relative frame

        if signed_track_error > 0:
            # NOTE: This only makes sense when path is on X-axis.
            self.air_vel_ref[1] = -self.air_vel_ref[1] # Invert Y-axis value, as positive orthogonal vel in VelCurve means going *towards the path (for now)

        # Rotate to world frame
        self.air_vel_ref = np.matmul(np.array([[np.cos(self.path_bearing), -np.sin(self.path_bearing)], [np.sin(self.path_bearing), np.cos(self.path_bearing)]]), self.air_vel_ref)

        # Calculate the action
        self.acc_ff_curvature = np.array([0.0, 0.0]) # Set 0 accel in 2D plane for now
        self.action = np.concatenate((self.air_vel_ref, self.acc_ff_curvature), axis=None)

        # Take a step in simulation
        self.observation, reward, done, info = self.env.step(self.action)

        # Cache state
        # print(self.name, self.observation)

        if self.state_history is not None:
            self.state_history = np.concatenate((self.state_history, [self.observation]), axis=0)
        else:
            self.state_history = np.array([self.observation])

        return done

    def render(self, surf: pygame.Surface):
        ''' Draw vehicle position history & state in the provied surface'''
        # Decode observation to get current vehicle state
        pos, vel, acc = self.env.decode_state(self.observation)

        # Draw vehicle path
        if self.position_history is not None:
            self.position_history = np.concatenate((self.position_history, [pos]), axis=0)
        else:
            self.position_history = np.array([pos])

        if (self.position_history is not None) and (self.position_history.shape[0] > 1):
            # Map the World_Size argument along the mapping function
            xys = list(map(world2screen, self.position_history, [self.world_width]*len(self.position_history)))
            pygame.draw.aalines(surf, points=xys, closed=False, color=pygame.Color(self.color))

        # Draw the Vehicle
        pygame.draw.circle(surf, pygame.Color(self.color), world2screen(pos, self.world_width), MULTICOPTER_CIRCLE_RADIUS)

    ''' Getters '''
    def get_position(self):
        '''
        Return the last position
        '''
        if self.position_history is not None:
            return self.position_history[-1]
        else:
            return None

    def get_state_history(self):
        return self.state_history
    
    def get_velCurve_object(self):
        return self.velCurve
    
    def get_name(self):
        return self.name
    
    def get_color(self):
        return self.color

class MC_velCurve_pointMass(unittest.TestCase):
    ''' VelCurve based PF testing class on a Point-mass modeled multicopter environment '''
    def __init__(self, path_bearing_deg=PATH_BEARING_DEG_DEFAULT, path_curvature=PATH_CURVATURE_DEFAULT, vehicle_speed=MAX_VELOCITY/2, world_width=WORLD_WIDTH_DEFAULT, debug_enable=False, stop_every_1sec=False, sim_time = SIM_DURATION_SEC):
        # Necessary constants
        GROUND_SPEED_DEFAULT = VEL_RANGE[1] # Only should be used by TJ NPFG

        # TrackRecords
        self.trackRecords = []
        self.trackRecords.append(TrackRecord(Unicyclic(VEL_RANGE, GROUND_SPEED_DEFAULT, TRACK_KEEPING_SPEED_DEFAULT), 'Unicyclic', path_bearing_deg, path_curvature, vehicle_speed, world_width, UNICYCLIC_COLOR))
        self.trackRecords.append(TrackRecord(HybridUnicyclicUniform(VEL_RANGE), 'Hybrid Unicyclic Uniform', path_bearing_deg, path_curvature, vehicle_speed, world_width, HYBRID_UNICYCLIC_UNIFORM_COLOR))

        # Runtime user settings
        self._stop_every_1_sec = stop_every_1sec
        self.debug_enable = debug_enable

        # Rendering
        self.screen = None
        self.clock = None

        # Runtime constants
        self.path_bearing = np.deg2rad(path_bearing_deg)
        self.path_unit_tangent_vec = np.array([np.cos(self.path_bearing), np.sin(self.path_bearing)])
        self.path_curvature = path_curvature
        self.path_position = np.array([0.0, 0.0])
        self.world_width = world_width
        self.sim_time = sim_time

    def render(self):
        ''' Render the simulation '''
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

        # Visualize simulation
        for tr in self.trackRecords:
            tr.render(self.surf)

        # Draw the path target
        self.draw_path(self.path_position, self.path_unit_tangent_vec)        

        # Draw on the screen
        self.surf = pygame.transform.flip(self.surf, False, True) # Flips the surface drawing in Y-axis, so that frame coordinate wise, X is RIGHT, Y is UP in the visualization
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        # self.clock.tick(1/SIM_TIME_DT) # Remove clock ticking to reduce simulation time.
        pygame.display.flip()

    def draw_path(self, path_position, path_unit_tangent_vec):
        PATH_LENGTH_HALF = 2000 # Arbitrary length to extend the path to draw the line in the frame
        path_start_pos = world2screen(path_position - path_unit_tangent_vec * PATH_LENGTH_HALF, self.world_width)
        path_end_pos = world2screen(path_position + path_unit_tangent_vec * PATH_LENGTH_HALF, self.world_width)
        
        if (np.isfinite(path_start_pos).all() and np.isfinite(path_end_pos).all()):
            # Only draw when the coordinates are finite (defined)
            try:
                pygame.draw.line(self.surf, (255, 0, 0), path_start_pos, path_end_pos)
            except Exception as e:
                print(e)
                print('Path start: {}, Path end: {}'.format(path_start_pos, path_end_pos))

    def handle_simulation_termination(self):
        ''' Waits for the user to press Escape (ESC) key, and until then keeps pumping the pygame (to prevent 'not responding' error) '''
        print('Press ESC to exit the simulation')
        while True:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
        print('Exiting the simulation ...')

    def test_env(self):
        ''' Executes the simulation'''
        # New path to switch to showcase change in path
        new_vertical_path_bearing_deg = 90
        new_vertical_path_position = [80.0, 0.0]

        # Path progress tracker for each PF algorithm
        path_progress = np.zeros(len(self.trackRecords))

        start_t=timer()

        for i,_ in enumerate(range(int(self.sim_time/SIM_TIME_DT))):
            # Update simulation
            for tr in self.trackRecords:
                tr.update()

                # Check if the path can be switched
                pos = tr.get_position()
                if pos is not None:
                    print(pos)
                    if (np.linalg.norm(pos - new_vertical_path_position) < 20):
                        # Switch path
                        print('Path switched!')
                        tr.setPath(new_vertical_path_bearing_deg, new_vertical_path_position)
                
            # Render
            if self.screen is not None:
                # Special new path drawing
                self.draw_path(new_vertical_path_position, np.array([np.cos(np.deg2rad(new_vertical_path_bearing_deg)), np.sin(np.deg2rad(new_vertical_path_bearing_deg))]))

            self.render()

        end_t=timer()
        print("Simulated time={}s, Computation time={}s".format(SIM_DURATION_SEC, (end_t-start_t)))

        # Handle simulation termination
        self.handle_simulation_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multicopter NPFG Control Simulation')
    parser.add_argument('--sim_time', type=float, default=SIM_DURATION_SEC, help='Simulation time in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug printout in the Environment')
    parser.add_argument('--path_bearing', type=float, default=PATH_BEARING_DEG_DEFAULT, help='Path target bearing in degrees')
    parser.add_argument('--path_curvature', type=float, default=PATH_CURVATURE_DEFAULT, dest='path_curvature', help='Path curvature (signed) in m^-1')
    parser.add_argument('--steps', action='store_true', default=False, help='Stop simulation every second to evaluate vehicle state')
    parser.add_argument('--vehicle_speed', type=float, dest='vehicle_speed', default=NOM_VELOCITY, help='Initial vehicle speed in m/s')
    parser.add_argument('--world_width', dest='world_width', default=WORLD_WIDTH_DEFAULT, help='World size in meters (will be a square with Size x Size shape)')
    args = parser.parse_args()

    env=MC_velCurve_pointMass(args.path_bearing, args.path_curvature, args.vehicle_speed, args.world_width, args.debug, args.steps, args.sim_time)
    env.test_env()
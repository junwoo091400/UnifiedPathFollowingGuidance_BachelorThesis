"""
Create artwork of different intiial / parameter settings for the Unicyclic / Hybrid methods!
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
UNICYCLIC_COLOR = 'red'
HYBRID_UNICYCLIC_UNIFORM_COLOR = 'orange'

# Vehicle constraints
MIN_VELOCITY = 0.0
NOM_VELOCITY = 6.0
MAX_VELOCITY = 15.0
VEL_RANGE = np.array([MIN_VELOCITY, NOM_VELOCITY, MAX_VELOCITY])
V_PATH = 12.0

# World
WORLD_WIDTH_DEFAULT = 150.0 # [m] Default simulated world width (also height)
PATH_BEARING_DEG_DEFAULT = 0.0
PATH_CURVATURE_DEFAULT = 0.0

# Simulation settings
SIM_DURATION_SEC = 20.0 # [s] How long the simulation will run (`step()` time)
SIM_TIME_DT = 0.03 # [s] Dt from each `step` (NOTE: Ideally, this should match 1/FPS of the environment, but since we don't render in the environment, this isn't necessary)

# Artwork settings
v_path_range = [1.0, 5.0, 7.0, 12.0]
v_approach_range = [3.0, NOM_VELOCITY, 9.0]

# Screen
pygame.init()
pygame.display.init()

screen = pygame.display.set_mode(
    SCREEN_SIZE_DEFAULT
)

surf = pygame.Surface(SCREEN_SIZE_DEFAULT)
surf.fill((255, 255, 255)) # White background

def world2screen(coords: np.array):
    '''
    Converts the world coordinate (in meters) to a pixel location in Rendering screen

    NOTE: This should be a global function to make sure we have consistent scaling of all simulation
    environments into a single screen
    '''
    world_to_screen_scaling = np.max(SCREEN_SIZE_DEFAULT)/WORLD_WIDTH_DEFAULT # NOTE: Assume our screen will be default.
    assert np.shape(coords) == (2, ), "Coordinates to transform in World2Screen must be of size (2, )!"
    # We want to place the (0.0, 0.0) in the middle of the screen, therefore the transform is:
    # (X): Screen-size/2 + coordinateX * scaling
    return np.array(SCREEN_SIZE_DEFAULT)/2 + coords * world_to_screen_scaling

# Setting
initial_pos = [-50.0, 70.0]

# Unicyclic
unicyclic = Unicyclic(VEL_RANGE, VEL_RANGE[1], 0.0)

position_history = [np.array(initial_pos)]
while True:
    # Path at (0,0), with bearing 0 (+X direction)
    pos = position_history[-1]

    unit_path_tangent = np.array([1.0, 0.0])
    position_error_vec = pos

    signed_track_error = np.cross(unit_path_tangent, position_error_vec) # If positive, vehicle is on left side of path

    air_vel_ref = unicyclic.calculate_velRef(np.abs(signed_track_error), 0.0000001) #Dummy VPath

    if signed_track_error > 0:
        # NOTE: This only makes sense when path is on X-axis.
        air_vel_ref[1] = -air_vel_ref[1] # Invert Y-axis value, as positive orthogonal vel in VelCurve means going *towards the path (for now)

    # Integrate path
    position_history.append(np.array(position_history[-1] + air_vel_ref*SIM_TIME_DT))

    # Convergence condition
    if position_history[-1][1] < 0.5:
        break
xys = list(map(world2screen, position_history))
pygame.draw.aalines(surf, points=xys, closed=False, color=pygame.Color([255, 0, 0])) # Redder with higher vel

# Simulate
for v_approach in v_approach_range:
    for v_path in v_path_range:
        hybrid = HybridUnicyclicUniform(VEL_RANGE, v_approach)
        # Simulate until on path
        position_history = [np.array(initial_pos)]
        while True:
            # Path at (0,0), with bearing 0 (+X direction)
            pos = position_history[-1]

            unit_path_tangent = np.array([1.0, 0.0])
            position_error_vec = pos

            signed_track_error = np.cross(unit_path_tangent, position_error_vec) # If positive, vehicle is on left side of path

            air_vel_ref = hybrid.calculate_velRef(np.abs(signed_track_error), v_path)

            if signed_track_error > 0:
                # NOTE: This only makes sense when path is on X-axis.
                air_vel_ref[1] = -air_vel_ref[1] # Invert Y-axis value, as positive orthogonal vel in VelCurve means going *towards the path (for now)

            # Integrate path
            position_history.append(np.array(position_history[-1] + air_vel_ref*SIM_TIME_DT))

            # Convergence condition
            if position_history[-1][1] < 0.5:
                break
        xys = list(map(world2screen, position_history))
        pygame.draw.aalines(surf, points=xys, closed=False, color=pygame.Color([0, v_path*v_approach*2, 0])) # Redder with higher vel

# Draw path to follow
pygame.draw.line(surf, (0, 0, 255), world2screen(np.array([-1000, 0])), world2screen(np.array([1000, 0]))) # Very long path

# Final draw
surf = pygame.transform.flip(surf, False, True) # Flips the surface drawing in Y-axis, so that frame coordinate wise, X is RIGHT, Y is UP in the visualization
screen.blit(surf, (0, 0))
pygame.event.pump()
pygame.display.flip()

print('Press ESC to exit the simulation')
while True:
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        break
print('Exiting the simulation ...')

exit()

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
        self.observation = np.concatenate([pos, vel, acc])

        self.air_vel_ref = np.array([0.0, 0.0])
        self.acc_ff_curvature = 0.0
        self.position_history = None

        # Cache runtime constants
        self.world_width = world_width
        self.color = color

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

        self.air_vel_ref = self.velCurve.calculate_velRef(np.abs(signed_track_error), V_PATH)

        if signed_track_error > 0:
            # NOTE: This only makes sense when path is on X-axis.
            self.air_vel_ref[1] = -self.air_vel_ref[1] # Invert Y-axis value, as positive orthogonal vel in VelCurve means going *towards the path (for now)

        # Kinematics: Assume first order model, perfectly tracking the Vector Field!!
        vel = self.air_vel_ref

        pos += vel * self.env.dt

        self.observation = np.array([pos[0], pos[1], vel[0], vel[1], None, None])

        if self.state_history is not None:
            self.state_history = np.concatenate((self.state_history, [self.observation]), axis=0)
        else:
            self.state_history = np.array([self.observation])

        return None

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
        self.trackRecords.append(TrackRecord(HybridUnicyclicUniform(VEL_RANGE, VEL_RANGE[1]), 'Hybrid Unicyclic', path_bearing_deg, path_curvature, vehicle_speed, world_width, HYBRID_UNICYCLIC_COLOR))
        self.trackRecords.append(TrackRecord(Unicyclic(VEL_RANGE, GROUND_SPEED_DEFAULT, TRACK_KEEPING_SPEED_DEFAULT), 'Unicyclic', path_bearing_deg, path_curvature, vehicle_speed, world_width, UNICYCLIC_COLOR))

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

    def test_env(self):
        ''' Executes the simulation'''
        start_t=timer()

        for i,_ in enumerate(range(int(self.sim_time/SIM_TIME_DT))):
            # Update simulation
            for tr in self.trackRecords:
                tr.update()
            
            # Render
            self.render()

            # # Debug (every 30 frames)
            # if (i % 30 == 0):
            #     print('Pos', pos, 'Vel', vel, 'Vel ref', self.air_vel_ref, 'Acc FF', self.acc_ff_curvature)

            # Take a snapshot & analyze
            if self._stop_every_1_sec:
                if i != 0 and i%100 == 0:
                    pygame.event.pump()
                    input('Input key to simulate 1 second further')

            # if(done):
            #     env.reset()

        end_t=timer()
        print("Simulated time={}s, Computation time={}s".format(SIM_DURATION_SEC, (end_t-start_t)))

        # Visualize state history
        self.draw_state_history()

        # Handle simulation termination
        self.handle_simulation_termination()

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
        PATH_LENGTH_HALF = 2000 # Arbitrary length to extend the path to draw the line in the frame
        path_start_pos = world2screen(self.path_position - self.path_unit_tangent_vec * PATH_LENGTH_HALF, self.world_width)
        path_end_pos = world2screen(self.path_position + self.path_unit_tangent_vec * PATH_LENGTH_HALF, self.world_width)
        
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
        # self.clock.tick(1/SIM_TIME_DT) # Remove clock ticking to reduce simulation time.
        pygame.display.flip()

    def handle_simulation_termination(self):
        ''' Waits for the user to press Escape (ESC) key, and until then keeps pumping the pygame (to prevent 'not responding' error) '''
        print('Press ESC to exit the simulation')
        while True:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                break
        print('Exiting the simulation ...')

    def draw_state_history(self):
        '''
        Draws the simulated state history of different velocity curves
        '''
        # Settings
        MARKER_TYPE = '*'
        TRACK_ERROR_BOUNDARY_STYLE = 'dashed'

        GROUND_TRUTH_MARKER_TYPE = ''
        GROUND_TRUTH_LINE_STYPE = 'dotted'

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('VelCurve Simulation Results: Vpath {} m/s'.format(V_PATH))

        # Create Axes
        ax_V_parallel = fig.add_subplot(4, 2, 1)
        ax_V_orthogonal = fig.add_subplot(4, 2, 2, sharex=ax_V_parallel)

        ax_Acc_parallel = fig.add_subplot(4, 2, 3, sharex=ax_V_parallel)
        ax_Acc_orthogonal = fig.add_subplot(4, 2, 4, sharex=ax_V_parallel)

        ax_norm = fig.add_subplot(4, 2, 5, sharex=ax_V_parallel)
        ax_track = fig.add_subplot(2, 2, 4, sharex=ax_V_parallel)

        # Title and Label settings
        ax_V_parallel.set_title('Velocity parallel to path')
        ax_V_parallel.set_ylabel('V parallel [m/s]')
        ax_V_parallel.set_xlabel('Track error [m]')
        ax_V_parallel.grid()

        ax_V_orthogonal.set_title('Velocity orthogonal to path')
        ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
        ax_V_orthogonal.set_xlabel('Track error [m]')
        ax_V_orthogonal.grid()

        ax_Acc_parallel.set_title('Acceleration parallel to path [m/s^2]')
        ax_Acc_parallel.grid()

        ax_Acc_orthogonal.set_title('Acceleration orthogonal to path [m/s^2]')
        ax_Acc_orthogonal.grid()

        ax_norm.set_title('Velocity norm [m/s]')
        ax_norm.grid()

        ax_track.set_title('Path drawn by Vel Curves')
        ax_track.grid()

        # Put data
        for tr in self.trackRecords:
            curve_name = tr.get_name()
            states = tr.get_state_history()
            velCurveObj = tr.get_velCurve_object()
            color = tr.get_color()

            pos_array = None
            vel_array = None
            acc_array = None

            for state in states:
                pos, vel, acc = MCPointMass.decode_state(MCPointMass, state)

                if pos_array is None or vel_array is None or acc_array is None:
                    pos_array = np.array([pos])
                    vel_array = np.array([vel])
                    acc_array = np.array([acc])
                else:
                    pos_array = np.concatenate((pos_array, [pos]))
                    vel_array = np.concatenate((vel_array, [vel]))
                    acc_array = np.concatenate((acc_array, [acc]))

            # Post processing
            track_error_array = np.cross(self.path_unit_tangent_vec, pos_array - self.path_position)

            # Draw Curves
            ax_V_parallel.plot(track_error_array, vel_array[:, 0], label=curve_name, marker=MARKER_TYPE, color=color)
            # Invert orth vel, as the sign convention is different
            ax_V_orthogonal.plot(track_error_array, -vel_array[:, 1], label=curve_name, marker=MARKER_TYPE, color=color)

            ax_Acc_parallel.plot(track_error_array, acc_array[:, 0], label=curve_name, marker=MARKER_TYPE, color=color)
            ax_Acc_orthogonal.plot(track_error_array, -acc_array[:, 1], label=curve_name, marker=MARKER_TYPE, color=color)

            ax_norm.plot(track_error_array, np.linalg.norm(vel_array, axis=1), label=curve_name, marker=MARKER_TYPE, color=color)

            # Draw Track Error Boundaries
            ax_V_parallel.axvline(velCurveObj.get_track_error_boundary(), ymin=np.min(track_error_array), ymax=np.max(track_error_array), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)
            ax_V_orthogonal.axvline(velCurveObj.get_track_error_boundary(), ymin=np.min(track_error_array), ymax=np.max(track_error_array), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

            # Draw Ground Truth curve too
            track_error_array = track_error_array[track_error_array >= 0] # Purify track error to only include positive errors
            vel_data = velCurveObj.calculate_velRef_array(track_error_array, V_PATH)
            ax_V_parallel.plot(track_error_array, vel_data[0], marker=GROUND_TRUTH_MARKER_TYPE, linestyle=GROUND_TRUTH_LINE_STYPE, color=color)
            ax_V_orthogonal.plot(track_error_array, vel_data[1], marker=GROUND_TRUTH_MARKER_TYPE, linestyle=GROUND_TRUTH_LINE_STYPE, color=color)
            ax_norm.plot(track_error_array, np.sqrt(np.square(vel_data[0])+np.square(vel_data[1])), marker=GROUND_TRUTH_MARKER_TYPE, linestyle=GROUND_TRUTH_LINE_STYPE, color=color)

            # Draw Path
            # X: track error (Y value), Y: path position (X value)
            ax_track.plot(pos_array[:, 1], pos_array[:, 0], label=curve_name, marker=MARKER_TYPE, color=color)

        # Legend: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        ax_V_parallel.legend(loc='upper right')
        ax_V_orthogonal.legend(loc='upper right')
        ax_norm.legend(loc='upper right')

        # Visualize the plot
        fig.tight_layout()
        plt.show()

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
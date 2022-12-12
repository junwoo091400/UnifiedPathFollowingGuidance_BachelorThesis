

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

import math
from typing import Optional
import math

import numpy as np

import gym
from gym.spaces import Dict, Box
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

from windywings.logger.default_logger import Logger
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Dirty import of NPFG library
from windywings.npfg.npfg import NPFG

# Dirty copy of FWLateral to accompany NPFG details (e.g. Path info)
class FWLateralNPFG(gym.Env):
    """
    # Description

    Environment to simulate fixed wing longitudinal dynamics

    # Axes
    
    The XYZ for the position uses a Right-hand coordinate system in East-North-Up schema
    Meaning that Z-axis points upwards, and heading is 0 when vehicle faces Eastwards.
    
    Note that this is different from a conventional North-East-Down aviation frame!!

    # Observation Space

    The observation is a `ndarray` with shape `(5,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Position of the plane (x)            | -Inf | Inf | position (m)   |
    | 1   | Position of the plane (y)            | -Inf | Inf | position (m)   |
    | 2   | Speed of the plane (x)               | 0.0  | Inf | velocity (m/s) |
    | 3   | Heading of the plane (rad)           | -PI  | PI  | rad            |
    | 4   | Current navigating path setpoint (x) | -Inf | Inf | position (m)   |
    | 5   | Current navigating path setpoint (y) | -Inf | Inf | position (m)   |
    | 6   | Heading of the path segment (rad)    | -PI  | PI  | rad            |
    | 7   | Curvature of the path segment        | 0.0  | Inf | curvature (1/m)|

    NOTE: Heading is defined to be 0 when aligned with X-axis, and PI/2 when aligned with Y-axis.

    # Action Space

    The action is a `ndarray` with shape `(2,)`, representing the aileron and throttle inputs.

    | Num | Action                               | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Longitudial acceleration             | 0    | 1   | m / s^2        |
    | 1   | Lateral acceleration                 | -9.0 | 9.0 | m / s^2        |

    NOTE: Lateral acceleration range is limited by coordinated-turn logic (lateral_acceleration = G * tan(roll_angle)). We assume max roll of ~45 degrees.

    # Transition Dynamics:

    Given an action, the plane is following transition dynamics:

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub> * dt*

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + throttle (cos(heading<sub>t</sub>), sin(heading<sub>t</sub>))* dt*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    # Reward


    # Starting State

    The position of the plane is assigned a uniform random value in `[0.0. , 0.0]`.
    The starting velocity of the plane is always assigned to `[15.0. , 0.0]`.

    # Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.

    # Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(self, screen_size = [1200, 1200], world_size = 100., render_mode: Optional[str] = None, DEBUG = False):
        """ Initialize the Environment (Screen size: Actual display size, World size: size of the world that will be self.world_to_screen_scalingd & projected onto the screen) """
        assert len(screen_size) == 2, "Screen size [Width, Height] not an array with length 2!"
        assert world_size > 0, "World size (width & height) must be a positive number!"

        # Open AI Gym related parameters (rendering)
        self.render_mode = render_mode
        self.screen_width, self.screen_height = screen_size
        self.world_size = world_size
        self.world_to_screen_scaling = np.max(screen_size)/world_size # We scale so that we fit the world to the maximum height/width, so unless square shaped, the world will be cut
        self.screen = None
        self.clock = None
        self.isopen = True

        # Vehicle dynamic configuration parameters
        self.dt = 0.03
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -world_size/2
        self.max_position = world_size/2
        self.max_speed = 50.0
        self.min_speed = 0.0
        self.max_acceleration = 1.0
        self.gravity = np.array([0.0, -9.81])

        # these are specifically for verbosely setting NPFG parameters
        self.airspeed_nom = 15.0
        self.airspeed_ref = 15.0 # NOTE: This can be set for 0 for multicopters, which can achieve 0 forward speed!

        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_speed, -np.pi, self.min_position, self.min_position, -np.pi, 0.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_speed, np.pi, self.max_position, self.max_position, np.pi, np.inf], dtype=np.float32
        )

        # NPFG instance
        self._npfg = NPFG(self.airspeed_nom, self.airspeed_ref)

        # Runtime variables (not part of state, but used for calculations)
        self.longitudinal_acceleration = 0.0
        self._debug_enable = DEBUG
        self._vehicle_time = 0.0 # Integral with the `self.dt`. Time that vehicle 'feels'.

        # Pyplot plotting
        # self.fig, self.ax = plt.subplots()
        # self.ln, = self.ax.plot([], [], 'ro')
        # self.fig_animation = FuncAnimation(self.fig, self.pyplot_animation, blit=True, interval=1000//self.metadata["render_fps"])
        # plt.show() # This is blocking. Need to get around that: https://matplotlib.org/stable/users/explain/interactive_guide.html

        # Statistics
        self.position_history = None

        # GYM internal variables
        self._action = None # Cache of the latest `action` command applied in `step` functino
        
        # Action space
        self.action_space = Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

        # Observation space: Same as `state` space
        self.observation_space = Box(
            low=self.low_state, high=self.high_state, shape=(8, )
        )

    def set_path(self, pos_setpoint, path_heading, path_curvature):
        ''' Sets a path to follow, which allows drawing the desired path (control isn't done inside the environment) '''
        self.state[4:6] = pos_setpoint
        self.state[7] = path_heading
        self.state[8] = path_curvature

    def step(self, action: np.ndarray):
        # Action info caching
        self._action = action

        # Advance vehicle time
        self._vehicle_time += self.dt
        
        # State retrieval
        position = self.state[0:2]  # position
        velocity = self.state[2]    # velocity
        heading = self.state[3]     # heading
        path_pos = self.state[4:6]
        path_bearing = self.state[6]
        path_curvature = self.state[7]

        lon_accel_cmd = action[0]
        lat_accel_cmd = action[1]

        self.longitudinal_acceleration = self.max_acceleration * lon_accel_cmd

        position = position + velocity * np.array([np.math.cos(heading), np.math.sin(heading)]) * self.dt
        # NOTE: Longitudinal acceleration setpoint gets instantaneously applied, and it doesn't affect yaw rate!
        velocity = velocity + self.longitudinal_acceleration * self.dt

        # 'Lateral acceleration' in the end results in a yaw-rate. Which has a following dynamic:
        # lat_acc = velocity * yaw_rate. Therefore yaw_rate = (lat_acc / velocity)!
        yawrate_cmd = lat_accel_cmd / velocity

        # NOTE: Yaw rate command gets instantaneously applied in global frame.
        heading = heading + yawrate_cmd * self.dt

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            velocity <= 0.0
        )

        reward = 0
        
        if terminated:
            reward = 0.0
        
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position[0], position[1], velocity, heading, path_pos[0], path_pos[1], path_bearing, path_curvature], dtype=np.float32)

        # Debug Output
        # if (self._debug_enable):
        #     # print('Debug, state: {}'.format(self.state))
        #     print('Position history shape: {}'.format(np.shape(self.position_history)))

        # Save Position history
        if self.position_history is not None:
            self.position_history = np.append(self.position_history, [[position[0], position[1]]], axis=0)
        else:
            self.position_history = np.array([[position[0], position[1]]])

        if self.render_mode == "human":
            self.render()

        # Observation, Reward, Done, Info: https://www.gymlibrary.dev/content/environment_creation/#step
        return self._get_obs(), reward, terminated, self._get_info()

    def control(self):
        ''' Return the control `action` vector with current vehicle state. Handles calculation using NPFG logic '''
        # Retrieve state vector info
        position = self.state[0:2]  # position
        speed = self.state[2]
        heading = self.state[3]
        path_pos = self.state[4:6]
        path_bearing = self.state[6]
        path_curvature = self.state[7]

        # Additional calculations
        path_unit_tangent_vector = np.array([np.cos(path_bearing), np.sin(path_bearing)])
        ground_vel = speed * np.array([np.cos(heading), np.sin(heading)])

        # NPFG logic
        lateral_acc_cmd = self._npfg.navigatePathTangent_nowind(position, path_pos, path_unit_tangent_vector, ground_vel, path_curvature)

        # Return the `action` vector
        # NOTE: Keep longitudinal acc cmd at 0 for now.
        return np.array([0.0, lateral_acc_cmd])
    
    def pyplot_animation(self, frame):
        ''' Boilerplate code for live-updating Pyplot to show debug values (e.g. NPFG) '''
        print('Animation: frame {}, time {}'.format(frame, self._vehicle_time))

    def _get_obs(self):
        """ Get Observation from internal state """
        return self.state

    def _get_info(self):
        """ Get Information from internal state """
        return {'linX': self.longitudinal_acceleration, 'linY': 0.0, 'angZ': 0.0}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              initial_state: Optional[np.ndarray] = [0.0, 0.0, 15.0, 0.0, np.nan, np.nan, np.nan, np.nan]):
        ''' Resets the environment '''
        super().reset(seed=seed)

        # Reset state = [PosX: 0, PosY: 0, Speed: 15, Heading: 0, PathX: nan, PathY: nan, PathHeading: nan, PathCurvature: nan]
        self.state = np.array(initial_state, dtype=np.float32)
        print('Reset: Setting state to: {}'.format(self.state))

        # Reset Statistics
        self.position_history = None
        self._vehicle_time = 0.0

        if self.render_mode == "human":
            self.render()
        
        return (self._get_obs(), self._get_info())

    def _height(self, xs):
        return 0.0 * xs

    def world2screen(self, coords: np.array):
        ''' Converts the world coordinate (in meters) to a pixel location in Rendering screen '''
        assert np.shape(coords) == (2, ), "Coordinates to transform in World2Screen must be of size (2, )!"
        # We want to place the (0.0, 0.0) in the middle of the screen, therefore the transform is:
        # (X): Screen-size/2 + coordinateX * scaling
        return np.array([self.screen_width/2, self.screen_height/2]) + coords * self.world_to_screen_scaling

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Import Pygame
        try:
            import pygame
            from pygame import gfxdraw

        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        # Construct screen
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption('Fixed Wing Lateral Acc NPFG Environment')
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        
        # Construct clock
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Constants
        carlength = 20
        carwidth = 10

        # Pygame surface construction
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # State retrieval
        vehicle_pos = self.world2screen(self.state[0:2])
        vehicle_yaw = self.state[3]

        # Draw Position History
        if (self.position_history is not None) and (self.position_history.shape[0] > 1):
            xys = list(zip(map(self.world2screen, self.position_history)))
            pygame.draw.aalines(self.surf, points=xys, closed=False, color=(120, 120, 255))

        # Draw the Vehicle
        r, t, b = carlength, 0.5 * carwidth, -0.5 * carwidth
        coords = []
        posScreen = vehicle_pos # Convert to screen pixel scale
        for c in [(0, t), (0, b), (r, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(vehicle_yaw)
            coords.append(
                (
                    c[0] + posScreen[0],
                    c[1] + posScreen[1]
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        # Draw the path target
        path_pos = self.state[4:6]
        path_bearing = self.state[6]
        path_curvature = self.state[7]
        path_unit_tangent_vector = np.array([np.cos(path_bearing), np.sin(path_bearing)])

        PATH_LENGTH_HALF = 2000 # Arbitrary length to extend the path to draw the line in the frame
        path_start_pos = self.world2screen(path_pos - path_unit_tangent_vector * PATH_LENGTH_HALF)
        path_end_pos = self.world2screen(path_pos + path_unit_tangent_vector * PATH_LENGTH_HALF)
        
        if (np.isfinite(path_start_pos).all() and np.isfinite(path_end_pos).all()):
            # Only draw when the coordinates are finite (defined)
            try:
                pygame.draw.line(self.surf, (255, 0, 0), path_start_pos, path_end_pos)
            except Exception as e:
                print(e)
                print('Path start: {}, Path end: {}'.format(path_start_pos, path_end_pos))

        # Draw NPFG internal calculations
        # Closest point on path (connect with vehicle)
        closest_point_on_path = self.world2screen(self._npfg.d_closest_point_on_path)
        pygame.draw.line(self.surf, pygame.Color('grey'), vehicle_pos, closest_point_on_path)
        # Unit path tangent vector
        UPT_LENGTH = 20 # Multiplier on the length of the unit path tangent vector to draw
        pygame.draw.line(self.surf, pygame.Color('green'), closest_point_on_path, closest_point_on_path + self._npfg.d_unit_path_tangent * self.world_to_screen_scaling * UPT_LENGTH) # Manually scale the vector
        # Track error bound
        TRACK_ERROR_BOUND_WIDTH = 1 # Thickness of the circle visualizing track error bound
        pygame.draw.circle(self.surf, pygame.Color('purple'), closest_point_on_path, self._npfg.d_track_error_bound * self.world_to_screen_scaling, width=TRACK_ERROR_BOUND_WIDTH) # Set width, to not fill the circle
        # Bearing setpoint from look-ahead-angle
        BEARING_LENGTH = 60 # Multiplier on the length of the unit path tangent vector to draw
        BEARING_WIDTH = 3
        pygame.draw.line(self.surf, pygame.Color('pink'), vehicle_pos, vehicle_pos + self._npfg.d_bearing_vector * self.world_to_screen_scaling * BEARING_LENGTH, width=BEARING_WIDTH) # Manually scale the vector

        # Drawing the diagram & flipping Y axis
        self.surf = pygame.transform.flip(self.surf, False, True) # Flips the surface drawing in Y-axis, so that frame coordinate wise, X is RIGHT, Y is UP in the visualization
        self.screen.blit(self.surf, (0, 0))

        # Draw debug info
        debug_text = ''
        # current_time = pygame.time.get_ticks() / 1000.
        current_time = self._vehicle_time
        debug_text += ('T: {:.2f} '.format(current_time))
        if self._action is not None:
            lat_accel_cmd = self._action[1]
            debug_text += ('Acc: {:.1f} '.format(lat_accel_cmd))
        debug_text += 'tE: {:.1f} m '.format(self._npfg.d_signed_track_error)
        debug_text += 'te: {:.2f} '.format(self._npfg.d_normalized_track_error)
        debug_text += 'tp: {:.2f} '.format(self._npfg.d_track_proximity)
        debug_text += 'at: {:.2f} '.format(self._npfg.d_lateral_accel_no_curve)
        debug_text += 'ac: {:.2f} '.format(self._npfg.d_lateral_accel_ff_curve)

        debug_font = pygame.font.SysFont(None, 24)
        debug_img = debug_font.render(debug_text, True, (0, 0, 0))
        self.screen.blit(debug_img, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



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
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

from windywings.logger.default_logger import Logger
from matplotlib import pyplot as plt


class FWLateral(gym.Env):
    """
    # Description

    Environment to simulate fixed wing longitudinal dynamics

    # Observation Space

    The observation is a `ndarray` with shape `(5,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Position of the plane (x)            | -Inf | Inf | position (m)   |
    | 1   | Position of the plane (y)            | -Inf | Inf | position (m)   |
    | 2   | Speed of the plane (x)               | 0.0  | Inf | velocity (m/s) |
    | 3   | Heading of the plane (rad)           | -PI  | PI  | rad            |

    # Action Space

    The action is a `ndarray` with shape `(2,)`, representing the aileron and throttle inputs.

    | Num | Action                               | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Longitudial acceleration             | 0    | 1   | m / s^2        |
    | 1   | Yaw rate                             | -1   | 1   | rad / s        |

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

    # Arguments

    ```
    gym.make('fixedwing-longituninal')
    ```

    # Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        # Angle of attack at stall
        self.dt = 0.03
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -100.0
        self.max_position = 100.0
        self.max_speed = 50.0
        self.min_speed = 0.0
        self.max_acceleration = 1.0

        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_speed, self.min_speed, -3.14, -100.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_speed, self.max_speed, 3.14, 100.0], dtype=np.float32
        )

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.gravity = np.array([0.0, -9.81])

        ##TODO: Hack! Initialize position history properly
        self.position_history = [[0.0, 0.0]]
        self.position_history = np.append(self.position_history, [[0.0, 1.0]], axis=0)

    def step(self, action: np.ndarray):
        position = self.state[0:2]  # position
        velocity = self.state[2]  # velocity
        heading = self.state[3]     # heading

        acceleration_cmd = action[0]
        yawrate_cmd = action[1]

        acceleration = self.max_acceleration * acceleration_cmd

        position = position + velocity * np.array([np.math.cos(heading), np.math.sin(heading)]) * self.dt
        velocity = velocity + acceleration * self.dt
        heading = heading + yawrate_cmd * self.dt

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            velocity <= 0.0
        )

        reward = 0
        if terminated:
            reward = 0.0
        reward -= math.pow(action[0], 2) * 0.1
        self.state = np.array([position[0], position[1], velocity, heading], dtype=np.float32)
        self.position_history = np.append(self.position_history, [[position[0], position[1]]], axis=0)
        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, \
            {'linX': acceleration, 'linY': 0.0, 'angZ': 0.0}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              initial_state: Optional[np.ndarray] = [0.0, 0, 15.0, 0.0, 0.0, 0.0]):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array(initial_state)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return 0.0 * xs

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw

        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carlength = 20
        carwidth = 10

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = np.array([self.state[0], self.state[1]])
        yaw = self.state[3]

        clearance = 200
        if self.position_history.shape[1] > 1:
            xys = list(zip((self.position_history[:, 0] + 50.0) * scale, self.position_history[:, 1] * scale + clearance))
            pygame.draw.aalines(self.surf, points=xys, closed=False, color=(120, 120, 255))

        r, t, b = carlength, 0.5 * carwidth, -0.5 * carwidth
        coords = []
        for c in [(0, t), (0, b), (r, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(yaw)
            coords.append(
                (
                    c[0] + (pos[0] + 50.0) * scale,
                    c[1] + clearance + pos[1] * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def control():
        return

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

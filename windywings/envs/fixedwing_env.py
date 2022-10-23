

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

from cProfile import label
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


class FWLongitudinal(gym.Env):
    """
    ### Description

    Environment to simulate fixed wing longitudinal dynamics

    ### Observation Space

    The observation is a `ndarray` with shape `(3,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | position of the plane (horizontal)   | -Inf | Inf | position (m)   |
    | 1   | position of the plane (altitude)     | -Inf | Inf | position (m)   |
    | 2   | horizontal velocity of the plane     | -Inf | Inf | velocity (m/s) |
    | 3   | vertical   velocity of the plane     | -Inf | Inf | velocity (m/s) |
    | 4   | pitch of the plane                   | -PI  | PI  | rad            |
    | 5   | pitch rate of the plane              | -Inf | Inf | rad/s          |

    ### Action Space

    The action is a `ndarray` with shape `(2,)`, representing the aileron and throttle inputs.
    The action is clipped in the range `[-1,1]`

    | Num | Action                               | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | throttle of the plane                | 0    | 1   | setting        |
    | 1   | elevator position of the plane       | -1   | 1   | setting        |

    ### Transition Dynamics:

    Given an action, the plane is following transition dynamics:

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub> * dt*

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + throttle * self.cf + LiftDrag*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward


    ### Starting State

    The position of the plane is assigned a uniform random value in `[0.0. , 0.0]`.
    The starting velocity of the plane is always assigned to `[15.0. , 0.0]`.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.

    ### Arguments

    ```
    gym.make('fixedwing-longituninal')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        # TODO: adapt these values according to vehicle properties
        self.mass = 20.0     # Mass of the vehicle
        self.gravity = 9.81  # Gravity
        self.chord = 0.5     # Chord length
        self.span = 4.0      # Span length
        self.rho = 1.225     # Air density
        self.Iyy = 20.0      # Moment of inertia
        self.Sref = self.chord * self.span  # Reference area
        self.ar = self.span**2 / self.Sref  # Aspect ratio

        # TODO: determine these values through system identification
        self.cm0 = 0.01           # Moment coefficient at zero angle of attack
        self.cmalpha = -0.487     # Moment coefficient slope
        self.cmdelta = -0.2       # Moment coefficient slope with elevator deflection
        self.cmq = -13.8          # Moment damping coefficient
        self.cl0 = 0.2            # Lift coefficient at zero angle of attack
        self.clalpha = 2 * np.pi  # Lift coefficient slope (ideally 2*pi)
        self.cl_stall = 0.7       # Lift coefficient at stall
        self.cldelta = 0.1        # Lift coefficient slope with elevator deflection
        self.cd0 = 0.03           # Drag coefficient at zero angle of attack
        self.cf = 50000           # Thrust coefficient
        self.alpha_stall = round(20 / 180.0 * np.pi,3)     # Angle of attack at stall

        self.dt = 0.03
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -100.0
        self.max_position = 100.0
        self.max_speed = 50.0
        self.min_speed = 0.0
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_speed, self.min_speed, -3.14, -100.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_speed, self.max_speed, 3.14, 100.0], dtype=np.float32
        )

        self.render_mode = render_mode

        self.screen_width = 1000
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

    def force_liftdrag(self, state, action):
        # total velocity from horizontal and vertical component
        v_total = np.sqrt(state[2]**2 + state[3]**2)

        # compute angle of attack = pitch angle - flight path angle
        gamma = np.arctan2(state[3], state[2])
        alpha = state[4] - gamma

        # compute lift coefficient
        # TODO: decide whether to use the linear or more complex approach to model lift characteristics
        # cl = self.cl0 + self.clalpha * alpha + self.cldelta * action
        # non-linear approach including simple stall model
        if (alpha < self.alpha_stall - 5.0 / 180.0 * np.pi).all():
            cl = self.cl0 + self.clalpha * \
                alpha + self.cldelta * action
        elif (alpha > self.alpha_stall - 5.0 / 180.0 * np.pi and alpha < self.alpha_stall + 5.0 / 180.0 * np.pi).all():
            # sample a sigmoid functino for the transition, assuming 10 degrees width for the transition
            sigmoid_sample = self.sample_sigmoid(self.alpha_stall, 10.0 / 180.0 * np.pi, alpha)
            cl = (self.cl0 + self.clalpha * alpha + self.cldelta * action) * \
                (1 - sigmoid_sample) + self.cl_stall * sigmoid_sample
        else:
            cl = self.cl_stall

        # compute drag coefficient
        cd = self.cd0 + cl**2 / (np.pi * np.e * self.ar)

        # compute lift and drag forces
        lift = 0.5 * self.rho * v_total**2 * self.Sref * cl
        drag = 0.5 * self.rho * v_total**2 * self.Sref * cd

        # compute forces in x and z direction in earth fixed NED frame
        # signs are chosen keeping in mind that the flight path angle is negative in gliding flight
        fx = - lift * np.sin(gamma) - drag * np.cos(gamma)
        fz = - lift * np.cos(gamma) + drag * np.sin(gamma)

        return np.array([fx, - fz])

    def demo_plot_cl_alpha(self):
        # set these parameters to change the cl vs alpha plot
        steps = 900
        alpha_min_deg = -30.0
        alpha_max_deg = 40.0

        alphas = np.linspace(alpha_min_deg / 180.0 * np.pi,
                            alpha_max_deg / 180.0 * np.pi, steps)
        action = np.linspace(1.0, -1.0, steps)

        cl = np.linspace(0.0, 0.0, 900)
        for idx, alpha in enumerate(alphas):
            if (alpha < self.alpha_stall - 5.0 / 180.0 * np.pi).all():
                cl[idx] = self.cl0 + self.clalpha * \
                    alpha + self.cldelta * action[idx]
            elif (alpha > self.alpha_stall - 5.0 / 180.0 * np.pi and alpha < self.alpha_stall + 5.0 / 180.0 * np.pi).all():
                sigmoid_sample = self.sample_sigmoid(self.alpha_stall, 10.0 / 180.0 * np.pi, alpha)
                cl[idx] = (self.cl0 + self.clalpha * alpha + self.cldelta * action[idx]) * \
                    (1 - sigmoid_sample) + self.cl_stall * sigmoid_sample
            else:
                cl[idx] = self.cl_stall

        plt.figure('Demo non-linear AoA vs CL')
        plt.plot(alphas, cl)
        plt.xlabel(r'Angle of attack $\alpha$ [rad]')
        plt.ylabel(r'Lift coefficient $c_L$')
        plt.show()

    def sample_sigmoid(self, center: float, width: float, sample: float):
        # rounding is required for consistent array lengths, trade-off between accuracy and computation speed
        # rounding the values to 3 decimals still results in reasonable accuracy
        rounded_width = round(width, 4)
        rounded_sample = round(sample, 4)
        rounded_center = round(center, 4)

        # compute the index of the sample in the discretized sigmoid function
        idx = int(round((rounded_sample - rounded_center + rounded_width / 2) * 10000, 0))

        # compute the argument of the sigmoid function assuming a clipped version in [-6.0, 6.0]
        sigmoid_arg = idx / (math.ceil(rounded_width / 0.0001) + 1) * 12.0 - 6.0

        # return the value of the scaled sigmoid function
        return 1.0 / (1.0 + np.exp(- sigmoid_arg))

    def force_thrust(self, state, action):
        gamma = np.arctan2(state[3], state[2])

        # compute thrust force
        thrust = self.power * action * self.cf

        # split the force into x and z components in earth fixed NED frame
        thrustX = thrust * np.cos(gamma)
        thrustZ = - thrust * np.sin(gamma)

        return np.array([thrustX, - thrustZ])

    def moment_liftdrag(self, state, action):
        # total velocity from horizontal and vertical component
        v_total = np.sqrt(state[2]**2 + state[3]**2)

        # compute moment coefficient
        cm = self.cm0 + self.cmalpha * state[4] + self.cmdelta * action[1] + \
            (state[5] * self.chord / (2 * v_total)) * self.cmq

        # compute moment
        moment = 0.5 * self.rho * v_total**2 * self.Sref * self.chord * cm

        return moment

    def visualize_results(paths, variablesX, variablesY, invertedY=False):
        for idx, path in enumerate(paths):
            data = Logger.get_data(path)
            plt.figure(idx + 1)
            plt.plot(data[variablesX[idx]], data[variablesY[idx]])
            plt.xlabel(variablesX[idx])
            plt.ylabel(variablesY[idx])

            if invertedY:
                plt.gca().invert_yaxis()

        plt.show()

    def step(self, action: np.ndarray):
        position = self.state[0:2]  # position
        velocity = self.state[2:4]  # velocity

        acceleration = (1/self.mass) * self.force_liftdrag(self.state, action[1]) \
            + (1/self.mass) * self.force_thrust(self.state, action[0]) \
            + self.gravity

        position = position + velocity * self.dt
        velocity = velocity + acceleration * self.dt

        pitch = self.state[4]
        pitch_rate = self.state[5]

        pitch = pitch + pitch_rate * self.dt
        moment = self.moment_liftdrag(self.state, action)
        pitch_rate = pitch_rate + (moment / self.Iyy) * self.dt

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            velocity[0] <= 0.0
        )

        reward = 0
        if terminated:
            reward = 0.0
        reward -= math.pow(action[0], 2) * 0.1
        self.state = np.array([position[0], position[1], velocity[0],
                              velocity[1], pitch, pitch_rate], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {'linX': acceleration[0],
                                                       'linZ': - acceleration[1], 'angY': moment / self.Iyy}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([0.0, 0, self.np_random.uniform(low=10.0, high=20.0),
                               self.np_random.uniform(low=0.0, high=5.0),
                               self.np_random.uniform(low=-0.2, high=0.3),
                               self.np_random.uniform(low=-0.1, high=0.1)])

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
        carwidth = 80
        carheight = 10

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = np.array([self.state[0], self.state[1]])
        pitch = self.state[4]

        clearance = 200

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t+t), (0.7 * l, t+t), (0.6 * l, t), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(pitch)
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

    def control(self, control='ramp_elevator', start_value=-1.0, transition_step=200,
                end_value=1.0, steps=800, fixed_value=0.0, iteration=0):

        if control == 'ramp_elevator':
            if iteration < transition_step:
                return np.array([fixed_value, start_value])
            else:
                return np.array([fixed_value, (iteration - transition_step) / (steps - transition_step) *
                                 (end_value - start_value) + start_value])

        if control == 'ramp_thrust':
            if iteration < transition_step:
                return np.array([start_value, fixed_value])
            else:
                return np.array([(iteration - transition_step) / (steps - transition_step) *
                                 (end_value - start_value) + start_value, fixed_value])

        return np.array([0.0, 0.0])

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

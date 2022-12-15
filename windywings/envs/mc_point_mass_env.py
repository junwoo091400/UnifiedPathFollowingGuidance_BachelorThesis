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

from typing import Optional
import numpy as np

import gym
from gym.spaces import Box

# Helper constants to do assertions. Make sure this matches the description below!
ACTION_SPACE_SHAPE = (4,)
STATE_SPACE_SHAPE = (6,)

class MCPointMass(gym.Env):
    """
    # Description

    Environment to simulate  Multicopter dynamics as a point-mass (no heading/orientation of the body), with a P controller for following velocity setpoint.

    For simplicity, render function isn't implemented.

    # Frame of Reference

    There is no body frame, we only consider the global frame, and consider multicopter as a point mass.

    ---
    X: East
    Y: North
    Z: Up

    # Environment

    Multicopter is simulated inside a square-shaped environment with the specified width, referenced as 'world_size'

    # Observation Space (Equals to State for simplicity)

    The observation is a `ndarray` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Position of the multicopter, global frame (x)            | -Inf | Inf | position (m)   |
    | 1   | Position of the multicopter, global frame (y)            | -Inf | Inf | position (m)   |
    | 2   | Velocity of the multicopter, global frame (x)               | -Inf  | Inf | velocity (m/s) |
    | 3   | Velocity of the multicopter, global frame (y)           | -Inf  | Inf  | velocity (m/s)    |
    | 4   | Acceleration of the multicopter (x), global frame             | -Inf  | Inf  | acceleration (m/s^2)    |
    | 5   | Acceleration of the multicopter (y), global frame             | -Inf  | Inf  | acceleration (m/s^2)    |

    # Action Space

    The action is a `ndarray` where the elements correspond to the following:

    | Num | Action                               | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Velocity setpoint (x), global frame             | -Inf  | Inf  | velocity (m/s)    |
    | 1   | Velocity setpoint (y), global frame             | -Inf  | Inf  | velocity (m/s)    |
    | 2   | Acceleration Feed-forward (x), global frame             | -Inf  | Inf  | acceleration (m/s^2)    |
    | 3   | Acceleration Feed-forward (y), global frame             | -Inf  | Inf  | acceleration (m/s^2)    |

    # Transition Dynamics:

    Given an action, multicopter is following transition dynamics:

    Position += Velocity * dt
    Velocity += Acceleration * dt

    Acceleration_feedback = P_gain * (Velocity_setpoint - Velocity)
    Acceleration = Acceleration_feedback + Acceleration_feedforward

    # Starting State

    If not specified, the multicopter starts at (-world_size/2, 0), the left-most point of the environment.
    With the velocity of (max_velocity/2, 0)

    # Episode End

    The episode ends if either of the following happens:
    1. Over the boundary: Multicopter position is out of the world (world_size x world_size)

    # Arguments

    ```
    gym.make('multicopter-pointmass')
    ```
    """

    # Metadata of the environment
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def _get_obs(self):
        """ Get Observation from internal state """
        return self.state

    def _get_info(self):
        """ Get Information from internal state """
        return { }

    def _is_terminated(self):
        """ Get Termination state from internal state """

    def __init__(self, render_mode: Optional[str] = None, world_size = 200.0, velocity_bounds = np.array([-10.0, 10.0]), acceleration_bounds = np.array([-9.0, 9.0])):
        """ Initialize the Environment """
        self.dt = 0.03
        self.world_size = world_size
        self.min_position, self.max_position = -world_size/2, world_size/2
        self.min_velocity, self.max_velocity = velocity_bounds
        self.min_acceleration,self.max_acceleration = acceleration_bounds

        # Velocity controller
        self.p_vel = 1.0

        # Action space
        self.min_action = np.array(
            [self.min_velocity, self.min_velocity, self.min_acceleration, self.min_acceleration]
        )
        self.max_action = np.array(
            [self.max_velocity, self.max_velocity, self.max_acceleration, self.max_acceleration]
        )
        self.action_space = Box(
            low=self.min_action, high=self.max_action, shape=ACTION_SPACE_SHAPE
        )

        # Observation space (equals state space)
        self.min_state = np.array(
            [self.min_position, self.min_position, self.min_velocity, self.min_velocity, self.min_acceleration, self.min_acceleration]
        )
        self.max_state = np.array(
            [self.max_position, self.max_position, self.max_velocity, self.max_velocity, self.max_acceleration, self.max_acceleration]
        )
        self.observation_space = Box(
            low=self.min_state, high=self.max_state, shape=STATE_SPACE_SHAPE
        )
    
    def decode_state(self, state):
        """ Decodes the state into discrete values """
        assert np.shape(state) == STATE_SPACE_SHAPE
        pos = state[0:2]
        vel = state[2:4]
        acc = state[4:6]
        
        return (pos, vel, acc)

    def decode_action(self, action=np.ndarray):
        ''' Decode action into discrete values. '''
        assert np.shape(action) == ACTION_SPACE_SHAPE
        vel_sp = action[0:2]
        acc_ff = action[2:4]
        
        return (vel_sp, acc_ff)

    def step(self, action: np.ndarray):
        assert(np.shape(action) == ACTION_SPACE_SHAPE)

        pos, vel, acc = self.decode_state(self.state)
        vel_sp, acc_ff = self.decode_action(action)

        # Integrate state
        pos += vel * self.dt
        vel += acc * self.dt

        # Apply action
        acc_fb = (vel_sp - vel) * self.p_vel
        acc = acc_fb + acc_ff

        # Set new state
        self.state = np.concatenate((pos, vel, acc), axis=None)
        assert np.shape(self.state) == STATE_SPACE_SHAPE

        # Observation, Reward, Done, Info: https://www.gymlibrary.dev/content/environment_creation/#step
        return self._get_obs(), 0.0, self._is_terminated(), self._get_info()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              initial_state: Optional[np.ndarray] = None):
        ''' Reset the environment '''
        super().reset(seed=seed)

        if (not isinstance(initial_state, np.ndarray)):
            # If not specified, initialize as described in the description
            initial_pos = np.array([-self.world_size/2, 0.0])
            initial_vel = np.array([self.max_velocity/2, 0.0])
            initial_acc = np.array([0.0, 0.0])
            initial_state = np.concatenate((initial_pos, initial_vel, initial_acc), axis=None)
        
        assert np.shape(initial_state) == STATE_SPACE_SHAPE
        self.state = initial_state

        print('Reset: State set to:', self.state)
        
        return (self._get_obs(), self._get_info())

    def render(self):
        """ Render function not implemented """
        return None

    def control(self):
        """ Control function not implemented """
        return None

    def close(self):
        """ Close function not implemented """
        return None

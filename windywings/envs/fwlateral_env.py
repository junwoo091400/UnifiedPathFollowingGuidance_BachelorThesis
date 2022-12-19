

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

# Helper constants to do assertions. Make sure this matches the description below!
ACTION_SPACE_SHAPE = (2,)
STATE_SPACE_SHAPE = (4,)

# Default vehicle constraints
AIRSPEED_MIN_DEFAULT = 5.0
AIRSPEED_MAX_DEFAULT = 15.0

class FWLateral(gym.Env):
    """
    # Description

    Environment to simulate fixed wing lateral dynamics

    # Observation Space

    The observation is a `ndarray` with shape `(5,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Position of the plane, global frame (x)            | -Inf | Inf | position (m)   |
    | 1   | Position of the plane, global frame (y)            | -Inf | Inf | position (m)   |
    | 2   | Speed of the plane, body frame (x)               | 0.0  | Inf | velocity (m/s) |
    | 3   | Heading of the plane, global frame (rad)           | -PI  | PI  | rad            |

    # Action Space

    The action is a `ndarray` with shape `(2,)`, representing the aileron and throttle inputs.

    | Num | Action                               | Min  | Max | Unit           |
    |-----|--------------------------------------|------|-----|----------------|
    | 0   | Airspeed setpoint, body frame (x)             | 0    | Inf   | m / s       |
    | 1   | Yaw rate setpoint, body frame (z)          | -Inf   | Inf   | rad / s        |

    # Transition Dynamics:

    Given an action, the plane is following transition dynamics:

    Position += Speed * [cos(heading), sin(heading)] * dt

    Acceleration_feedback = P_gain * (Airspeed_setpoint - Airspeed)
    Acceleration = Acceleration_feedback
    Airspeed += Acceleration * dt

    Heading += Yawrate_setpoint * dt

    # Starting State

    If not specified, the plane starts at (-world_size/2, 0), the left-most point of the environment.
    With the velocity of (max_airspeed/2, 0)

    ```
    gym.make('fixedwing-lateral')
    """

    def _get_obs(self):
        """ Get Observation from internal state """
        return self.state

    def _get_info(self):
        """ Get Information from internal state """
        return { }

    def _is_terminated(self):
        """ Get Termination state from internal state """

    def __init__(self, render_mode: Optional[str] = None, world_size = 200.0, airspeed_bounds = np.array([AIRSPEED_MIN_DEFAULT, AIRSPEED_MAX_DEFAULT]), yawrate_bounds = np.array([-1.0, 1.0]), acceleration_bounds = np.array([-1.0, 1.0])):
        """ Initialize the Environment """
        self.dt = 0.03
        self.world_size = world_size
        self.min_position, self.max_position = -world_size/2, world_size/2
        self.min_airspeed, self.max_airspeed = airspeed_bounds
        self.min_acceleration,self.max_acceleration = acceleration_bounds
        self.min_yawrate, self.max_yawrate = yawrate_bounds

        # Airspeed controller
        self.p_airspeed = 1.0

        # Action space
        self.min_action = np.array(
            [self.min_airspeed, self.min_yawrate]
        )
        self.max_action = np.array(
            [self.max_airspeed, self.max_yawrate]
        )
        self.action_space = Box(
            low=self.min_action, high=self.max_action, shape=ACTION_SPACE_SHAPE
        )

        # Observation space (equals state space)
        self.min_state = np.array(
            [self.min_position, self.min_position, self.min_airspeed, self.min_yawrate]
        )
        self.max_state = np.array(
            [self.max_position, self.max_position, self.max_airspeed, self.max_yawrate]
        )
        self.observation_space = Box(
            low=self.min_state, high=self.max_state, shape=STATE_SPACE_SHAPE
        )
    
    def decode_state(self, state):
        """ Decodes the state into discrete values """
        assert np.shape(state) == STATE_SPACE_SHAPE
        pos = state[0:2]
        airspeed = state[2]
        heading = state[3]
        
        return (pos, airspeed, heading)

    def decode_action(self, action=np.ndarray):
        ''' Decode action into discrete values. '''
        assert np.shape(action) == ACTION_SPACE_SHAPE
        airspeed_sp = action[0]
        yawrate_sp = action[1]
        
        return (airspeed_sp, yawrate_sp)

    def step(self, action: np.ndarray):
        assert(np.shape(action) == ACTION_SPACE_SHAPE)

        pos, airspeed, heading = self.decode_state(self.state)
        airspeed_sp, yawrate_sp = self.decode_action(action)

        # Integrate state
        pos += airspeed * np.array([np.cos(heading), np.sin(heading)]) * self.dt

        # Apply P-control on airspeed
        acc_fb = (airspeed_sp - airspeed) * self.p_airspeed
        airspeed += acc_fb * self.dt

        # Apply yawrate control on heading
        heading += yawrate_sp * self.dt

        # Set new state
        self.state = np.concatenate((pos, airspeed, heading), axis=None)

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
            initial_airspeed = np.array([self.max_airspeed/2])
            initial_heading = np.array([0.0])
            initial_state = np.concatenate((initial_pos, initial_airspeed, initial_heading), axis=None)
        
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

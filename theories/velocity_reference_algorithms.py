"""
Collections of new definition of the jerk limited velocity reference curve algorithm.

The jerk-limited track error boundary/ velocity curves will be drawn.

This will respect:
- Minimum, Nominal, Maximum Airspeed
- Maximum Acceleration and Jerk

As a variable, user can input:
- Desired speed on path
- Track error (orthogonal to path)

As an output it will return:
- [Vel_parallel, Vel_orthogonal] desired at a specified track error position
"""

import numpy as np

# Constants
VELOCITY_RANGE_SHAPE = (3,)

# Helper constants (for virtual path to follow, for some 2D algorithms)
PATH_POSITION = np.array([0, 0])
PATH_UNIT_TANGENT_VEC = np.array([1.0, 0.0])
PATH_CURVATURE = 0.0

class VelocityReferenceCurves:
    '''
    Base class for velocity reference curve formulations

    Input:
    - Minimum, Nominal, Maximum Airspeed
    - Track error (orthogonal distance from the path)
    - Desired Speed on path: Although PF algorithm shouldn't require this as an input, we would use 'nominal speed' otherwise, which is just a hidden assumption. This exposes that.

    Optional:
    - Maximum Acceleration (assuming point-mass model, applies for XY component in total)
    - Maximum Jerk [m/s^3]

    ---

    Output:
    - Velocity reference vector [parallel, orthogonal (to path)] in [m/s] (always positive)
    '''
    def __init__(self, vel_range, max_acc, max_jerk):
        self.vel_range = vel_range
        self.max_acc_xy = max_acc
        self.max_jerk_xy = max_jerk
        self.assert_class_variables()
    
    def assert_class_variables(self):
        assert np.shape(self.vel_range) == VELOCITY_RANGE_SHAPE
        assert self.max_acc_xy > 0
        assert self.max_jerk_xy > 0

    def assert_input_variables(self, track_error, desired_speed):
        assert track_error >= 0
        assert desired_speed >= 0
        assert desired_speed >= self.vel_range[0] and desired_speed <= self.vel_range[2]

    def calculate_velRef(self, track_error, desired_speed):
        '''
        Main function to calculate the velocity reference

        This needs to be different for each method derived from this base class
        '''
        assert False, "calculate_velRef of base class shouldn't be used directly!"

class TjNpfg(VelocityReferenceCurves):
    def __init__(self, vel_range, max_acc, max_jerk, ground_speed):
        # Initialize class
        super().init(vel_range, max_acc, max_jerk)
        self.ground_speed = ground_speed # Additional argument for TJ NPFG, stays constant
        assert self.ground_speed >= self.vel_range[0] and self.ground_speed <= self.vel_range[2]

        # Construct NPFG instance
        from windywings.libs.npfg import NPFG
        self.npfg = NPFG(vel_range[1], vel_range[2]) # Create NPFG instance

    def calculate_velRef(self, track_error, desired_speed):
        '''
        Returns velocity reference, as defined in TJ's NPFG

        NOTE
        - Velocity of vehicle is always in X-axis direction (doesn't affect NPFG calculation)
        - Position of vehicle is at (0, pos_y)
        '''
        self.assert_input_variables(track_error, desired_speed)

        # Augmented position of the vehicle from the track error. We place vehicle on y < 0 coordinate, under the line (y == 0)
        vehicle_pos = PATH_POSITION + track_error * np.array([0.0, -1.0])
        self.npfg.navigatePathTangent_nowind(vehicle_pos, PATH_POSITION, PATH_UNIT_TANGENT_VEC, self.ground_speed * np.array([1.0, 0.0]), PATH_CURVATURE)

        # X, Y component. Should represent Parallel and Orthogonal components of reference velocity curve
        return self.npfg.getAirVelRef()

class TjNpfgBearingFeasibilityStripped(VelocityReferenceCurves):
    '''
    This is a simplified implementation of TJ's NPFG, with the bearing feasibility functionality stripped.

    Because the feasibility function interferes with setting the desired minimum ground speed (when using track-keeping),
    so it has a dependency to vehicle's real-time ground speed (as that affects feasibility output).

    Therefore, this will solely include:
    - Track error based look ahead angle variation

    What will be different from original TJ's NPFG will be:
    - No Bearing Feasibility function input to the model

    And it includes the assumptions of:
    '''
    
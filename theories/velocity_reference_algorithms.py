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

    def get_track_error_boundary(self):
        '''
        Helper function to return track error boundary calculated from last `calculate_velRef` execution

        This needs to be different for each method derived from this base class
        '''
        assert False, "get_track_error_boundary of base class shouldn't be used directly!"

class TjNpfg(VelocityReferenceCurves):
    def __init__(self, vel_range, max_acc, max_jerk, ground_speed):
        # Initialize class
        super().__init__(vel_range, max_acc, max_jerk)
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

        # Note: the actual course angle of the vehicle wouldn't affect the air-velocity ref vector calculation. Only magnitude gets accounted for.
        self.npfg.navigatePathTangent_nowind(vehicle_pos, PATH_POSITION, PATH_UNIT_TANGENT_VEC, self.ground_speed * np.array([1.0, 0.0]), PATH_CURVATURE)

        # X, Y component. Should represent Parallel and Orthogonal components of reference velocity curve
        return self.npfg.getAirVelRef()

    def get_track_error_boundary(self):
        return self.npfg.d_track_error_bound

class TjNpfgBearingFeasibilityStripped(TjNpfg):
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

    # Constructor is shared with TjNpfg class

    def calculate_velRef(self, track_error, desired_speed):
        '''
        Returns velocity reference, bypassing bearing feasibility calculation

        Basically replaces the `guideToPath_nowind` function.
        '''
        self.assert_input_variables(track_error, desired_speed)

        # Constants
        NEGATIVE_TRACK_ERROR_AUGMENT = -1.0 # Augmented signed track error to have an effect of vehicle being at 'right' side of the path. Used in `bearingVec` func of NPFG
        ZERO_FEASIBILITY_COMBINED_AUGMENT = 0.0 # Augmented combined feasibility, to disable effect of feasibility in scaling in `minGroundSpeed` of NPFG

        # skip `adaptPeriod`, `pGain` and `timeConst` function calls

        # LOGIC
        # TEB calculation with time-constant is left as legacy
        track_error_bound = self.npfg.trackErrorBound(self.ground_speed, self.npfg.time_const)
        normalized_track_error = np.clip(track_error/track_error_bound, 0.0, 1.0)

        look_ahead_ang = self.npfg.lookAheadAngle(normalized_track_error) # LAA solely based on track proximity (normalized)
        bearing_vector = self.npfg.bearingVec(PATH_UNIT_TANGENT_VEC, look_ahead_ang, NEGATIVE_TRACK_ERROR_AUGMENT)
        minimum_groundspeed_reference = self.npfg.minGroundSpeed(normalized_track_error, ZERO_FEASIBILITY_COMBINED_AUGMENT)
        
        # X, Y component. Should represent Parallel and Orthogonal components of reference velocity curve
        return self.npfg.refAirVelocity(bearing_vector, minimum_groundspeed_reference)

class RuckigTimeOptimalTrajectory(VelocityReferenceCurves):
    '''
    Utilizes the `ruckig` package from the paper: "Jerk-limited Real-time Trajectory Generation with Arbitrary Target States" (http://arxiv.org/abs/2105.04830)

    This library creates a bang-bang controller type (on Jerk term) trajectory that guarantees a time-optimal
    trajectory that reaches the target condition.

    Goal of the trajectory:
    - As the 'parallel' position (distance) on the path as a target is undefined (we can freely choose), we simply
    choose to reach the 'target speed on path' in optimal time, along with coming to a stop in orthogonal direction.
    - Therefore, target pos = {track error, UNDEFINED} and target vel = {0, path speed target}, in orthogonal/parallel terms each.
    
    Limitations:
    - Jerk limit is applied to each axis, hence the max jerk is actually sqrt(2) * Jerk, in most cases actually (But could be improved on time synchronization step by modifying the library)

    Findings:
    - The 'initial' condition is actually not well defined, depending on the trajectory we can always have different speed/acceleration as an initial condition
    - One method to bypass this would be to consider that at the 'track error boundary', the vehicle is at it's max approach speed & 0 acceleration
    - This will then formulate the whole vector field, and trajectory won't be generated for each different track error conditions, but would simply provide output
    of the look-up table formulated initially from the track error boundary condition assumption.
    '''
    def __init__(self, vel_range, max_acc, max_jerk):
        super().init(vel_range, max_acc, max_jerk)

        # Calculate the track error boundary directly
        # We assume vehicle heading orthogonal to the path at MAXIMUM velocity, then slowing down

    def calculate_velRef(self, track_error, desired_speed):
        # Orthogonal velocity follows a simple 
        return None
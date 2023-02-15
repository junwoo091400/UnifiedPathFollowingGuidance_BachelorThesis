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

# Junk variables
MAX_ACC_DEFAULT = 0.1
MAX_JERK_DEFAULT = 0.1

# Helper constants (for virtual path to follow, for some 2D algorithms)
PATH_POSITION = np.array([0, 0])
PATH_UNIT_TANGENT_VEC = np.array([1.0, 0.0])
PATH_CURVATURE = 0.0

# Helper Function
def velocity_array_to_parallel_position_array(velocity_arrays, track_error_range):
    '''
    Dirty discrete integration function to convert velocity curves into the parallel position array (along track error).
    The dt is calculated based on remaining distance / orthogonal velocity & multiply with parallel velocity to get position advancement.
    Done to bypass using SciPy, which can be a bit more complicated / may not be necessary for viewing general movement.

    Input: Velocity reference vector [parallel, orthogonal (to path)] in [m/s] (always positive)
    Output: [Final parallel position, ... 0(= initial starting position entering boundary)]
    '''
    return

def vel_array_to_acc(vel_parallel_array, vel_orthogonal_array, track_error_range):
    '''
    Output: [[Parallel Accelerations ... ], [Orthogonal Accelerations ... ]]
    '''
    assert np.ndim(vel_parallel_array) == 1 and np.ndim(vel_orthogonal_array) == 1
    assert np.shape(vel_parallel_array) == np.shape(track_error_range) and np.shape(vel_orthogonal_array) == np.shape(track_error_range)
    
    # Calculate rough acceleration based on discrete track error boundary based parallel / orthogonal vel curves
    # We back-trace it starting from the on-path (minimum track error, index 0 of the error range array), and
    track_error_len = len(track_error_range)

    acc_P = np.empty(track_error_len)
    acc_O = np.empty(track_error_len)
    # dt = np.empty(track_error_len) # Dt for each segment [i, i+1]

    for i in range(track_error_len-1):
        # dt = dE/V_orth(e). NOTE that track error range's derivative is NEGATIVE of orthogonal vel
        # We increase 'accuracy' of velocity thing by incorporating average of two velocity values
        avg_vel_orth = (vel_orthogonal_array[i]+vel_orthogonal_array[i+1])/2

        if avg_vel_orth == 0.0:
            # Divide by zero case (basically infeasible Vector Field, as with 0 orth vel, it can't reach target)
            acc_O[i] = np.nan
            acc_P[i] = np.nan
        else:
            dt = (track_error_range[i+1]-track_error_range[i])/avg_vel_orth
            # Acc orth = d(V_orth)/dt
            acc_O[i] = (vel_orthogonal_array[i]-vel_orthogonal_array[i+1])/dt
            # Acc parallel = d(V_p)/dt
            acc_P[i] = (vel_parallel_array[i]-vel_parallel_array[i+1])/dt

    # provide NaN for the final track error range element, as we can't calculate the derivative there.
    acc_P[-1] = np.nan
    acc_O[-1] = np.nan

    return (acc_P, acc_O)

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
    def __init__(self, vel_range, max_acc=MAX_ACC_DEFAULT, max_jerk=MAX_JERK_DEFAULT):
        self.vel_range = vel_range
        self.max_acc_xy = max_acc
        self.max_jerk_xy = max_jerk
        self.assert_class_variables()
    
    def assert_class_variables(self):
        assert np.shape(self.vel_range) == VELOCITY_RANGE_SHAPE
        assert self.max_acc_xy > 0
        assert self.max_jerk_xy > 0

    def assert_input_variables(self, track_error, v_path):
        assert track_error >= 0
        assert v_path >= 0
        assert v_path >= self.vel_range[0] and v_path <= self.vel_range[2]

    def calculate_velRef(self, track_error, v_path):
        '''
        Main function to calculate the velocity reference

        This needs to be different for each method derived from this base class
        '''
        assert False, "calculate_velRef of base class shouldn't be used directly!"

    def calculate_velRef_array(self, track_error_range, v_path):
        '''
        Convenience function to return velRef curve for given track error range in to a list

        [[Parallel Vel Curve], [Orthogonal Vel Curve]] form
        '''
        return np.stack([self.calculate_velRef(e, v_path) for e in track_error_range], axis=1)

    def calculate_parallel_position_array(self, track_error_range, v_path):
        '''
        Calculates the path-parallel trajectory formed by the Vref curve.

        Logically, the integration would result in:
        [[0, max(track_error_range)], ... [Final parallel position, min(track_error_range)]]

        However, we preserve the track error array order for return:
        [[Final parallel position, min(track_error_range)], ... [0, max(track_error_range)]]
        '''
        assert False, "calculate_parallel_position_array should not be called directly!"

    def get_track_error_boundary(self):
        '''
        Helper function to return track error boundary calculated from last `calculate_velRef` execution

        This needs to be different for each method derived from this base class
        '''
        assert False, "get_track_error_boundary of base class shouldn't be used directly!"

class TjNpfg(VelocityReferenceCurves):
    def __init__(self, vel_range, ground_speed, track_keeping_speed, max_acc=MAX_ACC_DEFAULT, max_jerk=MAX_JERK_DEFAULT):
        # Initialize class
        super().__init__(vel_range, max_acc, max_jerk)
        self.ground_speed = ground_speed # Additional argument for TJ NPFG, stays constant
        assert self.ground_speed >= self.vel_range[0] and self.ground_speed <= self.vel_range[2]

        # Construct NPFG instance
        from windywings.libs.npfg import NPFG
        self.npfg = NPFG(vel_range[1], vel_range[2]) # Create NPFG instance

        # Set track keeping speed
        self.npfg.set_track_keeping_speed(track_keeping_speed)
        self.npfg.min_ground_speed = 0.0 # To enable track keeping, remove min ground speed constraint

    def set_ground_speed(self, ground_speed):
        self.ground_speed = ground_speed

    def calculate_velRef(self, track_error, v_path):
        '''
        Returns velocity reference, as defined in TJ's NPFG

        NOTEtrack_errort (0, pos_y)
        '''
        self.assert_input_variables(track_error, v_path)

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

    def calculate_velRef(self, track_error, v_path):
        '''
        Returns velocity reference, bypassing bearing feasibility calculation

        Basically replaces the `guideToPath_nowind` function.
        '''
        self.assert_input_variables(track_error, v_path)

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
        super().__init__(vel_range, max_acc, max_jerk)

        # Calculate the track error boundary directly
        # We assume vehicle heading orthogonal to the path at MAXIMUM velocity, then slowing down

    def calculate_velRef(self, track_error, v_path):
        # Orthogonal velocity follows a simple 
        return None

class TjNpfgCartesianlVapproachMin(TjNpfg):
    '''
    On top of TJ's NPFG:
    - Includes 'V_approach_min', which guarantees that vehicles approaches the path at minimum this (ground) speed
    - Decoupled orthogonal/parallel velocity reference relative to unit path tangent vector

    Decoupled logic:
    - If our velocity on path (v_path) is below the V_approach, we should behave in decoupled orthogonal/parallel logic
    - If the v_path is higher than V_approach, just consider it as a fixed-wing NPFG with nominal speed of v_path (since it's high enough)

    NOTE: We don't use the `ground_speed` variable, as we want to decouple that from the definition of v_approach, which should only
    be about the 'orthogonal' velocity component to the path tangent
    '''
    def __init__(self, vel_range, v_approach_min):
        GROUND_SPEED_DUMMY = vel_range[1] # We won't be using this part of the algorithm. Put any sane value in.

        # NOTE: Ideally, when we use legacy NPFG logic (V_path > V_approach), track-keeping part shouldn't interfere.
        # This was done mostly to do control consistent when using cartesian velocity formulation (where track keeping isn't considered yet)
        TRACK_KEEPING_SPEED_DUMMY = 0 # We don't use track keeping feature, so set it as dummy value of 0
        
        super().__init__(vel_range, GROUND_SPEED_DUMMY, TRACK_KEEPING_SPEED_DUMMY)
        self.v_approach_min = v_approach_min

    def calculate_velRef(self, track_error, v_path):
        '''
        Depending on (track_error, v_approach), draw different VF

        v_approach: Desired approaching speed (should ideally be user-configurable, or somehow set to constant by vehicle's actual approach speed)

        One challenge is that v_approach changes across the track_error (if we constantly use vehicle's state to determine the value), and hence
        the VF will constantly change while approaching the path
        '''
        # Constants
        NEGATIVE_TRACK_ERROR_AUGMENT = -1.0 # Augmented signed track error to have an effect of vehicle being at 'right' side of the path. Used in `bearingVec` func of NPFG
        ZERO_FEASIBILITY_COMBINED_AUGMENT = 0.0 # Augmented combined feasibility, to disable effect of feasibility in scaling in `minGroundSpeed` of NPFG

        # Set the approach speed
        v_approach = np.max([self.v_approach_min, self.vel_range[1], v_path])

        if v_path >= v_approach:
            # High speed approach, treat like TJ's native NPFG. Unicyclic ramp-in.
            # Set vehicle nominal speed to v_path (velocity on path)
            self.npfg.airspeed_nom = v_path

            # NOTE: The setting of airpseed_nom to a sane positive value higher than NPFG's airspeed buffer
            # is what allows us to bypass the problem of coupling on bearing Feasibility function slowing down vehicle
            # when approaching the path (e.g. Multicopter with Vnom = 0)

            self.track_error_bound = self.npfg.trackErrorBound(v_approach, self.npfg.time_const)
            normalized_track_error = np.clip(track_error/self.track_error_bound, 0.0, 1.0)
            
            look_ahead_ang = self.npfg.lookAheadAngle(normalized_track_error) # LAA solely based on track proximity (normalized)
            track_proximity = self.npfg.trackProximity(look_ahead_ang)
            bearing_vector = self.npfg.bearingVec(PATH_UNIT_TANGENT_VEC, look_ahead_ang, NEGATIVE_TRACK_ERROR_AUGMENT)
            minimum_groundspeed_reference = self.npfg.minGroundSpeed(normalized_track_error, ZERO_FEASIBILITY_COMBINED_AUGMENT)

            # X, Y component. Should represent Parallel and Orthogonal components of reference velocity curve
            return self.npfg.refAirVelocity(bearing_vector, minimum_groundspeed_reference)
        else:
            # Velocity on path is lower than the nominal unicyclic trajectory velocity constraint.
            self.track_error_bound = self.npfg.trackErrorBound(v_approach, self.npfg.time_const)
            normalized_track_error = np.clip(track_error/self.track_error_bound, 0.0, 1.0)
            
            look_ahead_ang = self.npfg.lookAheadAngle(normalized_track_error) # LAA solely based on track proximity (normalized)

            # Track proximity starts from 0, and reaches 1 when on path
            track_proximity = np.sin(look_ahead_ang)
            # track_proximity = self.npfg.trackProximity(look_ahead_ang)
            # track_proximity = np.sqrt(track_proximity) # We remove the squared component, to make the curve stiffer at origin (to drive V_orthogonal stepper to the path)

            # Simply apply ramp-in & ramp-out on the Vpath and Vapproach
            v_parallel = v_path * track_proximity
            v_orthogonal = v_approach * np.cos(look_ahead_ang) # Doing (1 - sin) gives lower stiffness (as sine curve flattens out around PI/2). So use cosine instead.
            
            # Parallel, Orthogonal vel component
            return np.array([v_parallel, v_orthogonal])

    def get_track_error_boundary(self):
        return self.track_error_bound

class MaxAccelCartesianVelCurve(VelocityReferenceCurves):
    def __init__(self, vel_range, max_acc_orth, max_acc_parallel, v_approach_min):
        super().__init__(vel_range)

        # Custom parameters
        self.max_acc_orth = max_acc_orth
        self.max_acc_parallel = max_acc_parallel
        self.v_approach_min = v_approach_min

        # Runtime variable
        self.track_error_boundary = -1.0
        self.e_min_approach = -1.0
        self.e_min_path = -1.0
        self.v_approach = -1.0

    def calculate_e_min_approach_for_S_orth_max_acc(self, v_approach, max_acc_orth):
        '''
        Minimum track error boundary to bring V_approach to 0 with given maximum orthogonal acceleration
        '''
        return np.square(v_approach)/(2*max_acc_orth)

    def calculate_S_orth_max_acc(self, v_approach, max_acc_orth, track_error_range):
        '''
        Calculate maximum acceleration orthogonal velocity curve

        X: cross track error
        Y: orthogonal velocity
        '''
        self.e_min_approach = self.calculate_e_min_approach_for_S_orth_max_acc(v_approach, max_acc_orth)
        return np.piecewise(track_error_range, [track_error_range < self.e_min_approach, track_error_range >= self.e_min_approach], [lambda e : np.sqrt(2*max_acc_orth*e), v_approach])

    def calculate_relaxed_S_orth(self, v_approach, max_acc_orth, relaxed_track_error_bound, track_error_range):
        '''
        Calculate relaxed maximum acceleration orthogonal velocity curve

        X: cross track error
        Y: orthogonal velocity
        '''
        self.e_min_approach = self.calculate_e_min_approach_for_S_orth_max_acc(v_approach, max_acc_orth)
        assert relaxed_track_error_bound >= self.e_min_approach, "Relaxed track error bound must be bigger than e_min_approach!"
        return np.piecewise(track_error_range, [track_error_range < relaxed_track_error_bound, track_error_range >= relaxed_track_error_bound], [lambda e : v_approach * np.sqrt(e/relaxed_track_error_bound), v_approach])

    def calculate_e_min_path_for_relaxed_S_orth_max_acc(self, v_path, max_acc_parallel, v_approach, track_error_bound):
        '''
        Minimum track error boundary to ramp in parallel velocity to V_path, with given relaxed (could be max-acc as well) orthogonal velocity curve
        '''
        return np.square(v_path*v_approach/(2*max_acc_parallel))/track_error_bound

    def calculate_S_parallel_max_acc_with_relaxed_S_orth(self, v_path, max_acc_parallel, track_error_range, v_approach, track_error_bound):
        '''
        Calculate maximum acceleration parallel velocity curve

        NOTE: As non-polynomial functions can't be defined in numpy directly, this has
        dependency to already calculated orthogonal velocity curve!

        It takes track error bound input from the orthogonal velocity curve. And must produce
        a curve with track error bound 'smaller' than the orthogonal curve's!

        X: cross track error
        Y: parallel velocity
        '''
        self.e_min_path = self.calculate_e_min_path_for_relaxed_S_orth_max_acc(v_path, max_acc_parallel, v_approach, track_error_bound)
        assert self.e_min_path <= track_error_bound, "Minimum track error boundary for achieving V_path must be smaller than track error bound of orthogonal velocity curve!"
        return np.piecewise(track_error_range, [track_error_range < self.e_min_path, track_error_range >= self.e_min_path], [lambda e : v_path - (2*max_acc_parallel*np.sqrt(track_error_bound)/v_approach)*np.sqrt(e), 0])

    def calculate_relaxed_S_parallel_max_acc_with_relaxed_S_orth(self, v_path, max_acc_parallel, track_error_range, v_approach, track_error_bound):
        '''
        Calculate relaxed maximum acceleration parallel velocity curve

        NOTE: As non-polynomial functions can't be defined in numpy directly, this has
        dependency to already calculated orthogonal velocity curve!

        It takes track error bound input from the orthogonal velocity curve. And must produce
        a curve with track error bound 'smaller' than the orthogonal curve's!

        X: cross track error
        Y: parallel velocity
        '''
        self.e_min_path = self.calculate_e_min_path_for_relaxed_S_orth_max_acc(v_path, max_acc_parallel, v_approach, track_error_bound)
        assert self.e_min_path <= track_error_bound, "Minimum track error boundary for achieving V_path must be smaller than track error bound of orthogonal velocity curve!"
        return np.piecewise(track_error_range, [track_error_range < track_error_bound, track_error_range >= track_error_bound], [lambda e : v_path*(1-np.sqrt(e/track_error_bound)), 0])

    def calculate_velRef_array(self, track_error, v_path):
        '''
        Calculate semi-relaxed track error boundary with 
        '''
        self.v_approach = np.max([self.vel_range[1], self.v_approach_min, v_path])

        # Calculate most aggressive track error boundary for given V_approach
        e_min_approach = self.calculate_e_min_approach_for_S_orth_max_acc(self.v_approach, self.max_acc_orth)
        # Calculate most aggressive track error boundary for given V_path, with max acc orthogonal velocity curve (e_min_approach)
        e_min_path = self.calculate_e_min_path_for_relaxed_S_orth_max_acc(v_path, self.max_acc_parallel, self.v_approach, e_min_approach)
    
        # Choose the bigger track error boundary (physical lower limit)
        self.track_error_boundary = np.max([e_min_approach, e_min_path])

        # Calculate velocity curves
        S_orth = self.calculate_relaxed_S_orth(self.v_approach, self.max_acc_orth, self.track_error_boundary, track_error)
        S_parallel = self.calculate_relaxed_S_parallel_max_acc_with_relaxed_S_orth(v_path, self.max_acc_parallel, track_error, self.v_approach, self.track_error_boundary)

        return np.array([S_parallel, S_orth])

    def get_track_error_boundary(self):
        return self.track_error_boundary
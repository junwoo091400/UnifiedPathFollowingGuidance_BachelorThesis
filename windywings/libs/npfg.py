"""
Implementation of the PX4 NPFG library.

Code: https://github.com/PX4/PX4-Autopilot/tree/main/src/lib/npfg
Paper: "On Flying Backwards: Preventing Run-away of Small, Low-speed, Fixed-wing UAVs in Strong Winds"

Frame of Reference:
The XYZ position coordinate uses East-North-Up frame, as corresponding to FWLateral NPFG environment.
Therefore the Z-axis points upwards (opposite of gravity)!

Author: Junwoo Hwang (junwoo091400@gmail.com)
"""

import numpy as np
from scipy.spatial.transform import Rotation

AIRSPEED_NOM_DEFAULT = 15.0 # [m/s] Default nominal airspeed in NPFG
AIRSPEED_MAX_DEFAULT = 25.0 # [m/s] Default maximum airspeed in NPFG

ACCEL_FF_SHAPE = (2,) # Shape of the acceleration feed-forward term for path curvature compensation

class NPFG:
    def __init__(self, airspeed_nom = AIRSPEED_NOM_DEFAULT, airspeed_max = AIRSPEED_MAX_DEFAULT):
        ''' Initialize NPFG library with vehicle specific parameters '''
        # Parameters (constant)
        self.track_error_bound_ground_speed_cutoff = 1.0 # [m/s] Ground speed cutoff under which track error bound forms a quadratic function that saturates at speed = 0.
        self.min_radius = 0.5 # [m] Minimum effective radius. For Path curvature compensating lateral accel calculation (MIN_RADIUS)
        self.normalized_track_error_bound_for_maximum_track_keeping = 0.5 # When normalized track error reaches this boundary, track keeping will command full authority on minimum ground speed (TODO:Why is this really needed? Having it to 1.0 won't push the vehicle *enough to the path in excess wind?)
        self.airspeed_buffer_for_bearing_feasibility = 1.5 # [m/s] Airspeed buffer, which is the size of the feasibility transition region at bearing & wind angle >= 90 deg.

        # Parameters (user-adjustable). TODO: Remove magic numbers
        self.period = 10.0 # [s] Nominal desired period
        self.damping = 0.7071 # Nominal desired damping ratio
        self.time_const = 7.071 # Time constant for ground speed based track error bound calculation. Equals period * damping,
        self.p_gain = 0.8885 # [rad/s] Proportional game, computed from period and damping
        self.airspeed_nom = airspeed_nom # [m/s] Nominal (desired) airspeed refernece, a.k.a cruise optimized airspeed
        self.airspeed_max = airspeed_max # [m/s] Maximum airspeed vehicle can achieve
        self.min_ground_speed = 0.0 # [m/s] Minimum ground speed to keep at all times (if not set to 0, this interferes and over-writes the track-keeping induced min ground speed!!)
        self.max_min_ground_speed_track_keeping = 5.0 # [m/s] Maximum 'minimum ground speed' track keeping feature can command (grows linearly with track error)

        # Internal Variables for Runtime calculation (needed for implementation details)
        self.air_vel_ref = np.array([0.0, 0.0]) # [m/s] Air velocity reference command generated
        self.accel_ff_curve = np.array([0.0, 0.0]) # [m/s^2] Feed-forward acceleration command for following the curvature of the path in global frame

        # Internal Variables for debugging only! (Should not be 'read' internally)
        self.d_position_error_vec = np.array([0.0, 0.0])
        self.d_unit_path_tangent = np.array([1.0, 0.0])
        self.d_signed_track_error = 0.0
        self.d_track_error_bound = 0.0
        self.d_normalized_track_error = 0.0
        self.d_look_ahead_angle_from_track_error = 0.0
        self.d_track_proximity = 0.0
        self.d_bearing_vector = np.array([1.0, 0.0])
        self.d_lateral_accel_no_curve = 0.0
        self.d_closest_point_on_path = np.array([0.0, 0.0])

    def set_track_keeping_speed(self, track_keeping_speed):
        ''' Set maximum value for minimum ground speed '''
        assert self.airspeed_max >= track_keeping_speed # Clip to sane range (assuming no wind)
        self.max_min_ground_speed_track_keeping = track_keeping_speed

    def trackErrorBound(self, ground_speed, time_constant):
        """ Calculates continuous ground track error bound depending on ground speed """
        assert np.shape(ground_speed) == ( ), "Ground speed : {}, shape: {}".format(ground_speed, np.shape(ground_speed))
        # Vg_co * T = 0.5 * (Vg_co^2 + C) * T. Hence, "C = Vg_co * (2 - Vg_co)"! C must be > 0.
        # NOTE: Therefore, Vg_co must be < 2 as well, to avoid track error bound becoming <= 0, at ground speed = 0.
        assert (self.track_error_bound_ground_speed_cutoff > 0 and self.track_error_bound_ground_speed_cutoff < 2.0), "Ground speed cutoff not within (0.0, 2.0) range!"

        # Calculate saturation value at ground speed == 0
        track_error_saturation_val = self.track_error_bound_ground_speed_cutoff * (2.0 - self.track_error_bound_ground_speed_cutoff)
        
        if (ground_speed > self.track_error_bound_ground_speed_cutoff):
            return ground_speed * time_constant
        else:
            return 0.5 * time_constant * (np.square(ground_speed) + track_error_saturation_val)

    def bearingFeasibility(self, wind_vec: np.ndarray, bearing_rad, airspeed):
        ''' Calculate continuous bearing feasibility with given wind vector (relative to unit path tangent) & current vehicle airspeed '''
        # NOTE: Originnal code relies on having 4 inputs, but I reduced it to 3 inputs by using vector for wind velocity
        assert self.airspeed_buffer_for_bearing_feasibility > 0.0
        assert np.shape(wind_vec) == (2,), "Wind vector isn't 2D! Shape of {} = {}".format(wind_vec, np.shape(wind_vec))

        wind_speed = np.linalg.norm(wind_vec)
        assert wind_speed >= 0.0

        bearing_vec = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
        wind_dot_bearing = np.dot(wind_vec, bearing_vec)
        wind_cross_bearing = np.cross(wind_vec, bearing_vec)

        if (wind_dot_bearing < 0.0):
            wind_cross_bearing = wind_speed
        else:
            wind_cross_bearing = abs(wind_cross_bearing)

        sin_val = np.sin(0.5*np.pi*np.clip((airspeed - wind_cross_bearing)/self.airspeed_buffer_for_bearing_feasibility, 0.0, 1.0))

        return np.square(sin_val)

    def lookAheadAngle(self, normalized_track_error):
        """ Look Ahead Angle (angle which bearing setpoint vector deviates from unit track error vector) soley based on normalized track error """
        assert (normalized_track_error >= 0.0 and normalized_track_error <= 1.0)
        # As we get closer to the track (track error nom 1.0 -> 0.0), angle goes from 0 to PI/2
        # TODO: Clarify what this actually entails physically
        return np.pi * 0.5 * np.square(1.0 - normalized_track_error)
    
    def trackProximity(self, look_ahead_ang):
        """ Calculates track proximity (0 when at track error boundary, 1 when on track) """
        return np.square(np.sin(look_ahead_ang))

    def bearingVec(self, unit_path_tangent, look_ahead_angle, signed_track_error):
        """ Calculate bearing setpoint based on look ahead angle, varying between unit_path_tangent (UPT) and a track error vector (which is orthogonal to UPT) """
        assert np.shape(unit_path_tangent) == (2, )
        # NOTE: STE equals to UPT x (Track error = Vehicle - Path point).
        # In the end, it is turning the unit track error vector by 'look ahead angle', in an appropriate direction
        if (signed_track_error > 0):
            # When POSITIVE, it means Vehicle is on the left side of the path
            # Bearing vector is UPT turned clockwise by (PI/2 - LAA), which is -Z rotation
            rot = Rotation.from_euler('z', -(np.pi/2 - look_ahead_angle))
            unit_path_tangent_3d = np.array([unit_path_tangent[0], unit_path_tangent[1], 0.0])
            rotated = rot.apply(unit_path_tangent_3d)
            # Return the 2D coordinates of the rotated 3D vector
            return rotated[0:2]
        else:
            # When NEGATIVE, it means Vehicle is on the right side of the path
            # Bearing vector is UPT turned counter-clockwise by (PI/2 - LAA), which is +Z rotation
            rot = Rotation.from_euler('z', (np.pi/2 - look_ahead_angle))
            unit_path_tangent_3d = np.array([unit_path_tangent[0], unit_path_tangent[1], 0.0])
            rotated = rot.apply(unit_path_tangent_3d)
            # Return the 2D coordinates of the rotated 3D vector
            return rotated[0:2]

    def lateralAccel(self, air_velocity, air_velocity_reference):
        ''' Calculates lateral acceleration based on course error. Resulting accelerationg is a scalar value, applied orthogonal to air_velocity vector '''
        assert np.shape(air_velocity) == (2, )
        assert np.shape(air_velocity_reference) == (2, )

        air_speed = np.linalg.norm(air_velocity)
        air_speed_ref = np.linalg.norm(air_velocity_reference)
        air_vel_error_dot = np.dot(air_velocity, air_velocity_reference)
        air_vel_error_crossed = np.cross(air_velocity, air_velocity_reference)

        if (air_vel_error_dot < 0):
            # Heading error > PI/2. If cross > 0 (need to turn counter-clockwise), command positive accel (in Y-axis, body frame)
            if (air_vel_error_crossed > 0):
                return self.p_gain * air_speed
            else:
                return -self.p_gain * air_speed
        else:
            # print('NPFG:LateralAccel: p_gain:{}, air_vel_ref_crossed:{}, air speed ref:{}'.format(self.p_gain, air_vel_error_crossed, air_speed_ref))
            return self.p_gain * air_vel_error_crossed / air_speed_ref

    def refAirVelocity(self, bearing_vector, min_ground_speed_ref = 0.0):
        ''' Calculate reference (target) airmass-relative velocity, corresponding to the bearing vector '''
        assert np.shape(bearing_vector) == (2, )
        # NOTE: This is the crucial function that handles track keeping feature, etc.
        # Since we assume no wind, bearing is always feasible. Just return airspeed vector in bearing vector direction

        # We are always in a condition to respect minimum ground speed / track keeping
        airspeed_min = min_ground_speed_ref # Since there's no wind, minimum airspeed required is equal to minimum ground speed

        if airspeed_min > self.airspeed_max:
            # Infeasible airspeed setting
            # Since there's no wind and bearing is always feasible, directly calculate max vel in bearing setpoint direction
            return self.airspeed_max * bearing_vector

        elif airspeed_min > self.airspeed_nom:
            # Feasible range between nominal speed ~ maximum speed
            return airspeed_min * bearing_vector

        else:
            # Definitely achievable airspeed (low)
            return self.airspeed_nom * bearing_vector

    def minGroundSpeed(self, normalized_track_error, feasibility_combined):
        ''' Calculate minimum ground speed setpoint, depending on track error / user set minimum ground speed '''
        assert (feasibility_combined <= 1.0 and feasibility_combined >= 0.0)
        assert (normalized_track_error <= 1.0 and normalized_track_error >= 0.0)

        min_gsp_track_keeping = (1.0 - feasibility_combined) * self.max_min_ground_speed_track_keeping * np.clip(normalized_track_error/self.normalized_track_error_bound_for_maximum_track_keeping, 0.0, 1.0)
        return max(self.min_ground_speed, min_gsp_track_keeping) # Set minimum bound to user set min ground speed

    def lateralAccelFF(self, unit_path_tangent, ground_velocity, air_speed, signed_track_error, path_curvature):
        ''' Calculates additional lateral acceleration (2D, global frame) for following path curvature '''
        # NOTE: Calculation is done as if vehicle is at the path setpoint with 0 error, and having to follow the path curvature!
        assert np.shape(unit_path_tangent) == (2, )
        assert np.shape(ground_velocity) == (2, )

        # Path frame curvature is an instantaneous curvature at current distance from the path
        # E.g. concentric circles emanating (rel to `min_radius`?)
        # TODO: Understand this concretely / find reference in the paper
        path_frame_curvature = path_curvature / (max(1.0 - path_curvature * signed_track_error, abs(path_curvature) * self.min_radius))
        tangent_ground_speed = max(np.dot(ground_velocity, unit_path_tangent), 0) # Clip minimum value at 0 m/s
        path_frame_rate = path_frame_curvature * tangent_ground_speed
        speed_ratio = 1.0 # ALWAYS 1.0, TODO: Verify if this is a sane value
        accel_ff_magnitude = air_speed * speed_ratio * path_frame_rate

        # Since it is to follow a curve, it is orthogonal to unit path tangent
        # With curvature > 0, it is counter-clockwise turn, hence rotation of PI/2 from unit path tangent
        rot = Rotation.from_euler('z', (np.pi/2))
        unit_path_tangent_3d = np.array([unit_path_tangent[0], unit_path_tangent[1], 0.0])
        unit_path_tangent_orthogonal = rot.apply(unit_path_tangent_3d)[0:2]
        curvature_sign = bool(path_curvature > 0)
                
        return accel_ff_magnitude * unit_path_tangent_orthogonal * curvature_sign

    def getAirVelRef(self):
        ''' Air velocity reference vector (desired air-velocity) '''
        assert np.shape(self.air_vel_ref) == (2,)
        return self.air_vel_ref

    def getAccelFFCurvature(self):
        ''' Acceleration Feed-forward (2D vector, global frame) required for following path curvature '''
        assert np.shape(self.accel_ff_curve) == ACCEL_FF_SHAPE
        return self.accel_ff_curve

    def guideToPath_nowind(self, ground_vel, unit_path_tangent, signed_track_error, curvature):
        ''' Compute the lateral acceleration and airspeed reference to guide to specified path for fixed-wing (unicyclic motion vehicles) (wind is not considered) '''
        # Assert input variable shapes
        assert np.shape(ground_vel) == (2, ), "Ground Velocity shape isn't (2, )!"
        assert np.shape(unit_path_tangent) == (2, ), "Unit Path Tangent shape isn't (2, )!"
        assert curvature >= 0, "Path curvature value is negative!"

        # Constants
        NO_WIND_VECTOR = np.array([0.0, 0.0])

        # Variables setup
        ground_speed = np.linalg.norm(ground_vel)
        air_vel = ground_vel # Airmass relative velocity equals ground vel with no wind
        air_speed = np.linalg.norm(air_vel)
        feas_on_track = self.bearingFeasibility(NO_WIND_VECTOR, np.arctan2(unit_path_tangent[1], unit_path_tangent[0]), air_speed) # Bearing feasibility in unit path tangent direction. Wind velocity set to 0.0
        track_error = abs(signed_track_error)
        # skip `adaptPeriod`, `pGain` and `timeConst` function calls

        # LOGIC
        track_error_bound = self.trackErrorBound(ground_speed, self.time_const)
        normalized_track_error = np.clip(track_error/track_error_bound, 0.0, 1.0)
        look_ahead_ang = self.lookAheadAngle(normalized_track_error) # LAA solely based on track proximity (normalized)
        track_proximity = self.trackProximity(look_ahead_ang)
        bearing_vector = self.bearingVec(unit_path_tangent, look_ahead_ang, signed_track_error)
        feas_current = self.bearingFeasibility(NO_WIND_VECTOR, np.arctan2(bearing_vector[1], bearing_vector[0]), air_speed) # Bearing feasibility of the ref bearing vector. Wind velocity = 0.0.
        feas_combined = feas_on_track * feas_current
        minimum_groundspeed_reference = self.minGroundSpeed(normalized_track_error, feas_combined)
        self.air_vel_ref = self.refAirVelocity(bearing_vector, minimum_groundspeed_reference)

        # Debug output
        # print('Feas combined: {:.2f}, Normalized Track error: {:.2f}, Min groundspeed:{:.1f}'.format(feas_combined, normalized_track_error, minimum_groundspeed_reference))

        # OUTPUT
        lateral_accel = self.lateralAccel(air_vel, self.air_vel_ref)
        self.accel_ff_curve = self.lateralAccelFF(unit_path_tangent, ground_vel, air_speed, signed_track_error, curvature)
        assert np.isscalar(lateral_accel)
        assert np.shape(self.accel_ff_curve) == ACCEL_FF_SHAPE

        # Debug values
        self.d_track_error_bound = track_error_bound
        self.d_normalized_track_error = normalized_track_error
        self.d_look_ahead_angle_from_track_error = look_ahead_ang
        self.d_track_proximity = track_proximity
        self.d_bearing_vector = bearing_vector
        self.d_lateral_accel_no_curve = lateral_accel

        # NOTE: For now, we just take the norm of the accel ff term and assume it is in 'lateral' direction to air velocity vector (which probably isn't true..?)
        return lateral_accel + feas_combined * track_proximity * np.linalg.norm(self.accel_ff_curve)

    def navigatePathTangent_nowind(self, vehicle_pos, position_setpoint, tangent_setpoint, ground_vel, curvature):
        ''' Follow the line path specified by position, path tangent & curvature. Acts as a proxy to the `guidetoPath` function '''
        assert np.shape(vehicle_pos) == (2, ), "Vehicle pos shape: {}".format(np.shape(vehicle_pos))
        assert np.shape(position_setpoint) == (2, )
        assert np.shape(tangent_setpoint) == (2, )
        assert np.shape(ground_vel) == (2, ), "Ground vel: {}. Shape: {}".format(ground_vel, np.shape(ground_vel))

        unit_path_tangent = tangent_setpoint / np.linalg.norm(tangent_setpoint)
        position_error_vec = vehicle_pos - position_setpoint
        signed_track_error = np.cross(unit_path_tangent, position_error_vec)
        
        closest_point_on_path = position_setpoint + np.dot(unit_path_tangent, position_error_vec) * unit_path_tangent
        # TODO: Clarify why the PX4 implementation uses the `position setpoint` directly as the closest point on path, cuz it's not guaranteed to be :/
        # But, it seems that `closest_point_on_path_` doesn't really affect the behavior of function tough, so not relevant.

        # Debug values
        self.d_closest_point_on_path = closest_point_on_path
        self.d_position_error_vec = position_error_vec
        self.d_unit_path_tangent = unit_path_tangent
        self.d_signed_track_error = signed_track_error

        return self.guideToPath_nowind(ground_vel, unit_path_tangent, signed_track_error, curvature)
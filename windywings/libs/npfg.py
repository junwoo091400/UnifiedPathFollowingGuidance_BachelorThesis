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

class NPFG:
    def __init__(self, airspeed_nom = 15.0):
        ''' Initialize NPFG library with vehicle specific parameters '''
        # Parameters (constant)
        self.track_error_bound_ground_speed_cutoff = 1.0 # [m/s] Ground speed cutoff under which track error bound forms a quadratic function that saturates at speed = 0.
        self.min_radius = 0.5 # [m] Minimum effective radius. For Path curvature compensating lateral accel calculation (MIN_RADIUS)

        # Parameters (user-adjustable)
        self.period = 10.0 # [s] Nominal desired period
        self.damping = 0.7071 # Nominal desired damping ratio
        self.time_const = 7.071 # Time constant for ground speed based track error bound calculation. Equals period * damping,
        self.p_gain = 0.8885 # [rad/s] Proportional game, computed from period and damping
        self.airspeed_nom = airspeed_nom # [m/s] Nominal (desired) airspeed refernece, a.k.a cruise optimized airspeed

        # Internal Variables for Runtime calculation (needed for implementation details)

        # Internal Variables for debugging only! (Should not be 'read' internally)
        self.d_position_error_vec = np.array([0.0, 0.0])
        self.d_unit_path_tangent = np.array([1.0, 0.0])
        self.d_signed_track_error = 0.0
        self.d_track_error_bound = 0.0
        self.d_normalized_track_error = 0.0
        self.d_look_ahead_angle_from_track_error = 0.0
        self.d_track_proximity = 0.0
        self.d_bearing_vector = np.array([1.0, 0.0])
        self.d_air_vel_ref = np.array([15.0, 0.0])
        self.d_lateral_accel_no_curve = 0.0
        self.d_lateral_accel_ff_curve = 0.0
        self.d_closest_point_on_path = np.array([0.0, 0.0])

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

    def lookAheadAngle(self, normalized_track_error):
        """ Look Ahead Angle (angle which bearing setpoint vector deviates from unit track error vector) soley based on normalized track error """
        assert (normalized_track_error >= 0.0 and normalized_track_error <= 1.0)
        # As we get closer to the track (track error nom 1.0 -> 0.0), angle goes from 0 to PI/2
        # TODO: Clarify what this actually entails physically
        return np.pi * 0.5 * np.square(1.0 - normalized_track_error)
    
    def trackProximity(self, look_ahead_ang):
        """ Calculates track proximity (0 when at track error boundary, 1 when on track) """
        return np.square(np.sin(look_ahead_ang))

    """
    def rotate2DVector(self, vec, angle_rad):
        ''' Helper function that rotates `vec` in XY plane by `angle_rad` in Z-axis '''
        # https://www.rollpie.com/post/311
        # NOTE: The SciPy already has this implemented: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """

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
        ''' Calculates lateral acceleration based on heading error '''
        assert np.shape(air_velocity) == (2, )
        assert np.shape(air_velocity_reference) == (2, )

        air_speed = np.linalg.norm(air_velocity)
        air_vel_error_dot = np.dot(air_velocity, air_velocity_reference)
        air_vel_error_crossed = np.cross(air_velocity, air_velocity_reference)

        if (air_vel_error_dot < 0):
            # Heading error > PI/2. If cross > 0 (need to turn counter-clockwise), command positive accel (in Y-axis, body frame)
            if (air_vel_error_crossed > 0):
                return self.p_gain * air_speed
            else:
                return -self.p_gain * air_speed
        else:
            return self.p_gain * air_vel_error_crossed / self.air_speed

    def refAirVelocity(self, bearing_vector, min_ground_speed_ref = None):
        ''' Calculate reference (target) airmass-relative velocity, corresponding to the bearing vector '''
        assert np.shape(bearing_vector) == (2, )
        # NOTE: This is the crucial function that handles track keeping feature, etc.
        # Since we assume no wind, bearing is always feasible. Just return airspeed vector in bearing vector direction
        # NOTE: Didn't implement the function in full (regarding min ground speed), just using nominal airpeed user setting
        return self.airspeed_nom * bearing_vector

    def lateralAccelFF(self, unit_path_tangent, ground_velocity, air_speed, signed_track_error, path_curvature):
        ''' Calculates additional lateral acceleration for path curvature '''
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

        return air_speed * speed_ratio * path_frame_rate

    def guideToPath_nowind(self, ground_vel, unit_path_tangent, signed_track_error, curvature):
        ''' Compute the lateral acceleration and airspeed reference to guide to specified path (wind is not considered) '''
        # Assert input variable shapes
        assert np.shape(ground_vel) == (2, ), "Ground Velocity shape isn't (2, )!"
        assert np.shape(unit_path_tangent) == (2, ), "Unit Path Tangent shape isn't (2, )!"
        assert curvature >= 0, "Path curvature value is negative!"

        # VAR setup
        ground_speed = np.linalg.norm(ground_vel)
        air_vel = ground_vel # Airmass relative velocity equals ground vel with no wind
        air_speed = np.linalg.norm(air_vel)
        feas_on_track = 1.0 # Bearing feasiblity always 1.0 with no wind
        track_error = abs(signed_track_error)
        # skip `adaptPeriod`, `pGain` and `timeConst` function calls

        # LOGIC
        track_error_bound = self.trackErrorBound(ground_speed, self.time_const)
        normalized_track_error = np.clip(track_error/track_error_bound, 0.0, 1.0)
        look_ahead_ang = self.lookAheadAngle(normalized_track_error) # LAA solely based on track proximity (normalized)
        track_proximity = self.trackProximity(look_ahead_ang)
        bearing_vector = self.bearingVec(unit_path_tangent, look_ahead_ang, signed_track_error)
        feas_combined = 1.0 # With no wind, feasibility is 1.0 (on track feas) * 1.0 (with wind)
        air_vel_ref = self.refAirVelocity(bearing_vector)

        # OUTPUT
        lateral_accel = self.lateralAccel(air_vel, air_vel_ref)
        lateral_accel_curve = self.lateralAccelFF(unit_path_tangent, ground_vel, air_speed, signed_track_error, curvature)

        # Debug values
        self.d_track_error_bound = track_error_bound
        self.d_normalized_track_error = normalized_track_error
        self.d_look_ahead_angle_from_track_error = look_ahead_ang
        self.d_track_proximity = track_proximity
        self.d_bearing_vector = bearing_vector
        self.d_air_vel_ref = air_vel_ref
        self.d_lateral_accel_no_curve = lateral_accel
        self.d_lateral_accel_ff_curve = lateral_accel_curve

        return lateral_accel + feas_combined * track_proximity * lateral_accel_curve

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

        
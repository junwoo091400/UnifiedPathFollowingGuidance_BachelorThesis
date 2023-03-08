'''
For: Final presentation

Description: This script visualizes the ground track of the unicyclic / hybrid unicyclic methods

Note: This script can handle the different ground tracks drawn via wind, and solves wind triangle.

Coordinate frame: X (Right), Y (Up). Path extends in +X direction with specified curvature.
'''

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

# Constants
TEB_TCONST = 7.071
TEB_GS_CUTOFF = 1.0 # Under this ground speed, the track error boundary gets saturated to non-zero value (until ground speed is 0)

# User adjustable variables
VEL_NOM = 10.0 # For Unicyclic

VEL_APPROACH = 10.0 # For Hybrid Unicyclic
VEL_PATH = 10.0 # For Hybrid Unicyclic

# Wind
WIND_P = 7.0 # Parallel to path
WIND_O = 5.0 # Orthogonal to path

WIND_VEL = np.array([WIND_P, WIND_O])

# Path
PATH_CURVATURE = 0.0 # 1/r

# Helper
def rotateVector2(vec, angle):
    '''
    Rotates a 2D vector by angle (in radians)
    '''
    rot = Rotation.from_euler('z', angle)
    vec_3d = np.array([vec[0], vec[1], 0.0])
    rotated = rot.apply(vec_3d)
    return rotated[0:2]

# Unicyclic
def bearingIsFeasible(wind_cross_bearing, wind_dot_bearing, airspeed, wind_speed):
    '''
    Logical bearing feasibility check
    '''
    return (np.abs(wind_cross_bearing) < airspeed) and (wind_dot_bearing > 0 or wind_speed < airspeed)

def projectAirSpOnBearing(airspeed, wind_cross_bearing):
    '''
    Projection of the air velocity vector onto the bearing line considering a connected wind triangle.

    Basically returns the amount of airspeed *left to spare, to counter the wind in cross-bearing direction

    NOTE: wind_cross_bearing must be less than airspeed to use this function (must be feasible)
    '''
    return np.sqrt(np.square(airspeed) - np.square(wind_cross_bearing))

def solveWindTriangle(air_speed, wind_vel, desired_course):
    '''
    Calculates the air velocity reference that should be commanded (with magnitude air_speed) to achieve desired_course.

    Result should be: bearing(air_vel + wind_vel) = desired_course

    Note: for infeasible solution, it will return None
    '''
    unit_course_vec = np.array([np.cos(desired_course), np.sin(desired_course)])
    wind_dot_course = np.dot(wind_vel, unit_course_vec)
    wind_cross_course = np.cross(wind_vel, unit_course_vec)

    # Check if solution is feasible
    if(bearingIsFeasible(wind_cross_course, wind_dot_course, air_speed, np.linalg.norm(wind_vel))):
        # Feasible
        airspeed_dot_bearing = projectAirSpOnBearing(air_speed, wind_cross_course)
        # Add the airspeed component along bearing & wind cross component to bearing
        return airspeed_dot_bearing * unit_course_vec + wind_cross_course * np.array([-unit_course_vec[1], unit_course_vec[0]])
    else:
        # Infeasible. Don't consider it for now
        return None

def getUnicyclicGroundVel(signed_track_error, unit_path_tangent, ground_speed, wind_vel):
    '''
    Returns the ground velocity vector (2D) and track error boundary

    Track error: Unit path tangent x (Vehicle pos - Path pos)
    '''
    # TEB
    if ground_speed > TEB_GS_CUTOFF:
        track_error_boundary = ground_speed * TEB_TCONST
    else:
        # Calculate saturation value at ground speed == 0
        track_error_saturation_val = TEB_GS_CUTOFF * (2.0 - TEB_GS_CUTOFF)
        track_error_boundary = 0.5 * TEB_TCONST * (np.square(ground_speed) + track_error_saturation_val)

    normalized_track_error = np.abs(signed_track_error)/track_error_boundary

    # Ground course
    # Note: if track error > 0, it means vehicle is on 'left' of the path, so it's look-ahead angle is in positively increasing course direction
    path_course = np.arctan2(unit_path_tangent[1], unit_path_tangent[0])

    if normalized_track_error > 1.0:
        if normalized_track_error > 0:
            ground_course = path_course - np.pi * 0.5
        else:
            ground_course = path_course + np.pi * 0.5
    else:
        # Within track error boundary
        laa = np.pi * 0.5 * np.square(1.0 - normalized_track_error) # Look ahead angle
        if normalized_track_error > 0:
            ground_course = path_course - np.pi*0.5 + laa
        else:
            ground_course = path_course + np.pi*0.5 - laa
    
    # Wind Triangle
    air_vel = solveWindTriangle(VEL_NOM, wind_vel, ground_course)

    # Resulting ground velocity (Should be in line with ground course!)
    ground_vel = air_vel + wind_vel

    return (ground_vel, track_error_boundary)

# Hybrid-Unicyclic
def getHybridUnicyclicGroundVel(signed_track_error, unit_path_tangent, v_approach, v_path):
    '''
    Returns hybrid unicyclic ground velocity vector (2D) and track error boundary

    Track error: Unit path tangent x (Vehicle pos - Path pos)

    NOTE: Wind triangle isn't considered, and we assume the vehicle is always able to achieve the desired ground velocity.
    '''
    # TEB
    track_error_boundary = v_approach * TEB_TCONST
    normalized_track_error = np.abs(signed_track_error)/track_error_boundary

    # Ground course (Still in path relative frame)
    # Note: if track error > 0, it means vehicle is on 'left' of the path, so it's look-ahead angle is in positively increasing course direction
    path_course = np.arctan2(unit_path_tangent[1], unit_path_tangent[0])

    if normalized_track_error > 1.0:
        if normalized_track_error > 0:
            ground_course = - np.pi * 0.5
        else:
            ground_course = + np.pi * 0.5
    else:
        # Within track error boundary
        laa = np.pi * 0.5 * np.square(1.0 - normalized_track_error) # Look ahead angle
        if normalized_track_error > 0:
            ground_course = path_course - np.pi*0.5 + laa
        else:
            ground_course = path_course + np.pi*0.5 - laa

    ground_vel_rel_to_path = np.array([v_path * np.cos(ground_course), v_approach * np.sin(ground_course)])

    # Rotate to global frame
    ground_vel = rotateVector2(ground_vel_rel_to_path, path_course)

    return ground_vel, track_error_boundary

# Main Function
def main():
    # Const
    TRACK_ERROR_LEN = 100
    UNIT_PATH_TANGENT = np.array([1.0, 0.0])
    TRACK_ERROR_MIN = 1.0 # Don't put it to 0, as it causes div by 0 and it will never actually converge to 0

    # Variable
    u_vg_norm_last = VEL_NOM # Last ground speed to reference to adapting proper track error boundary changing via ground speed
    u_pp = 0.0 # Parallel position on path starts from 0
    h_pp = 0.0

    # Logging
    # [Vg[0, 1], eb, Pp: Position along path integrated]
    u_log = np.zeros((TRACK_ERROR_LEN, 4))
    h_log = np.zeros((TRACK_ERROR_LEN, 4))

    # Start from outside the bound, to the path
    track_error_range = np.linspace(100.0, TRACK_ERROR_MIN, TRACK_ERROR_LEN)
    de = (np.max(track_error_range) - np.min(track_error_range)) / (TRACK_ERROR_LEN - 1) # Since we have len-1 intervals

    for idx in range(TRACK_ERROR_LEN):
        e = track_error_range[idx]
        u_vg, u_teb = getUnicyclicGroundVel(e, UNIT_PATH_TANGENT, u_vg_norm_last, WIND_VEL)
        h_vg, h_teb = getHybridUnicyclicGroundVel(e, UNIT_PATH_TANGENT, VEL_APPROACH, VEL_PATH)
        
        print('e:{} | Unicyclic Vg:{}, eb:{} | Hybrid Vg:{}, eb:{}'.format(e, u_vg, u_teb, h_vg, h_teb))

        # Variable
        u_vg_norm_last = np.linalg.norm(u_vg)
        u_pp += np.abs(u_vg[0]/u_vg[1]) * de
        h_pp += np.abs(h_vg[0]/h_vg[1]) * de

        # Log
        u_log[idx][:] = np.array([u_vg[0], u_vg[1], u_teb, u_pp])
        h_log[idx][:] = np.array([h_vg[0], h_vg[1], h_teb, h_pp])

    # print(u_log, h_log)

    plt.arrow(0.0, 0.0, WIND_VEL[0], WIND_VEL[1], head_width=2.0, width=0.5, color='skyblue') # Wind vector

    plt.plot(u_log[:, 0], track_error_range, label='Unicyclic Vg_x')
    plt.plot(u_log[:, 1], track_error_range, label='Unicyclic Vg_y')
    plt.plot(np.linalg.norm(u_log[:, :2], axis=1), track_error_range, label='Unicyclic Vg_norm')
    plt.plot(u_log[:, 2], track_error_range, marker='x', label='Unicyclic eb')
    plt.plot(u_log[:, 3], track_error_range, marker='*', label='Unicyclic Pp')

    plt.plot(h_log[:, 0], track_error_range, label='Hybrid Vg_x')
    plt.plot(h_log[:, 1], track_error_range, label='Hybrid Vg_y')
    plt.plot(np.linalg.norm(h_log[:, :2], axis=1), track_error_range, label='Hybrid Vg_norm')
    plt.plot(h_log[:, 2], track_error_range, marker='x', label='Hybrid eb')
    plt.plot(h_log[:, 3], track_error_range, marker='*', label='Hybrid Pp')

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
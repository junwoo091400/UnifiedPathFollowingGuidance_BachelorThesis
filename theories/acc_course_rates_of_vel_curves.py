"""
After meeting on Feb 13th, need for drawing acceleration / course rates were raised to compare
different velocity curves.

Given a path and vehicle's position, we want to figure out what parameters affect
the generated vector field (air velocity reference), and to test different curves
on different vehicle configurations.

Big idea is to have these 'curves' (each identifiable via it's shape) universally
define the path converging behavior, taking into consideration vehicle constraints
such as maximum, minimum speed and acceleration.

Note:
- Path is a straight line with no curvature, connecting (0, 0) and (1, 0)
- Position error is defined to be vehicle's position relative to path (in Y-axis): `Vehicle_Y`
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the velocity curve functions
from velocity_reference_algorithms import *

# Constants
TRACK_ERROR_MAX = 70 # [m] Maximum track error we simulate (in Y-direction, as path is parallel to X axis)
GRID_SIZE = 1.0 # [m] Interval that data will be calculated along track error axis

# User adjustable
VELOCITY_NOM = 5.0
PATH_DESIRED_SPEED = 8.0 # [m/s] Desired speed on path
APPROACH_SPEED_MINIMUM_DEFAULT = 3.0
GROUND_SPEED_DEFAULT = 5.0 # Only should be used by TJ NPFG
TJ_NPFG_TRACK_KEEPING_SPD = 5.0 # Max minimum track keeping ground speed variable (only for TJ NPFG derived algorithms)
MAX_ACC_ORTH = 3.0
MAX_ACC_PARALLEL = 3.0

# Calculated
VELOCITY_RANGE_DEFAULT = np.array([0.0, VELOCITY_NOM, 12.0]) # Arbitrary min, nom and max speed

# Not used for now
MAX_ACC_DEFAULT = 10.0 # Default max acc value [m/s^2]
MAX_JERK_DEFAULT = 5.0 # Default max jerk [m/s^3]

# Visualization
TRACK_ERROR_BOUNDARY_STYLE = 'dashed'
VEL_CURVE_MARKER_TYPE = 'o'

ACC_CURVE_MARKER_TYPE = '*'

TJ_NPFG_COLOR = 'b'
TJ_NPFG_LEGEND = 'TJ NPFG'

TJ_NPFG_BF_STRIPPED_COLOR = 'g'
TJ_NPFG_BF_STRIPPED_LABEL = 'TJ NPFG BF Stripped'

TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR = 'r'
TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL = 'TJ NPFG Cartesian V_approach min'

RELAXED_MAX_ACCEL_CARTESIAN_COLOR = 'brown'
RELAXED_MAX_ACCEL_CARTESIAN_LABEL = 'Relaxed Max Acc Cartesian'

def draw_Vel_Curves(ax_V_parallel: plt.Axes, ax_V_orthogonal: plt.Axes, velCurveObject: VelocityReferenceCurves, track_error_range, v_path, label, color):
    '''
    Draws velocity curves and track error boundary in the given axes with a given color

    Optionally, can draw Vector Field as well
    '''
    vel_data = velCurveObject.calculate_velRef_array(track_error_range, v_path)
    ax_V_parallel.plot(np.abs(track_error_range), vel_data[0], marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_V_parallel.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)
    ax_V_orthogonal.plot(np.abs(track_error_range), vel_data[1], marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_V_orthogonal.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

def draw_aux_curves(ax_acc_p, ax_acc_o, ax_norm, ax_course_rates, ax_p_pos, velCurveObject: VelocityReferenceCurves, track_error_range, v_path, label, color):
    '''
    Draw auxillary analysis curves
    '''
    S_p, S_o = velCurveObject.calculate_velRef_array(track_error_range, v_path)
    S_acc_p, S_acc_o = vel_array_to_acc(S_p, S_o, track_error_range)
    S_norm = vel_array_to_vel_norm(S_p, S_o)
    S_rates = vel_array_to_course_rate(S_p, S_o, track_error_range)
    S_p_pos = velocity_array_to_parallel_position_array(S_p, S_o, track_error_range)

    # Assume we draw over vel curves axis again (no need to draw Track error boundary)
    # NO LABEL, as it clutters the legend along with vel curve.
    ax_acc_p.plot(track_error_range, S_acc_p, marker=ACC_CURVE_MARKER_TYPE, color=color)
    ax_acc_o.plot(track_error_range, S_acc_o, marker=ACC_CURVE_MARKER_TYPE, color=color)
    
    ax_norm.plot(track_error_range, S_norm, marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_norm.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

    ax_course_rates.plot(track_error_range, S_rates, marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_course_rates.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

    ax_p_pos.plot(track_error_range, S_p_pos, marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_p_pos.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

    # Total Time Printout (debug)
    clipped_track_error_range = track_error_range[track_error_range < velCurveObject.get_track_error_boundary()]
    tot_time = vel_array_to_converge_time(S_o, clipped_track_error_range)
    print('Converge time for {}: {}s, track_error_len:{}'.format(label, tot_time, len(clipped_track_error_range)))

def main():
    # Visualization data setup
    track_error_range = np.arange(0.0, TRACK_ERROR_MAX + GRID_SIZE/2, GRID_SIZE)

    # Instances for each algorithms
    tj_npfg = TjNpfg(VELOCITY_RANGE_DEFAULT, GROUND_SPEED_DEFAULT, TJ_NPFG_TRACK_KEEPING_SPD)
    tj_npfg_bf_stripped = TjNpfgBearingFeasibilityStripped(VELOCITY_RANGE_DEFAULT, GROUND_SPEED_DEFAULT, TJ_NPFG_TRACK_KEEPING_SPD)
    tj_npfg_cartesian_v_approach_min = TjNpfgCartesianlVapproachMin(VELOCITY_RANGE_DEFAULT, APPROACH_SPEED_MINIMUM_DEFAULT)
    max_accel_relaxed_cartesian = MaxAccelCartesianVelCurve(VELOCITY_RANGE_DEFAULT, MAX_ACC_ORTH, MAX_ACC_PARALLEL, APPROACH_SPEED_MINIMUM_DEFAULT)

    # Draw the result
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Vnom {}m/s, Vmax {}m/s, Vpath {} m/s, Vg = {}m/s, Vapproach_min = {}m/s".format(VELOCITY_RANGE_DEFAULT[1], VELOCITY_RANGE_DEFAULT[2], PATH_DESIRED_SPEED, GROUND_SPEED_DEFAULT, APPROACH_SPEED_MINIMUM_DEFAULT))

    # Create Axes
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
    ax_V_parallel = fig.add_subplot(4, 2, 1)
    ax_V_orthogonal = fig.add_subplot(4, 2, 2, sharex=ax_V_parallel)

    ax_Acc_parallel = fig.add_subplot(4, 2, 3, sharex=ax_V_parallel)
    ax_Acc_orthogonal = fig.add_subplot(4, 2, 4, sharex=ax_V_parallel)

    ax_norm = fig.add_subplot(4, 2, 5, sharex=ax_V_parallel)
    ax_coure_rates = fig.add_subplot(4, 2, 7, sharex=ax_V_parallel)

    ax_track = fig.add_subplot(2, 2, 4, sharex=ax_V_parallel)
    
    # Title and Label settings
    ax_V_parallel.set_title('Velocity parallel to path')
    ax_V_parallel.set_ylabel('V parallel [m/s]')
    ax_V_parallel.set_xlabel('Track error [m]')
    ax_V_parallel.grid()

    ax_V_orthogonal.set_title('Velocity orthogonal to path')
    ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
    ax_V_orthogonal.set_xlabel('Track error [m]')
    ax_V_orthogonal.grid()

    ax_Acc_parallel.set_title('Acceleration parallel to path [m/s^2]')
    ax_Acc_parallel.grid()

    ax_Acc_orthogonal.set_title('Acceleration orthogonal to path [m/s^2]')
    ax_Acc_orthogonal.grid()

    ax_norm.set_title('Velocity norm [m/s]')

    ax_coure_rates.set_title('Course rate [rad/s]')

    ax_track.set_title('Track of Vector Field')
    ax_track.grid()

    # Draw Velocity Curves
    # TJ NPFG Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_LEGEND, TJ_NPFG_COLOR)
    # TJ NPFG BF Stripped Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg_bf_stripped, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_STRIPPED_LABEL, TJ_NPFG_BF_STRIPPED_COLOR)
    # TJ NPFG BF Cartesian V_approach_min Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg_cartesian_v_approach_min, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR)
    # Max Acc Relaxed Cartesian
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, max_accel_relaxed_cartesian, track_error_range, PATH_DESIRED_SPEED, RELAXED_MAX_ACCEL_CARTESIAN_LABEL, RELAXED_MAX_ACCEL_CARTESIAN_COLOR)

    # Draw auxilary Curves
    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_coure_rates, ax_track, tj_npfg, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_LEGEND, TJ_NPFG_COLOR)
    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_coure_rates, ax_track, tj_npfg_bf_stripped, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_STRIPPED_LABEL, TJ_NPFG_BF_STRIPPED_COLOR)
    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_coure_rates, ax_track, tj_npfg_cartesian_v_approach_min, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR)
    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_coure_rates, ax_track, max_accel_relaxed_cartesian, track_error_range, PATH_DESIRED_SPEED, RELAXED_MAX_ACCEL_CARTESIAN_LABEL, RELAXED_MAX_ACCEL_CARTESIAN_COLOR)

    # Velocity constraints plot
    VEL_CONSTRAINTS_PLOT_STYLE = 'dashed'
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#selectively-marking-horizontal-regions-across-the-whole-axes
    V_NOM_COLOR = 'cornsilk'
    V_MAX_COLOR = 'mistyrose'
    ax_V_orthogonal.fill_between(track_error_range, VELOCITY_RANGE_DEFAULT[0], VELOCITY_RANGE_DEFAULT[1], color=V_NOM_COLOR)
    ax_V_parallel.fill_between(track_error_range, VELOCITY_RANGE_DEFAULT[0], VELOCITY_RANGE_DEFAULT[1], color=V_NOM_COLOR)
    ax_V_orthogonal.fill_between(track_error_range, VELOCITY_RANGE_DEFAULT[1], VELOCITY_RANGE_DEFAULT[2], color=V_MAX_COLOR)
    ax_V_parallel.fill_between(track_error_range, VELOCITY_RANGE_DEFAULT[1], VELOCITY_RANGE_DEFAULT[2], color=V_MAX_COLOR)

    # Velocity cutoff range draw
    APPROACH_SPEED_MINIMUM_LABEL = 'V_approach_minimum'
    ax_V_orthogonal.axhline(APPROACH_SPEED_MINIMUM_DEFAULT, xmin=np.min(track_error_range), xmax=np.max(track_error_range), color='grey', linestyle=VEL_CONSTRAINTS_PLOT_STYLE, label=APPROACH_SPEED_MINIMUM_LABEL)

    PATH_DESIRED_SPEED_LABEL = 'V_path'
    ax_V_parallel.axhline(PATH_DESIRED_SPEED, xmin=np.min(track_error_range), xmax=np.max(track_error_range), color='grey', linestyle=VEL_CONSTRAINTS_PLOT_STYLE, label=PATH_DESIRED_SPEED_LABEL)

    TJ_NPFG_TRACK_KEEPING_SPD_LABEL = 'V_tk'
    TJ_NPFG_TRACK_KEEPING_SPD_COLOR = 'orange'
    ax_V_orthogonal.axhline(TJ_NPFG_TRACK_KEEPING_SPD, xmin=np.min(track_error_range), xmax=np.max(track_error_range), color=TJ_NPFG_TRACK_KEEPING_SPD_COLOR, linestyle=VEL_CONSTRAINTS_PLOT_STYLE, label=TJ_NPFG_TRACK_KEEPING_SPD_LABEL)

    # Legend
    ax_V_parallel.legend()
    ax_V_orthogonal.legend()
    ax_norm.legend()
    ax_coure_rates.legend()
    ax_track.legend()
    
    fig.tight_layout() # Make sure graphs don't overlap
    plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
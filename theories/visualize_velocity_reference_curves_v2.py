"""
After meeting on January 9th, 2023, need for formulating the parallel/orthogonal
velocity ramping curve in a clean sheet was raised.

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
VELOCITY_NOM = 10.0
PATH_DESIRED_SPEED = 0.0 # [m/s] Desired speed on path
APPROACH_SPEED_MINIMUM_DEFAULT = 3.0
GROUND_SPEED_DEFAULT = 5.0 # Only should be used by TJ NPFG
TJ_NPFG_TRACK_KEEPING_SPD = 5.0 # Max minimum track keeping ground speed variable (only for TJ NPFG derived algorithms)
MAX_ACC_ORTH = 1.0
MAX_ACC_PARALLEL = 2.0

# Calculated
VELOCITY_RANGE_DEFAULT = np.array([0.0, VELOCITY_NOM, 12.0]) # Arbitrary min, nom and max speed

# Not used for now
MAX_ACC_DEFAULT = 10.0 # Default max acc value [m/s^2]
MAX_JERK_DEFAULT = 5.0 # Default max jerk [m/s^3]

# Algorithms
PF_ALGORITHMS_COUNT = 4 # Used for creating the data bucket for storing calculations
PF_ALGORITHM_TJ_NPFG_IDX = 0
PF_ALGORITHM_TJ_NPFG_BF_STRIPPED_IDX = 1
PF_ALGORITHM_TJ_NPFG_CARTESIAN_V_APPROACH_MIN_IDX = 2
PF_ALGORITHM_MAX_ACC_RELAXED_CARTESIAN_IDX = 3

# Visualization
TRACK_ERROR_BOUNDARY_STYLE = 'dashed'
VEL_CURVE_MARKER_TYPE = 'o'

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

def main():
    # Visualization data setup
    track_error_range = np.arange(0.0, TRACK_ERROR_MAX + GRID_SIZE/2, GRID_SIZE)
    track_error_len = len(track_error_range)

    # Data for each different formulations
    grid_data = np.empty((PF_ALGORITHMS_COUNT, track_error_len, 2))

    # Instances for each algorithms
    tj_npfg = Unicyclic(VELOCITY_RANGE_DEFAULT, GROUND_SPEED_DEFAULT, TJ_NPFG_TRACK_KEEPING_SPD)
    tj_npfg_bf_stripped = TjNpfgBearingFeasibilityStripped(VELOCITY_RANGE_DEFAULT, GROUND_SPEED_DEFAULT, TJ_NPFG_TRACK_KEEPING_SPD)
    tj_npfg_cartesian_v_approach_min = HybridUnicyclic(VELOCITY_RANGE_DEFAULT, APPROACH_SPEED_MINIMUM_DEFAULT)
    max_accel_relaxed_cartesian = MaxAccelCartesianVelCurve(VELOCITY_RANGE_DEFAULT, MAX_ACC_ORTH, MAX_ACC_PARALLEL, APPROACH_SPEED_MINIMUM_DEFAULT)

    # Calculation for Vector Field
    for y_idx in range(track_error_len):
        grid_data[PF_ALGORITHM_TJ_NPFG_IDX][y_idx] = tj_npfg.calculate_velRef(track_error_range[y_idx], PATH_DESIRED_SPEED)
        grid_data[PF_ALGORITHM_TJ_NPFG_BF_STRIPPED_IDX][y_idx] = tj_npfg_bf_stripped.calculate_velRef(track_error_range[y_idx], PATH_DESIRED_SPEED)
        # Just consider 'ground speed' used by TJ NPFG, as the 'approach' speed we want in Cartesian V_approach_min algorithm
        grid_data[PF_ALGORITHM_TJ_NPFG_CARTESIAN_V_APPROACH_MIN_IDX][y_idx] = tj_npfg_cartesian_v_approach_min.calculate_velRef(track_error_range[y_idx], PATH_DESIRED_SPEED)
    
    # Flip to have shape of (track_error_len, 2)
    grid_data[PF_ALGORITHM_MAX_ACC_RELAXED_CARTESIAN_IDX, :, :] = np.transpose(max_accel_relaxed_cartesian.calculate_velRef_array(track_error_range, PATH_DESIRED_SPEED))

    # Draw the result
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Vnom {}m/s, Vmax {}m/s, Vpath {} m/s, Vg = {}m/s, Vapproach_min = {}m/s".format(VELOCITY_RANGE_DEFAULT[1], VELOCITY_RANGE_DEFAULT[2], PATH_DESIRED_SPEED, GROUND_SPEED_DEFAULT, APPROACH_SPEED_MINIMUM_DEFAULT))

    # Create Axes
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
    ax_V_parallel = fig.add_subplot(3, 1, 1)
    ax_V_orthogonal = fig.add_subplot(3, 1, 2, sharex=ax_V_parallel)
    ax_VF = fig.add_subplot(3, 1, 3)
    
    # Title and Label settings
    ax_V_parallel.set_title('Velocity parallel to path')
    ax_V_parallel.set_ylabel('V parallel [m/s]')
    ax_V_parallel.set_xlabel('Track error [m]')
    ax_V_parallel.grid()

    # Shares the X-axis range with parallel velocity plot (dynamically updates when user zooms in the plot)
    ax_V_orthogonal.set_title('Velocity orthogonal to path')
    ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
    ax_V_orthogonal.set_xlabel('Track error [m]')
    ax_V_orthogonal.grid()

    # Draw Velocity Curves
    # TJ NPFG Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_LEGEND, TJ_NPFG_COLOR)
    # TJ NPFG BF Stripped Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg_bf_stripped, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_STRIPPED_LABEL, TJ_NPFG_BF_STRIPPED_COLOR)
    # TJ NPFG BF Cartesian V_approach_min Velocity curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, tj_npfg_cartesian_v_approach_min, track_error_range, PATH_DESIRED_SPEED, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR)
    # Max Acc Relaxed Cartesian
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, max_accel_relaxed_cartesian, track_error_range, PATH_DESIRED_SPEED, RELAXED_MAX_ACCEL_CARTESIAN_LABEL, RELAXED_MAX_ACCEL_CARTESIAN_COLOR)

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

    ## Vector Field Drawing
    # Draw the path to follow (Y+ direction, passing through origin)
    ax_VF.axvline(0.0, ymin=np.min(track_error_range), ymax=np.max(track_error_range), color='dimgray', linestyle='dotted')
    ax_VF.set_title('Vector Field (Path = +Y direction, passing through origin)')

    VF_Y_MAX = 20.0 # Maximum (both negative & positive direction) of where the Vector's origins will be located
    VF_Y_RANGE = np.linspace(-VF_Y_MAX, VF_Y_MAX, PF_ALGORITHMS_COUNT) # Equally divide the range for each algorithm
    VF_Y_BUFFER = 2.0 # Buffer [m] to have in Y-axis range
    # VF_VEC_LENGTH = 8 # How much Y-coordinate a single vector can consume (for adjusting spacing & y limits to display)

    VF_SUBSAMPLE_RATIO = 5 # Select only 1 in this ratio number of samples, to reduce clutter
    VF_TRACK_ERROR_IDXS = range(0, track_error_len, VF_SUBSAMPLE_RATIO)

    # Meshgrid must be indexed in matrix form xy, returning shape of (len(algorithms), len(track_error_range)).
    # Meshgrid indexed via 'ij', returning algorithms on the 2nd axis, which is helpful for setting colors (as they apply to 2nd axis)
    VF_X_ORIGINS, VF_Y_ORIGINS = np.meshgrid(track_error_range[VF_TRACK_ERROR_IDXS], VF_Y_RANGE, indexing='ij')
    
    # We 'swap' the axis, to make algorithm index go to 2nd index, since color scheme applies that way & meshgrid follows that convention
    # X is V_orthogonal * -1, which is -grid[1]
    VF_X_VALUES = np.swapaxes(-grid_data[:, VF_TRACK_ERROR_IDXS, 1], 0, 1)
    # Y is V_parallel, if we assume following path in Y+ direction, which is grid[0]
    VF_Y_VALUES = np.swapaxes(grid_data[:, VF_TRACK_ERROR_IDXS, 0], 0, 1)

    # Draw the vectors
    # Scale = 1.0 (identical to m/s).  headlength=2.0, headaxislength=1.5, `units` affects the arrow HEAD & WIDTH size.
    ax_VF.quiver(VF_X_ORIGINS, VF_Y_ORIGINS, VF_X_VALUES, VF_Y_VALUES, scale=1.0, scale_units='xy', units='height', width=0.015, color=[TJ_NPFG_COLOR, TJ_NPFG_BF_STRIPPED_COLOR, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR, RELAXED_MAX_ACCEL_CARTESIAN_COLOR])

    # # Draw the track error boundary
    ax_VF.axvline(tj_npfg.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=TJ_NPFG_COLOR, linestyle=TRACK_ERROR_BOUNDARY_STYLE)
    ax_VF.axvline(tj_npfg_bf_stripped.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=TJ_NPFG_BF_STRIPPED_COLOR, linestyle=TRACK_ERROR_BOUNDARY_STYLE)
    ax_VF.axvline(tj_npfg_cartesian_v_approach_min.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

    # Set Y-limit to cover worst case scenario range
    ax_VF.set_ylim(np.min(VF_Y_RANGE)+np.min(VF_Y_VALUES)-VF_Y_BUFFER, np.max(VF_Y_RANGE)+np.max(VF_Y_VALUES)+VF_Y_BUFFER)
    # ax_VF.set_ylim(np.min(VF_Y_RANGE)-VF_VEC_LENGTH, np.max(VF_Y_RANGE)+VF_VEC_LENGTH)
    ax_VF.grid()
    
    fig.tight_layout() # Make sure graphs don't overlap
    plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
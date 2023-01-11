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

from windywings.libs.npfg import NPFG

'''
Definition of Velocity Reference Generating Functions

Input:
- Minimum, Nominal, Maximum Airspeed
- Maximum Acceleration (assuming point-mass model, applies for XY component in total)
- Desired Speed on path: Although PF algorithm shouldn't require this as an input, we would use 'nominal speed' otherwise, which is just a hidden assumption. This exposes that.
- Track error (orthogonal distance from the path)
- Vehicle speed (Forced to input, due to TJ NPFG constraints)

NOTE: Track Error boundary will be determined automatically

Output:
- Velocity reference vector [X, Y] in [m/s] (always positive)
'''

# Constants
VELOCITY_RANGE_SHAPE = (3,)

PATH_POSITION = np.array([0, 0])
PATH_UNIT_TANGENT_VEC = np.array([1.0, 0.0])
PATH_CURVATURE = 0.0
PATH_DESIRED_SPEED = 10.0 # [m/s] Desired speed on path

TRACK_ERROR_MAX = 100.0 # [m] Maximum track error we simulate (in Y-direction, as path is parallel to X axis)
GRID_SIZE = 5.0 # [m] Interval that data will be calculated along track error axis

# User adjustable
VELOCITY_RANGE_DEFAULT = np.array([0.0, 10.0, 20.0]) # Arbitrary min, nom and max speed
MAX_ACC_DEFAULT = 10.0 # Default max acc value [m/s^2]
GROUND_SPEED_DEFAULT = VELOCITY_RANGE_DEFAULT[1]

# Algorithms
PF_ALGORITHMS_COUNT = 1 # Used for creating the data bucket for storing calculations
PF_ALGORITHM_TJ_NPFG_IDX = 0

def assert_input_variables(vel_range, max_acc_xy, desired_speed, track_error, ground_speed):
    assert np.shape(vel_range) == VELOCITY_RANGE_SHAPE
    assert ground_speed > vel_range[0] and ground_speed < vel_range[2] # Velocity in sane vehicle limit range
    assert ground_speed > 0
    assert max_acc_xy > 0
    assert track_error >= 0
    assert desired_speed >= 0

def TJ_NPFG(vel_range, max_acc_xy, desired_speed, track_error, ground_speed):
    '''
    Returns velocity reference, as defined in TJ's NPFG

    NOTE
    - Velocity of vehicle is always in X-axis direction (doesn't affect NPFG calculation)
    - Position of vehicle is at (0, pos_y)
    '''
    assert_input_variables(vel_range, max_acc_xy, desired_speed, track_error, ground_speed)

    npfg = NPFG(vel_range[1], vel_range[2])
    
    # Augmented position of the vehicle from the track error. We place vehicle on y < 0 coordinate, under the line (y == 0)
    vehicle_pos = PATH_POSITION + track_error * np.array([0.0, -1.0])

    npfg.navigatePathTangent_nowind(vehicle_pos, PATH_POSITION, PATH_UNIT_TANGENT_VEC, ground_speed * np.array([1.0, 0.0]), PATH_CURVATURE)

    return npfg.getAirVelRef()

def main():
    # Visualization data setup
    track_error_range = np.arange(0.0, TRACK_ERROR_MAX + GRID_SIZE/2, GRID_SIZE)
    track_error_len = len(track_error_range)

    # Data for each different formulations
    grid_data = np.empty((PF_ALGORITHMS_COUNT, track_error_len, 2))

    # Calculation for Visualization
    for y_idx in range(track_error_len):
        # TJ NPFG
        grid_data[PF_ALGORITHM_TJ_NPFG_IDX][y_idx] = TJ_NPFG(VELOCITY_RANGE_DEFAULT, MAX_ACC_DEFAULT, PATH_DESIRED_SPEED, track_error_range[y_idx], GROUND_SPEED_DEFAULT)

        # Modified NPFG (no bearing feasibility stuff)
        # TODO

    # Special data per algorithm for visualization
    tj_npfg_track_error_boundary = NPFG().trackErrorBound(GROUND_SPEED_DEFAULT, NPFG().time_const)

    ## Velocity ramp-in Drawing

    # User Configurations
    TJ_NPFG_LINE_COLOR = 'b'
    TJ_NPFG_MARKER_TYPE = 'o'
    TJ_NPFG_LEGEND = 'TJ NPFG'
    TJ_NPFG_TRACK_ERROR_BOUNDARY_STYLE = 'dashed'
    TJ_NPFG_TRACK_ERROR_BOUNDARY_COLOR = 'b' # Not sure if I would keep it equal to line color

    # Draw the result
    fig = plt.figure(figsize=(5, 10))
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
    ax_V_parallel = fig.add_subplot(3, 1, 1)
    ax_V_parallel.set_title('Velocity parallel to path')
    ax_V_parallel.set_ylabel('V parallel [m/s]')
    ax_V_parallel.set_xlabel('Track error [m]')
    ax_V_parallel.grid()

    # Shares the X-axis range with parallel velocity plot (dynamically updates when user zooms in the plot)
    ax_V_orthogonal = fig.add_subplot(3, 1, 2, sharex=ax_V_parallel)
    ax_V_orthogonal.set_title('Velocity orthogonal to path')
    ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
    ax_V_orthogonal.set_xlabel('Track error [m]')
    ax_V_orthogonal.grid()

    print('TJ NPFG Track error boundary: {}m'.format(tj_npfg_track_error_boundary))

    # X-axis is always the 'normalized' track error
    # X index of the grid data doesn't matter, as it's a straight line path in X direction (hence doesn't vary the calculation)
    # X (0th index of the vector in the grid) is the 'parallel' component to the path
    ax_V_parallel.plot(np.abs(track_error_range), grid_data[PF_ALGORITHM_TJ_NPFG_IDX, : , 0], marker=TJ_NPFG_MARKER_TYPE, label=TJ_NPFG_LEGEND, color=TJ_NPFG_LINE_COLOR)
    ax_V_orthogonal.plot(np.abs(track_error_range), grid_data[PF_ALGORITHM_TJ_NPFG_IDX, : , 1], marker=TJ_NPFG_MARKER_TYPE, label=TJ_NPFG_LEGEND, color=TJ_NPFG_LINE_COLOR)

    # Track error boundary
    ax_V_parallel.axvline(tj_npfg_track_error_boundary, ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=TJ_NPFG_TRACK_ERROR_BOUNDARY_COLOR, linestyle=TJ_NPFG_TRACK_ERROR_BOUNDARY_STYLE)
    ax_V_orthogonal.axvline(tj_npfg_track_error_boundary, ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=TJ_NPFG_TRACK_ERROR_BOUNDARY_COLOR, linestyle=TJ_NPFG_TRACK_ERROR_BOUNDARY_STYLE)

    # Legend
    ax_V_parallel.legend()
    ax_V_orthogonal.legend()
    
    plt.show()

    ## Vector Field Drawing
    # ax_VF = fig.add_subplot(2, 1, 2) # Lower

    

    # # Draw the track error boundary
    # ax_VF.hlines(tj_npfg_track_error_boundary, xmin=np.min(grid_xrange), xmax=np.max(grid_xrange), colors=TJ_NPFG_TRACK_ERROR_BOUNDARY_COLOR, linestyles=TJ_NPFG_TRACK_ERROR_BOUNDARY_STYLE)
    # ax_VF.hlines(-tj_npfg_track_error_boundary, xmin=np.min(grid_xrange), xmax=np.max(grid_xrange), colors=TJ_NPFG_TRACK_ERROR_BOUNDARY_COLOR, linestyles=TJ_NPFG_TRACK_ERROR_BOUNDARY_STYLE)

    # # Meshgrid must be indexed in matrix form (i, j), so to have the ROW be constant X values (as in data bucket)
    # # Normalized y-range, for better sync (visualization) with the velocity curve
    # grid_yrange_normalized = grid_yrange / tj_npfg_track_error_boundary

    # X, Y = np.meshgrid(grid_xrange, grid_yrange_normalized, indexing='ij')
    # vx = grid_data[:, :, 0]
    # vy = grid_data[:, :, 1]
    # ax_VF.quiver(X, Y, vx, vy, color='b')

    # ax_VF.set_title('Va_ref VF of NPFG. Vg={:.1f}, Vnom={:.1f}'.format(GROUND_SPEED_DEFAULT, VELOCITY_RANGE_DEFAULT[1]))
    # ax_VF.set_xlim(np.min(grid_xrange), np.max(grid_xrange))
    # ax_VF.set_ylim(np.min(grid_yrange_normalized), np.max(grid_yrange_normalized) + GRID_SIZE/2) # Make sure the top (Y=0) vectors get shown, by increasing Ymax range
    # ax_VF.set_ylabel('Normalized position')
    # ax_VF.grid()

    # # https://stackoverflow.com/questions/45423166/matplotlib-share-xaxis-with-yaxis-from-another-plot
    # # https://stackoverflow.com/questions/31490436/matplotlib-finding-out-xlim-and-ylim-after-zoom
    # def on_normalized_path_error_bound_changed(event_ax):
    #     normalized_path_error_bound = np.array(event_ax.get_xlim())
    #     # Multiply -1, to negative y range, as we are only visualizing (y < 0) portion
    #     # Also flip the array, so that we actually have [A, B] bound where A < B.
    #     ax_VF.set_ylim(-np.flip(normalized_path_error_bound))

    # # Connect hook when user zooms onto specific normalized path error (X-axis of the Velocity Curve Plot)
    # ax_V_parallel.callbacks.connect('xlim_changed', on_normalized_path_error_bound_changed)

    # plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
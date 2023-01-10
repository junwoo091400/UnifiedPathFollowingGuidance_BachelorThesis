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
- Position (XY value of vehicle's position in global frame)
- Vehicle speed (Forced to input, due to TJ NPFG constraints)

NOTE: Track Error boundary will be determined automatically, or fixed as constant

Output:
- Velocity reference vector [X, Y] in [m/s]
'''

# Constants
VELOCITY_RANGE_SHAPE = (3,)
POSITION_SHAPE = (2,)

PATH_POSITION = (0, 0)
PATH_UNIT_TANGENT_VEC = np.array([1.0, 0.0])
PATH_CURVATURE = 0.0

WORLD_HALF_SIZE = (100, 100) # World to visualize in meters, each half distance (X, Y)
GRID_LENGTH = 10.0 # Interval that air velocity reference vector will be calculated at in both X and Y direction

# User adjustable
TRACK_ERROR_BOUNDARY = 50 # NOTE: This isn't respected in TJ NPFG
VELOCITY_RANGE_DEFAULT = np.array([0.0, 10.0, 20.0]) # Arbitrary min, nom and max speed
MAX_ACC_DEFAULT = 10.0 # Default max acc value [m/s^2]
GROUND_SPEED_DEFAULT = VELOCITY_RANGE_DEFAULT[1]

def assert_input_variables(vel_range, max_acc_xy, pos, ground_speed):
    assert np.shape(vel_range) == VELOCITY_RANGE_SHAPE
    assert np.shape(pos) == POSITION_SHAPE
    assert ground_speed > vel_range[0] and ground_speed < vel_range[2] # Velocity in sane vehicle limit range
    assert ground_speed > 0
    assert max_acc_xy > 0

def TJ_NPFG(vel_range, max_acc_xy, pos, ground_speed):
    '''
    Returns velocity reference, as defined in TJ's NPFG

    NOTE
    - Velocity of vehicle is always in X-axis direction (doesn't affect NPFG calculation)
    - Position of vehicle is at (0, pos_y)
    '''
    assert_input_variables(vel_range, max_acc_xy, pos, ground_speed)

    npfg = NPFG(vel_range[1], vel_range[2])
    npfg.navigatePathTangent_nowind(pos, PATH_POSITION, PATH_UNIT_TANGENT_VEC, ground_speed * np.array([1.0, 0.0]), PATH_CURVATURE)

    return npfg.getAirVelRef()

def main():
    # Visualization data setup
    grid_range = np.arange(-WORLD_HALF_SIZE[0], WORLD_HALF_SIZE[0] + GRID_LENGTH/2, GRID_LENGTH)
    grid_length = len(grid_range)

    # Data for each different formulations
    grid_data_tj_npfg = np.empty((grid_length, grid_length, 2)) # Placeholder for data for each grid section

    # Calculation for Visualization
    for x_idx in range(grid_length):
        for y_idx in range(grid_length):
            vehicle_position = np.array([grid_range[x_idx], grid_range[y_idx]])
            
            # TJ NPFG
            grid_data_tj_npfg[x_idx][y_idx] = TJ_NPFG(VELOCITY_RANGE_DEFAULT, MAX_ACC_DEFAULT, vehicle_position, GROUND_SPEED_DEFAULT)

            # Modified NPFG (no bearing feasibility stuff)
            # TODO

    # Special data per algorithm for visualization
    tj_npfg_track_error_boundary = NPFG().trackErrorBound(GROUND_SPEED_DEFAULT, NPFG().time_const)

    ## Velocity ramp-in Drawing
    # User Configurations
    TJ_NPFG_MARKER_TYPE = 'o'
    TJ_NPFG_LEGEND = 'TJ NPFG'

    # Draw the result
    fig = plt.figure(figsize=(10, 10))
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
    ax_V_parallel = fig.add_subplot(2, 1, 1)
    ax_V_parallel.set_title('Velocity parallel to path')
    ax_V_parallel.set_ylabel('V parallel [m/s]')
    ax_V_parallel.grid()

    # Shares the X-axis range with parallel velocity plot (dynamically updates when user zooms in the plot)
    ax_V_orthogonal = fig.add_subplot(2, 1, 2, sharex=ax_V_parallel)
    ax_V_orthogonal.set_title('Velocity orthogonal to path')
    ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
    ax_V_orthogonal.set_xlabel('Normalized track error boundary')
    ax_V_orthogonal.grid()

    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
    # We only clip the Y axis range with negative coordinate, as then the vector field in XY direction are both positive (Better to plot)
    path_error_Y_range = grid_range[grid_range < 0]
    path_error_Y_length = len(path_error_Y_range)

    print('TJ NPFG Track error boundary: {}m'.format(tj_npfg_track_error_boundary))

    # X-axis is always the 'normalized' track error
    # X index of the grid data doesn't matter, as it's a straight line path in X direction (hence doesn't vary the calculation)
    # X (0th index of the vector in the grid) is the 'parallel' component to the path
    ax_V_parallel.plot(np.abs(path_error_Y_range/tj_npfg_track_error_boundary), grid_data_tj_npfg[0, 0:path_error_Y_length, 0], marker=TJ_NPFG_MARKER_TYPE, label=TJ_NPFG_LEGEND)

    ax_V_orthogonal.plot(np.abs(path_error_Y_range/tj_npfg_track_error_boundary), grid_data_tj_npfg[0, 0:path_error_Y_length, 1], marker=TJ_NPFG_MARKER_TYPE, label=TJ_NPFG_LEGEND)

    ax_V_parallel.legend()
    ax_V_orthogonal.legend()
    plt.show()

    print('Exiting main function ...')
    exit()

    ## Vector Field Drawing

    # Meshgrid must be indexed in matrix form (i, j), so to have the ROW be constant X values (as in data bucket)
    X, Y = np.meshgrid(grid_range, grid_range, indexing='ij')
    vx = grid_data_tj_npfg[:, :, 0]
    vy = grid_data_tj_npfg[:, :, 1]

    plt.quiver(X, Y, vx, vy, color='b')
    plt.title('Va_ref VF of NPFG. Vg={:.1f}, Vnom={:.1f}'.format(GROUND_SPEED_DEFAULT, VELOCITY_RANGE_DEFAULT[1]))

    # Finalize Plot
    plt.xlim(-WORLD_HALF_SIZE[0], WORLD_HALF_SIZE[0])
    plt.ylim(-WORLD_HALF_SIZE[1], WORLD_HALF_SIZE[1])
    plt.grid()
    plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
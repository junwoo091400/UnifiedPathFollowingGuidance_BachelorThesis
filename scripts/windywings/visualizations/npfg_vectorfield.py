'''
Visualizes a vector field formed around the path by the 'air velocity reference vector'.

Input:
- Path position
- Path tangent vector
- Path curvature
- Vehicle position
- Vehicle ground speed

Output:
- Air velocity reference vector (2D)

Constant:
- Nominal airspeed of the vehicle (parameter in NPFG)
- Maximum airspeed of the vehicle (parameter in NPFG)

NOTES:
- Since the course angle of the ground velocity only affects the 'acceleration feed-forward for curvature'
term, the ground 'speed' is used as an input, instead of the 'velocity'
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
import matplotlib.cm as cm

from windywings.libs.npfg import NPFG

def main():
    # User adjustable
    PATH_POS = (0, 0)
    PATH_BEARING_RAD = 0.0
    PATH_CURVATURE = 0.0
    VEHICLE_GROUND_SPEED = 20.0

    # Constants
    VEHICLE_NOMINAL_AIRSPEED = 15.0
    VEHICLE_MAXIMUM_AIRSPEED = 20.0
    WORLD_HALF_SIZE = (100, 100) # World to visualize in meters, each half distance (X, Y)
    GRID_LENGTH = 10.0 # Interval that air velocity reference vector will be calculated at in both X and Y direction

    # Auxilary calculation
    PATH_UNIT_TANGENT = np.array([np.cos(PATH_BEARING_RAD), np.sin(PATH_BEARING_RAD)])
    VEHICLE_GROUND_VELOCITY = np.array([VEHICLE_GROUND_SPEED, 0.0]) # Bearing itself isn't critical, set it to 0 (x-axis direction)

    # NPFG setup
    npfg = NPFG(VEHICLE_NOMINAL_AIRSPEED, VEHICLE_MAXIMUM_AIRSPEED)

    # Visualization data setup
    grid_range = np.arange(-WORLD_HALF_SIZE[0], WORLD_HALF_SIZE[0] + GRID_LENGTH/2, GRID_LENGTH)
    grid_length = len(grid_range)
    grid_data = np.empty((grid_length, grid_length, 2)) # Placeholder for data for each grid section

    print('Initialized grid data bucket with shape:', grid_data.shape)

    # Calculation for Visualization
    for x_idx in range(grid_length):
        for y_idx in range(grid_length):
            vehicle_position = np.array([grid_range[x_idx], grid_range[y_idx]])
            npfg.navigatePathTangent_nowind(vehicle_position, PATH_POS, PATH_UNIT_TANGENT, VEHICLE_GROUND_VELOCITY, PATH_CURVATURE)
            
            # Save 2D Air velocity reference vector
            grid_data[x_idx][y_idx] = npfg.getAirVelRef()

    print('At origin:', grid_data[grid_length//2, grid_length//2])
    print('At LEFT X = -width/4, Y = 0:', grid_data[grid_length//4, grid_length//2])
    print('At RIGHT X = width/4, Y = 0:', grid_data[(grid_length//4) * 3, grid_length//2])
    print('At DOWN Y = -width/4, X = 0:', grid_data[grid_length//2, grid_length//4])
    print('At UP Y = width/4, X = 0:', grid_data[grid_length//2, (grid_length//4) * 3])

    print('Track error bound: {}m'.format(npfg.d_track_error_bound))

    # Draw the result
    # Meshgrid must be indexed in matrix form (i, j), so to have the ROW be constant X values (as in data bucket)
    X, Y = np.meshgrid(grid_range, grid_range, indexing='ij')
    vx = grid_data[:, :, 0]
    vy = grid_data[:, :, 1]

    # Build a color map, corresponding to magnitude of the air vel reference vector
    # colors = np.linalg.norm(grid_data, axis=2) # Normalize the vectors of each data point in grid
    # norm = Normalize()
    # norm.autoscale(colors)
    # colormap = cm.inferno
    # colors = colormap(norm(colors))

    # colors = np.arctan2(vy, vx)
    # print(colors.shape)

    plt.quiver(X, Y, vx, vy, color='b')
    plt.title('Va_ref VF of NPFG. Vg={:.1f}, Vnom={:.1f}'.format(VEHICLE_GROUND_SPEED, VEHICLE_NOMINAL_AIRSPEED))
    plt.xlim(-WORLD_HALF_SIZE[0], WORLD_HALF_SIZE[0])
    plt.ylim(-WORLD_HALF_SIZE[1], WORLD_HALF_SIZE[1])

    plt.grid()
    plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
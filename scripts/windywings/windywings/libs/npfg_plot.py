'''
Helper script to visualize how the NPFG functinos work
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from windywings.libs.npfg import NPFG

def visualize_bearingFeasibility():
    '''
        X axis: Bearing setpoint [deg], relative to wind velocity
        Y axis: Wind factor (Windspeed/Airspeed)
        Plotted: Bearing feasibility
    '''
    N = 100 # Discretization
    FIGURE_SIZE_INCHES = 10 # Size of the figure in inches
    AIRSPEEDS = [1.5, 5.0, 10.0, 20.0]

    _npfg = NPFG()
    xaxis = np.linspace(0.0, 180.0, N)
    yaxis = np.linspace(0.0, 1.5, N)
    z = np.empty((N, N))

    # Construct subplots
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(FIGURE_SIZE_INCHES, FIGURE_SIZE_INCHES) # Scale up the image

    # Different airspeeds
    for AIRSPEED, ax in zip(AIRSPEEDS, axes.flat):
        for i in range(len(xaxis)):
            for j in range(len(yaxis)):
                WINDSPEED = AIRSPEED * yaxis[j]
                WIND_VECTOR = WINDSPEED * np.array([1.0, 0.0]) # Wind always in X axis direction
                BEARING_RAD = np.deg2rad(xaxis[i])
                # X-axis is the row, Y-axis is the column. Hence the indexes need to be flipped for `imshow` function.
                z[j, i] = _npfg.bearingFeasibility(WIND_VECTOR, BEARING_RAD, AIRSPEED)

        im = ax.imshow(z, vmin=z.min(), vmax=z.max(), aspect='auto', origin='lower', extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()])
        contour_levels = [0.999] # Contour where the feasibility starts to deteriorate (below 1.0)
        contour_lines = ax.contour(z, contour_levels, colors=['red'], extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()])
        ax.set_title('Bearing feasibility. Va={}m/s, Va buf={}m/s'.format(AIRSPEED, _npfg.airspeed_buffer_for_bearing_feasibility))
        plt.colorbar(im)
        ax.set_xlabel('Bearing [deg]')
        ax.set_ylabel('Wind factor (Vw/Va)')
    
    plt.show()

def visualize_bearingFeasibility_noWind():
    '''
        Visualize how in a no-wind condition (wind factor = INF), vehicle's airspeed affects
        bearing feasibility, depending on different airspeed buffer sizes.

        NOTE: Bearing setpoint isn't so relevant, as vector product with the wind (=0) will zero
        out the term anyway.

        X axis: Vehicle airspeed
        Y axis: Bearing feasibility
    '''
    N = 100 # Discretization
    FIGURE_SIZE_INCHES = 10 # Size of the figure in inches
    AIRSPEED_BUFFERS = [0.1, 0.5, 1.0, 1.5]
    AIRSPEED_MAX = 2.0

    _npfg = NPFG()
    xaxis = np.linspace(0.0, AIRSPEED_MAX, N)

    # Construct subplots
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(FIGURE_SIZE_INCHES, FIGURE_SIZE_INCHES) # Scale up the image

    # Different airspeeds
    for AIRSPEED_BUFFER, ax in zip(AIRSPEED_BUFFERS, axes.flat):
        yaxis = np.empty((N,))
        for i in range(len(xaxis)):
                _npfg.airspeed_buffer_for_bearing_feasibility = AIRSPEED_BUFFER
                yaxis[i] = _npfg.bearingFeasibility(np.array([0.0, 0.0]), 0.0, xaxis[i])
        im = ax.plot(xaxis, yaxis)
        # im = ax.imshow(z, vmin=z.min(), vmax=z.max(), aspect='auto', origin='lower', extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()])
        contour_levels = [0.999] # Contour where the feasibility starts to deteriorate (below 1.0)
        ax.set_title('Bearing feasibility. Airspeed buf={}m/s'.format(_npfg.airspeed_buffer_for_bearing_feasibility))
        ax.set_xlabel('Vehicle airspeed [m/s]')
        ax.set_ylabel('Bearing feasibility')
    
    plt.show()

if __name__ == '__main__':
    # visualize_bearingFeasibility()
    visualize_bearingFeasibility_noWind()
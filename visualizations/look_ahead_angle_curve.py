'''
The 'look-ahead angle' curve gives a good sense on how the Vector Field
around the Path forms, on a given PF algorithm.

This curve has 2 components:
* X-axis: Normalized track error
* Y-axis: Look-ahead angle [deg] (measured from error vector to closest point on path, towards unit path tangent)

The Jerk and Accelerations required by the vehicle can be calculated off of
the curve alone as well.

Assumptions:

Wind-less, constant vehicle velocity assumptions)

Also, the 'track error boundary' is assumed to be constant for now (for NPFG, this means constant ground speed of the vehicle)

The curvature of the path is set to 0 (which removes the ff acceleration & d_shift from 3D NPFG), further simplifying the jerk calculation.

This is a first take at visualizing / pondering on jerk-limited NPFG.
'''

import numpy as np
import matplotlib.pyplot as plt

from windywings.libs.npfg import NPFG

def main():
    x_range = np.arange(0.0, 1.05, 0.05)

    # User adjustable parameters
    track_error_boundary = 70 # [m] Fixed in 3D NPFG, varying in TJ NPFG
    vehicle_groundspeed = 10 # [m/s] Roughly having 7.0 time constant (as in TJ NPFG) relative to track error boundary size
    
    # NPFG: No need to specify any parameters, the LAA doesn't have dependencies
    npfg = NPFG()
    npfg_laa = [npfg.lookAheadAngle(x) for x in x_range]
    npfg_acc = ((x_range * np.sin(npfg_laa)) / track_error_boundary) * vehicle_groundspeed**2
    npfg_jerk = ((x_range * np.sin(npfg_laa)) / track_error_boundary) * npfg_acc * vehicle_groundspeed

    # 3D NPFG: Three-Dimensional Nonlinear Differential Geometric Path-Following Guidance Law
    # Refer to equation 31 & 32 for the formulation
    _3d_npfg_eps = 1E-4
    _3d_npfg_laa_eq31 = [((np.pi/2) * np.sqrt(1 - (1 - _3d_npfg_eps) * np.clip(x, 0.0, 1.0))) for x in x_range]
    _3d_npfg_laa_eq31_acc = ((x_range * np.sin(_3d_npfg_laa_eq31)) / track_error_boundary) * vehicle_groundspeed**2
    _3d_npfg_laa_eq31_jerk = ((x_range * np.sin(_3d_npfg_laa_eq31)) / track_error_boundary) * _3d_npfg_laa_eq31_acc * vehicle_groundspeed

    _3d_npfg_laa_eq32 = [np.arccos((1 - _3d_npfg_eps) * np.clip(x, 0.0, 1.0)) for x in x_range]
    _3d_npfg_laa_eq32_acc = ((x_range * np.sin(_3d_npfg_laa_eq32)) / track_error_boundary) * vehicle_groundspeed**2
    _3d_npfg_laa_eq32_jerk = ((x_range * np.sin(_3d_npfg_laa_eq32)) / track_error_boundary) * _3d_npfg_laa_eq32_acc * vehicle_groundspeed

    # Plot
    window_title = 'Track error boundary={}m, vehicle ground speed={}m/s, No Wind'.format(track_error_boundary, vehicle_groundspeed)
    fig, ax = plt.subplots(3, num=window_title, sharex=True) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html#sharing-axes
    fig.suptitle(window_title)
    fig.set_size_inches(15, 10) # Increase size for visibility

    # look ahead angle [deg]
    ax[0].set_title('Look ahead angle [deg]')
    ax[0].plot(x_range, np.rad2deg(npfg_laa), label='TJ NPFG')
    ax[0].plot(x_range, np.rad2deg(_3d_npfg_laa_eq31), label='3D NPFG Eq.31')
    ax[0].plot(x_range, np.rad2deg(_3d_npfg_laa_eq32), label='3D NPFG Eq.32')
    ax[0].grid()
    ax[0].legend()

    # Acceleration [m/s^2] (normalized to vehicle velocity)
    ax[1].set_title('Acceleration [m/s^2]')
    ax[1].plot(x_range, npfg_acc, label='TJ NPFG')
    ax[1].plot(x_range, _3d_npfg_laa_eq31_acc, label='3D NPFG Eq.31')
    ax[1].plot(x_range, _3d_npfg_laa_eq32_acc, label='3D NPFG Eq.32')
    ax[1].grid()
    ax[1].legend()

    # Jerk [m/s^3] (normalized to vehicle velocity)
    ax[2].set_title('Jerk [m/s^3]')
    ax[2].plot(x_range, npfg_jerk, label='TJ NPFG')
    ax[2].plot(x_range, _3d_npfg_laa_eq31_jerk, label='3D NPFG Eq.31')
    ax[2].plot(x_range, _3d_npfg_laa_eq32_jerk, label='3D NPFG Eq.32')
    ax[2].grid()
    ax[2].legend()
    
    plt.xlabel('Normalized Track Error')
    plt.show()

if __name__ == '__main__':
    main()
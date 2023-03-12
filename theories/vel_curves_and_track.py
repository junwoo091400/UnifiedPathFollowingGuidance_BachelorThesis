"""
Draw just vel curve and track for Final Presentation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Import the velocity curve functions
from velocity_reference_algorithms import *

# Constants
TRACK_ERROR_MAX = 70 # [m] Maximum track error we simulate (in Y-direction, as path is parallel to X axis)
GRID_SIZE = 1.0 # [m] Interval that data will be calculated along track error axis

# User adjustable
IS_MC = True

if IS_MC:
    # Multicopter constraints
    VELOCITY_MIN = 0.0
    VELOCITY_NOM = 6.0
    VELOCITY_MAX = 15.0
else:
    # Fixed wing constraints
    VELOCITY_MIN = 6.0
    VELOCITY_NOM = 10.0
    VELOCITY_MAX = 15.0

PATH_DESIRED_SPEED = 7.0 # [m/s] Desired speed on path

# Calculated
VELOCITY_RANGE_DEFAULT = np.array([VELOCITY_MIN, VELOCITY_NOM, VELOCITY_MAX])

# Visualization
TRACK_ERROR_BOUNDARY_STYLE = 'dashed'
VEL_CURVE_MARKER_TYPE = 'o'

ACC_CURVE_MARKER_TYPE = '*'

TJ_NPFG_COLOR = 'r'
TJ_NPFG_LEGEND = 'Unicyclic'

# TJ_NPFG_BF_STRIPPED_COLOR = 'g'
# TJ_NPFG_BF_STRIPPED_LABEL = 'TJ NPFG BF Stripped'

# TJ_NPFG_BF_STRIPPED_V_PATH_SQUASHED_COLOR = 'pink'
# TJ_NPFG_BF_STRIPPED_V_PATH_SQUASHED_LABEL = 'TJ NPFG squashed'

TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR = 'green'
TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL = 'Hybrid'

RELAXED_MAX_ACCEL_CARTESIAN_COLOR = 'brown'
RELAXED_MAX_ACCEL_CARTESIAN_LABEL = 'Maximum Acceleration'

# Vehicle constraints
# MC_ACC_PATH_ORTH

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

    ax_p_pos.plot(track_error_range, S_p_pos, marker=VEL_CURVE_MARKER_TYPE, label=label, color=color)
    ax_p_pos.axvline(velCurveObject.get_track_error_boundary(), ymin=np.min(track_error_range), ymax=np.max(track_error_range), color=color, linestyle=TRACK_ERROR_BOUNDARY_STYLE)

    # Total Time Printout (debug)
    clipped_track_error_range = track_error_range[track_error_range < velCurveObject.get_track_error_boundary()]
    tot_time = vel_array_to_converge_time(S_o, clipped_track_error_range)
    print('Converge time for {}: {}s, track_error_len:{}'.format(label, tot_time, len(clipped_track_error_range)))

def drawCurves(ax_V_parallel, ax_V_orthogonal, ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_course_rates, ax_track, v_path, vel_range):
    '''
    Draw (refresh) all the curves with given Axes objects & conditions
    '''
    assert np.shape(vel_range) == VELOCITY_RANGE_SHAPE

    # Algorithm parameters
    APPROACH_SPEED_MINIMUM_DEFAULT = 0.0 # Disable V_approach_min, as V_nom !=0 now. And it creates weird artifacts in total velocity magnitude curves, as V_nom and V_approach_min (biggers) differs.
    TJ_NPFG_TRACK_KEEPING_SPD = 0.0 # Max minimum track keeping ground speed variable (only for TJ NPFG derived algorithms). This essentially demoishes the purpose of having BF stripped formulation (for max track keeping vel input)
    GROUND_SPEED_DEFAULT = vel_range[1] # Only should be used by TJ NPFG

    MAX_ACC_ORTH = 4.0
    MAX_ACC_PARALLEL = 4.0

    # Clear the plots
    ax_V_parallel.cla()
    ax_V_orthogonal.cla()
    ax_track.cla()

    # Title and Label settings
    ax_V_parallel.set_title('Velocity parallel to path')
    ax_V_parallel.set_ylabel('V parallel [m/s]')
    ax_V_parallel.set_xlabel('Track error [m]')
    ax_V_parallel.grid()

    ax_V_orthogonal.set_title('Velocity orthogonal to path')
    ax_V_orthogonal.set_ylabel('V orthogonal [m/s]')
    ax_V_orthogonal.set_xlabel('Track error [m]')
    ax_V_orthogonal.grid()

    ax_track.set_title('Path drawn by Vel Curves')
    ax_track.grid()

    # Visualization data setup
    track_error_range = np.arange(0.0, TRACK_ERROR_MAX + GRID_SIZE/2, GRID_SIZE)

    # Instances for each algorithms
    unicyclic = Unicyclic(vel_range, GROUND_SPEED_DEFAULT, TJ_NPFG_TRACK_KEEPING_SPD)
    hybrid_unicyclic = HybridUnicyclicUniform(vel_range, vel_range[1]) # Give V_nom as V_approach by default

    # Draw Velocity Curves
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, unicyclic, track_error_range, v_path, TJ_NPFG_LEGEND, TJ_NPFG_COLOR)  
    draw_Vel_Curves(ax_V_parallel, ax_V_orthogonal, hybrid_unicyclic, track_error_range, v_path, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR)

    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_course_rates, ax_track, unicyclic, track_error_range, v_path, TJ_NPFG_LEGEND, TJ_NPFG_COLOR)
    draw_aux_curves(ax_Acc_parallel, ax_Acc_orthogonal, ax_norm, ax_course_rates, ax_track, hybrid_unicyclic, track_error_range, v_path, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_LABEL, TJ_NPFG_BF_CARTESIAN_V_APPROACH_MIN_COLOR)


    # Velocity constraints plot
    VEL_CONSTRAINTS_PLOT_STYLE = 'dashed'
    
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#selectively-marking-horizontal-regions-across-the-whole-axes
    V_NOM_COLOR = 'cornsilk'
    V_MAX_COLOR = 'mistyrose'
    ax_V_orthogonal.fill_between(track_error_range, vel_range[0], vel_range[1], color=V_NOM_COLOR)
    ax_V_parallel.fill_between(track_error_range, vel_range[0], vel_range[1], color=V_NOM_COLOR)

    PATH_DESIRED_SPEED_LABEL = 'V_path'
    ax_V_parallel.axhline(v_path, xmin=np.min(track_error_range), xmax=np.max(track_error_range), color='grey', linestyle=VEL_CONSTRAINTS_PLOT_STYLE, label=PATH_DESIRED_SPEED_LABEL)

    # Legend: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    ax_V_parallel.legend(loc='upper right')
    ax_V_orthogonal.legend(loc='upper right')
    ax_track.legend(loc='upper right')

def main():
    # Runtime variables (used in slider callback)
    vel_range = VELOCITY_RANGE_DEFAULT
    v_path = PATH_DESIRED_SPEED

    # Draw
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Vnom {}m/s, Vmax {}m/s, Vpath {} m/s".format(vel_range[1], vel_range[2], v_path))

    # Create Axes
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
    ax_V_parallel = fig.add_subplot(3, 1, 1)
    ax_V_orthogonal = fig.add_subplot(3, 1, 2, sharex=ax_V_parallel)
    ax_track = fig.add_subplot(3, 1, 3, sharex=ax_V_parallel)

    ## Sliders
    fig.subplots_adjust(bottom=0.1) # Leave margin in bottom for the slider

    axVpath = plt.axes([0.25, 0.05, 0.5, 0.03])
    sliderVpath = Slider(axVpath, 'V_path [m/s]', 0, VELOCITY_RANGE_DEFAULT[2], valinit=PATH_DESIRED_SPEED, valfmt='%d', valstep=1.0)

    def updateVpath(val):
        print('V_path set to:', val)
        v_path = val

        drawCurves(ax_V_parallel, ax_V_orthogonal, None, None, None, None, ax_track, v_path, vel_range)

        fig.suptitle("Vnom {}m/s, Vmax {}m/s, Vpath {} m/s".format(vel_range[1], vel_range[2], v_path))
        fig.subplots_adjust(bottom=0.1) # Leave margin in bottom for the slider
        fig.canvas.draw_idle()

    sliderVpath.on_changed(updateVpath)

    axVnom = plt.axes([0.25, 0.025, 0.5, 0.03])
    sliderVnom = Slider(axVnom, 'V_nom [m/s]', VELOCITY_RANGE_DEFAULT[0], VELOCITY_RANGE_DEFAULT[2], valinit=VELOCITY_RANGE_DEFAULT[1], valfmt='%d', valstep=1.0)

    def updateVnom(val):
        print('V nom set to:', val)
        vel_range[1] = val

        drawCurves(ax_V_parallel, ax_V_orthogonal, None, None, None, None, ax_track, v_path, vel_range)

        fig.suptitle("Vnom {}m/s, Vmax {}m/s, Vpath {} m/s".format(vel_range[1], vel_range[2], v_path))
        fig.subplots_adjust(bottom=0.1) # Leave margin in bottom for the slider
        fig.canvas.draw_idle()

    sliderVnom.on_changed(updateVnom)

    # Initial draw
    drawCurves(ax_V_parallel, ax_V_orthogonal, None, None, None, None, ax_track, PATH_DESIRED_SPEED, VELOCITY_RANGE_DEFAULT)
    fig.tight_layout() # Make sure graphs don't overlap

    plt.show()

    print('Main function exiting ...')

if __name__ == '__main__':
    main()
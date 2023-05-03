'''
Creates analysis of a ground course curve that is defined using the 3 stage method:

1. Find look-ahead angle (relative to orthogonal unit vector to closest point on path) vs normalized track error
2. Find track error boundary
3. Find ground speed target profile along normalized track error

Based on that, we can compute different quantitative metrics for each of the curves.
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy

e = sympy.Symbol('e') # Normalized track error

## Look ahead angle
LAAs = []
LAAs.append(sympy.pi/2 * (1 - e)**2) # Formulation 1, TJ NPFG
LAAs.append(sympy.pi/2 * sympy.sqrt(1-e)) # Formulation 2, roughly similar to Eq 31 in 3D NPFG
LAAs.append(sympy.acos(e*(2-e))) # Formulation 3, Eq 9 from TJ NMPC

def test_laa(laa):
    # At boundary (e=1), laa should be 0
    assert np.equal(sympy.limit(laa, e, 1), 0)
    # At e=0, laa should be pi/2
    assert np.equal(sympy.limit(laa, e, 0), sympy.pi/2)

for laa in LAAs:
    test_laa(laa)

## Track error boundary
EBs = []
EBs.append(100.0)
EBs.append(70.0)

## Ground speed profile
VGs = []
VGs.append(10.0) # Constant speed, unicyclic
VGs.append(5.0*(1-e) + 10.0*e) # 5m/s on path, 10m/s when approaching

## Quantitative metrics
def t_conv(laa, eb, vg):
    '''
    Time to converge from the error boundary to 1.0 meter absolute offset from path
    '''
    return (eb*sympy.integrate(1/(vg*sympy.cos(laa)), (e, 1.0/eb, 1.0))).evalf()

def p_conv(laa, eb, vg):
    '''
    Distance traveled parallel to path to converge (1.0 meter absolute)

    NOTE: Vg doesn't matter at all
    '''
    return (eb*sympy.integrate(sympy.tan(laa), (e, 1.0/eb, 1.0))).evalf()

def vg_monotonic(laa, eb, vg):
    '''
    Is the ground speed profile monotonic?
    '''
    # Capture the case where Vg is constant ('increasing')
    return sympy.is_monotonic(vg, sympy.Interval(0.0, 1.0)) or sympy.is_increasing(vg, sympy.Interval(0.0, 1.0))

def a_rms(laa, eb, vg):
    '''
    RMS acceleration until convergence
    '''
    return ((sympy.integrate((1/eb)*(vg*sympy.cos(laa))*(sympy.diff(vg, e)**2 + (vg*sympy.diff(laa))**2), (e, 1.0/eb, 1.0)))**0.5).evalf()

def a_max(laa, eb, vg):
    '''
    Maximum acceleration until convergence
    '''
    return sympy.maximum((1/eb)*(vg*sympy.cos(laa))*(sympy.diff(vg, e)**2 + (vg*sympy.diff(laa))**2)**0.5, e, sympy.Interval(1.0/eb, 1.0)).evalf()

## Binary metrics

## Scripts
def t_conv_compare():
    V_APPROACH = 10.0
    V_PATH = 15.0

    eb = 70.0 # Arbitrary, to 10 m/s approach roughly
    laa = LAAs[0]
    vg_unicyclic = 10.0 # Fixed ground speed
    vg_hybrid = sympy.sqrt((V_APPROACH*sympy.cos(laa))**2 + (V_PATH*sympy.sin(laa)**2))

    t_conv_unicyclic = t_conv(laa, eb, vg_unicyclic)
    t_conv_hybrid = t_conv(laa, eb, vg_hybrid)

    print('Time to converge: unicyclic: {}, hybrid: {}'.format(t_conv_unicyclic, t_conv_hybrid))
    print('Pp conv: unicyclic: {}, hybrid: {}'.format(p_conv(laa, eb, vg_unicyclic), p_conv(laa, eb, vg_hybrid)))
    print('Vg monotonic: unicyclic: {}, hybrid: {}'.format(vg_monotonic(laa, eb, vg_unicyclic), vg_monotonic(laa, eb, vg_hybrid)))
    print('Acc rms unicyclic: {}, hybrid: {}'.format(a_rms(laa, eb, vg_unicyclic), a_rms(laa, eb, vg_hybrid)))

# Execute
t_conv_compare()
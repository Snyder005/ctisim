"""Deferred charge correction module.

To Do:
    * Rename module to something better.
    * Test out new trap operator that takes SerialTraps as args (rather than trap params).
"""
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv
from scipy.special import comb

def cti_operator(cti, ncols):
    """Calculate a sparse matrix representing CTI operator."""

    b = cti
    a = 1-cti

    diags = np.asarray([[a**i for i in range(1, ncols+1)],
                        [i*b*(a**i) for i in range(1, ncols+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, ncols+1)]])

    D = dia_matrix((diags, [0, -1, -2]), shape=(ncols, ncols))

    return D

def two_trap_operator(pixel_signals, trapsize1, scaling, trapsize2, f0, k, tau):
    """Calculate trap operator for linear plus logistic trapping parameters."""
    
    def f(pixel_signals):

        return np.minimum(trapsize1, pixel_signals*scaling) + trapsize2/(1.+np.exp(-k*(pixel_signals-f0)))
    
    r = np.exp(-1/tau)
    S_estimate = pixel_signals + f(pixel_signals)
    
    C = f(S_estimate)
    R = np.zeros(C.shape)
    R[:, 1:] = f(S_estimate)[:,:-1]*(1-r)
    R[:, 2:] += np.maximum(0, (f(S_estimate[:, :-2])-f(S_estimate[:, 1:-1]))*r*(1-r))
    T = R - C
    
    return T

def trap_operator(pixel_signals, *traps):
    """Calculate trap operator for given serial traps."""

    def f(pixel_signals):
        
        y = 0
        for trap in traps:
            y += trap.f(pixel_signals)

        return y

    r = np.exp(-1/tau)
    S_estimate = pixel_signals + f(pixel_signals)
    
    C = f(S_estimate)
    R = np.zeros(C.shape)
    R[:, 1:] = f(S_estimate)[:,:-1]*(1-r)
    R[:, 2:] += np.maximum(0, (f(S_estimate[:, :-2])-f(S_estimate[:, 1:-1]))*r*(1-r))
    T = R - C
    
    return T

def electronics_operator(pixel_signals, scale, tau, num_previous_pixels=4):

    r = np.exp(-1/tau)

    ny, nx = pixel_signals.shape

    offset = np.zeros((num_previous_pixels, ny, nx))
    offset[0, :, :] = scale*pixel_signals[:, :]

    for n in range(1, num_previous_pixels):
        offset[n, :, n:] = scale*pixel_signals[:, :-n]*(r**(n))

    E = np.amax(offset, axis=0)

    return E
        


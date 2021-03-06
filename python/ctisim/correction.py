"""Deferred charge correction module.

To Do:
    * Rename module to something better.
    * Test out new trap operator that takes SerialTraps as args (rather than trap params).
"""
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv
from scipy.special import comb

def cti_inverse_operator(cti, ncols):
    """Calculate a sparse matrix representing CTI operator."""

    b = cti
    a = 1-cti

    diags = np.asarray([[a**i for i in range(1, ncols+1)],
                        [i*b*(a**i) for i in range(1, ncols+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, ncols+1)]])

    D = dia_matrix((diags, [0, -1, -2]), shape=(ncols, ncols))
    invD = inv(D)

    return invD

def trap_inverse_operator(pixel_signals, *traps):
    """Calculate trapping inverse operator for given serial traps."""

    def f(pixel_signals):
        
        y = 0
        for trap in traps:
            y += trap.f(np.maximum(0, pixel_signals))

        return y

    S_estimate = pixel_signals # modify for pixel estimate
    
    C = f(S_estimate)
    R = np.zeros(C.shape)
    R[:, 1:] = f(S_estimate)[:, :-1]
    T = R - C
    
    return T

def electronics_inverse_operator(pixel_signals, scale, tau, 
                                 num_previous_pixels=4):
    """Calculate electronics inverse operator for given parameterization."""

    r = np.exp(-1/tau)

    ny, nx = pixel_signals.shape

    offset = np.zeros((num_previous_pixels, ny, nx))
    offset[0, :, :] = scale*np.maximum(0, pixel_signals[:, :])

    for n in range(1, num_previous_pixels):
        offset[n, :, n:] = scale*np.maximum(0, pixel_signals[:, :-n])*(r**(n))

    L = np.amax(offset, axis=0)

    return L
        


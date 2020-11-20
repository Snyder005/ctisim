"""Deferred charge forward/inverse estimators module.

To Do:
    * Optimize new forward/inverse operators.
"""
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv
from scipy.special import comb

def cti_operator(pixel_signals, cti):
    """Apply classical CTI forward operator to pixel signals."""
    
    results = np.zeros(pixel_signals.shape)
    nrows, ncols = pixel_signals.shape
    
    a = 1-cti
    b = cti
    
    diags = np.asarray([[a**i for i in range(1, ncols+1)],
                        [i*b*(a**i) for i in range(1, ncols+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, ncols+1)]])

    D = dia_matrix((diags, [0, -1, -2]), shape=(ncols, ncols))
    
    for n in range(nrows):
        results[n, :] = D.dot(pixel_signals[n, :])
    
    return results

def cti_inverse_operator(pixel_signals, cti):
    """Apply classical CTI inverse operator to pixel signals."""
    
    results = np.zeros(pixel_signals.shape)
    nrows, ncols = pixel_signals.shape
    
    a = 1-cti
    b = cti
    
    diags = np.asarray([[a**i for i in range(1, ncols+1)],
                        [i*b*(a**i) for i in range(1, ncols+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, ncols+1)]])

    Dinv = inv(dia_matrix((diags, [0, -1, -2]), shape=(ncols, ncols)))
    
    for n in range(nrows):
        results[n, :] = Dinv.dot(pixel_signals[n, :])
    
    return results

def localized_trap_operator(pixel_signals, trap, cti=0.0, num_previous_pixels=4):
    """Apply localizted trapping forward operator to pixel signals."""

    def f(s):
        
        y = trap.f(np.maximum(0, s))
            
        return y
    
    S_estimate = pixel_signals
    a = 1 - cti
    r = np.exp(-1/trap.emission_time)
    
    ## Estimate trap occupancies during readout
    trap_occupancy = np.zeros((num_previous_pixels, pixel_signals.shape[0], pixel_signals.shape[1]))
    for n in range(num_previous_pixels):
        trap_occupancy[n, :, n+1:] = f(S_estimate)[:, :-(n+1)]*(r**n)
    trap_occupancy = np.amax(trap_occupancy, axis=0)
    
    ## Estimate captured charge
    C = f(S_estimate) - trap_occupancy*r
    C[C < 0] = 0.
    
    ## Estimate released charge
    R = np.zeros(pixel_signals.shape)
    R[:, 1:] = trap_occupancy[:, 1:]*(1-r)
    T = R - C
    
    return pixel_signals + a*T

def localized_trap_inverse_operator(pixel_signals, trap, cti=0.0, num_previous_pixels=4):
    """Apply localized trapping inverse operator to pixel signals."""

    def f(s):

        y = trap.f(np.maximum(0, s))
            
        return y
    
    S_estimate = pixel_signals
    a = 1 - cti
    r = np.exp(-1/trap.emission_time)
    
    ## Estimate trap occupancies during readout
    trap_occupancy = np.zeros((num_previous_pixels, pixel_signals.shape[0], pixel_signals.shape[1]))
    for n in range(num_previous_pixels):
        trap_occupancy[n, :, n+1:] = f(S_estimate)[:, :-(n+1)]*(r**n)
    trap_occupancy = np.amax(trap_occupancy, axis=0)
    
    ## Estimate captured charge
    C = f(S_estimate) - trap_occupancy*r
    C[C < 0] = 0.
    
    ## Estimate released charge
    R = np.zeros(pixel_signals.shape)
    R[:, 1:] = trap_occupancy[:, 1:]*(1-r)
    T = R - C
    
    return pixel_signals - a*T

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
        


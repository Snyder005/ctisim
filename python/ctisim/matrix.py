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

def linear_trap_operator(pixel_signals, trapsize, scaling, tau):

    r = np.exp(-1/tau)

    def f(pixel_signals):

        return np.minimum(trapsize, pixel_signals*scaling)

    r = np.exp(-1/tau)
    S_estimate = pixel_signals + f(pixel_signals)

    C = f(S_estimate)
    R = np.zeros(C.shape)
    R[:, 1:] = f(S_estimate)[:,:-1]*(1-r)
    R[:, 2:] += np.maximum(0, (f(S_estimate[:, :-2])-f(S_estimate[:, 1:-1]))*r*(1-r))
    T = R - C
    
    return T

def two_trap_operator(pixel_signals, trapsize1, scaling, trapsize2, f0, k, tau):
    
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

def amplifier_operator(pixel_signals, scale, tau, num_previous_pixels=4):

    r = np.exp(-1/tau)

    ny, nx = pixel_signals.shape

    offset = np.zeros((num_previous_pixels, ny, nx))
    offset[0, :, :] = scale*pixel_signals[:, :]*r

    for n in range(1, num_previous_pixels):
        offset[n, :, n:] = scale*pixel_signals[:, :-n]*(r**(n+1))

    E = np.amax(offset, axis=0)

    return E
        


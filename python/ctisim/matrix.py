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

def one_trap_operator(pixel_signals, trapsize, scaling):
    """Calculate a linear operator for charge trapping."""
    
    def f(pixel_signals):

        return np.minimum(trapsize, pixel_signals*scaling)
    
    S_estimate = pixel_signals + f(pixel_signals)

    T = -f(S_estimate)
    T[:, 1:] += f(S_estimate)[:, :-1]
    
    return T

def two_trap_operator(pixel_signals, trapsize1, scaling, trapsize2, f0, k):

    def f(pixel_signals):

        return np.minimum(trapsize1, pixel_signals*scaling) + trapsize2/(1.+np.exp(-k*(pixel_signals-f0)))
    
    S_estimate = pixel_signals + f(pixel_signals)

    T = -f(S_estimate)
    T[:, 1:] += f(S_estimate)[:, :-1]
    
    return T


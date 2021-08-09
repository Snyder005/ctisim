"""Deferred charge forward/inverse estimators module.

To Do:
    * Optimize new forward/inverse operators.
"""
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv
from scipy.special import comb

def global_cti_operator(inputArr, global_cti):
    """Apply classical CTI forward operator to pixel signals."""
    
    outputArr = np.zeros(inputArr.shape)
    Ny, Nx = inputArr.shape
    
    a = 1-global_cti
    b = global_cti
    
    diags = np.asarray([[a**i for i in range(1, Nx+1)],
                        [i*b*(a**i) for i in range(1, Nx+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, Nx+1)]])

    D = dia_matrix((diags, [0, -1, -2]), shape=(Nx, Nx))
    
    for n in range(Ny):
        outputArr[n, :] = D.dot(inputArr[n, :])
    
    return outputArr

def global_cti_inverse_operator(inputArr, global_cti):
    """Apply classical CTI inverse operator to pixel signals."""
    
    outputArr = np.zeros(inputArr.shape)
    Ny, Nx = inputArr.shape
    
    a = 1-global_cti
    b = global_cti
    
    diags = np.asarray([[a**i for i in range(1, Nx+1)],
                        [i*b*(a**i) for i in range(1, Nx+1)],
                        [comb(i+1, i-1)*(a**i)*(b**2.) for i in range(1, Nx+1)]])

    Dinv = inv(dia_matrix((diags, [0, -1, -2]), shape=(Nx, Nx)))
    
    for n in range(Ny):
        outputArr[n, :] = Dinv.dot(inputArr[n, :])
    
    return outputArr

def localized_trap_operator(inputArr, trap, global_cti=0.0, num_previous_pixels=4):
    """Apply localizted trapping forward operator to pixel signals."""
    
    Ny, Nx = inputArr.shape
    a = 1 - global_cti
    r = np.exp(-1/trap.emission_time)
    
    ## Estimate trap occupancies during readout
    trap_occupancy = np.zeros((num_previous_pixels, Ny, Nx))
    for n in range(num_previous_pixels):
        trap_occupancy[n, :, n+1:] = trap.capture(np.maximum(0, inputArr))[:, :-(n+1)]*(r**n)
    trap_occupancy = np.amax(trap_occupancy, axis=0)
    
    ## Estimate captured charge
    C = trap.capture(np.maximum(0, inputArr)) - trap_occupancy*r
    C[C < 0] = 0.
    
    ## Estimate released charge
    R = np.zeros(inputArr.shape)
    R[:, 1:] = trap_occupancy[:, 1:]*(1-r)
    T = R - C

    outputArr = inputArr + a*T
    
    return outputArr

def localized_trap_inverse_operator(inputArr, trap, global_cti=0.0, num_previous_pixels=4):
    """Apply localized trapping inverse operator to pixel signals."""
    
    Ny, Nx = inputArr.shape
    a = 1 - global_cti
    r = np.exp(-1/trap.emission_time)
    
    ## Estimate trap occupancies during readout
    trap_occupancy = np.zeros((num_previous_pixels, Ny, Nx))
    for n in range(num_previous_pixels):
        trap_occupancy[n, :, n+1:] = trap.capture(np.maximum(0, inputArr))[:, :-(n+1)]*(r**n)
    trap_occupancy = np.amax(trap_occupancy, axis=0)
    
    ## Estimate captured charge
    C = trap.capture(np.maximum(0, inputArr)) - trap_occupancy*r
    C[C < 0] = 0.
    
    ## Estimate released charge
    R = np.zeros(inputArr.shape)
    R[:, 1:] = trap_occupancy[:, 1:]*(1-r)
    T = R - C

    outputArr = inputArr - a*T
    
    return outputArr

def local_offset_operator(inputArr, scale, decay_time, num_previous_pixels=4):

    r = np.exp(-1/decay_time)
    Ny, Nx = inputArr.shape

    offset = np.zeros((num_previous_pixels, Ny, Nx))
    offset[0, :, :] = scale*np.maximum(0, inputArr)
    
    for n in range(1, num_previous_pixels):
        offset[n, :, n:] = scale*np.maximum(0, inputArr[:, :-n])*(r**n)

    L = np.amax(offset, axis=0)
    
    outputArr = inputArr + L

    return outputArr

def local_offset_inverse_operator(inputArr, scale, decay_time, num_previous_pixels=4):

    r = np.exp(-1/decay_time)
    Ny, Nx = inputArr.shape

    offset = np.zeros((num_previous_pixels, Ny, Nx))
    offset[0, :, :] = scale*np.maximum(0, inputArr)

    for n in range(1, num_previous_pixels):
        offset[n, :, n:] = scale*np.maximum(0, inputArr[:, :-n])*(r**n)

    L = np.amax(offset, axis=0)

    outputArr = inputArr - L

    return outputArr

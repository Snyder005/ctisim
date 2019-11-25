import numpy as np

def trap_model(params, pixel_array, flux_array, num_transfers):
    """Model fixed loss and proportional loss from CTI."""

    ctiexp, A, tau = params
    cti = 10**ctiexp

    model_results = np.zeros((flux_array.shape[0], pixel_array.shape[0]))
    for i, flux in enumerate(flux_array):

        model_results[i, :] = A*np.exp(-pixel_array/tau) + \
                              (cti**pixel_array)*num_transfers*flux

    return model_results

def biasdrift_model(params, pixel_array, flux_array, num_transfers):
    """Model bias drift and proportional loss from CTI."""

    ctiexp, A, tau = params
    cti = 10**ctiexp

    model_results = np.zeros((flux_array.shape[0], pixel_array.shape[0]))
    for i, flux in enumerate(flux_array):

        model_results[i, :] = (A*(flux)/10000.)*np.exp(-pixel_array/tau) + \
                              (cti**pixel_array)*num_transfers*flux

    return model_results

def cti_model(ctiexp, pixel_array, flux_array, num_transfers):
    """Model proportional loss from CTI."""
    
    cti = 10**ctiexp

    model_results = np.zeros((flux_array.shape[0], pixel_array.shape[0]))
    for i, flux in enumerate(flux_array):

        model_results[i, :] = (cti**pixel_array)*num_transfers*flux

    return model_results

def overscan_model_error(params, overscan_data, pixel_array, flux_array,
                         num_transfers, model_func=cti_model):
    """Calculate sum of least-squares errors for data fitting."""

    overscan_model = model_func(params, pixel_array, flux_array, num_transfers)
    error = np.sum(np.square(overscan_model-overscan_data))

    return error

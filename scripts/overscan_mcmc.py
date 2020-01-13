import numpy as np
from astropy.io import fits
import argparse
import emcee
import os
import copy
import scipy
from scipy import optimize
import time

from lsst.eotest.sensor.AmplifierGeometry import parse_geom_kwd

from ctisim import ITL_AMP_GEOM
from ctisim.mcmc import TrapModelFitting
from ctisim.model import trap_model, cti_model, biasdrift_model, overscan_model_error
from ctisim.core import LinearTrap, LogisticTrap
from ctisim.utils import save_mcmc_results, save_ccd_results

def summed_charge(A, tau, npixs=10):
    
    x = np.arange(1, npixs+1)
    dc = np.sum(A*np.exp(-x/tau))
    
    return dc

def main(sensor_id, amp, overscan_results_file, walkers=8, steps=500, burn_in=100, 
         threads=1, output_dir='./', model='linear'):

    assert steps > burn_in, "steps must be greater than burnin"

    ## Get amplifier results
    overscan_results = fits.open(overscan_results_file)
    datasec = overscan_results[0].header['DATASEC']
    amp_geom = parse_geom_kwd(datasec)
    xmax = amp_geom['xmax']

    ## Perform for single segment to test
    meanrows_data = overscan_results[amp].data['MEANROW']
    flux_array = overscan_results[amp].data['FLUX']

    ## Initial low signal fitting
    low_indices = flux_array < 10000.
    low_flux_array = flux_array[low_indices]
    low_meanrows_data = meanrows_data[low_indices, xmax:xmax+5]
    pixel_array = np.arange(1, 6)

    guess = (-6.0, 30.0, 0.6)
    bounds = [(-6.3, -5.3), (0.0, None), (0.001, 3.0)]
    args = (low_meanrows_data, pixel_array, low_flux_array, xmax, trap_model)

    lf_results = scipy.optimize.minimize(overscan_model_error, guess, args=args,
                                     bounds=bounds, method='SLSQP')
    ctiexp0, C0, traptau0 = lf_results.x
    trapsize0 = summed_charge(C0, traptau0, 10)

    if trapsize0 > 1.0:

        outfile = os.path.join(output_dir, '{0}_Amp{1}_mcmc_results.fits'.format(sensor_id,
                                                                                 amp))
        a = time.time()
        ## Initialize MCMC walkers

        if model == 'linear':
            params0 = [ctiexp0, trapsize0, traptau0, 0.1, 10.0]
            constraints = [(-6.3, -5.3),
                           (0.0, 20.0),
                           (0.001, 3.0),
                           (0.001, 1.0),
                           (0.0, 100.0)]
            scale_list = [0.1, 0.2, 0.05, 0.02, 10.0]
            ndim = 5
            trap_type = LinearTrap

        elif model == 'logistic':
            params0 = [ctiexp0, trapsize0, traptau0, 20.0, 0.5]
            constraints = [(-6.3, -5.3),
                           (0.0, 20.0),
                           (0.001, 3.0),
                           (0.0, 100.0),
                           (0.1, 1.0)]
            scale_list = [0.1, 0.2, 0.05, 5.0, 0.1]
            ndim = 5
            trap_type=LogisticTrap

        fitter = TrapModelFitting(params0, constraints, ITL_AMP_GEOM, 
                                  trap_type=trap_type)
        p0 = fitter.initialize_walkers(scale_list, walkers)

        ## Perform MCMC optimization
        sampler = emcee.EnsembleSampler(walkers, ndim, fitter.logprobability, threads=threads, 
                                    args=[low_flux_array, meanrows_data[low_indices, :], 6.5, 1])
        sampler.run_mcmc(p0, steps)

        ## Save MCMC chain to FITs file
        save_mcmc_results(sensor_id, amp, sampler.chain, outfile, trap_type)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('amp', type=int)
    parser.add_argument('overscan_results_file', type=str)
    parser.add_argument('-w', '--walkers', type=int, default=10,
                        help='Number of random walkers.')
    parser.add_argument('-s', '--steps', type=int, default=100, 
                        help='Number of random steps to perform.')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of multiprocessing threads.')
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    parser.add_argument('-b', '--burn_in', type=int, default=0)
    parser.add_argument('--model', type=str, default='linear')
    args = parser.parse_args()

    sensor_id = args.sensor_id
    amp = args.amp
    overscan_results_file = args.overscan_results_file
    walkers = args.walkers
    steps = args.steps
    threads = args.threads
    output_dir = args.output_dir
    burn_in = args.burn_in
    model = args.model

    main(sensor_id, amp, overscan_results_file, walkers=walkers, steps=steps, 
         threads=threads, output_dir=output_dir, burn_in=burn_in, model=model)

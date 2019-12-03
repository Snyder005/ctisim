import numpy as np
from astropy.io import fits
import argparse
import emcee
import os
import copy
import scipy
from scipy import optimize

from lsst.eotest.sensor.AmplifierGeometry import parse_geom_kwd

from ctisim import ITL_AMP_GEOM
from ctisim.mcmc import TrapModelFitting
from ctisim.model import trap_model, cti_model, biasdrift_model, overscan_model_error

def summed_charge(A, tau, npixs=10):
    
    x = np.arange(1, npixs+1)
    dc = np.sum(A*np.exp(-x/tau))
    
    return dc

def main(sensor_id, overscan_results_file, walkers=8, steps=500, burnin=50, 
         threads=1, output_dir='./'):

    assert steps > burnin, "steps must be greater than burnin"

    ## Get amplifier results
    overscan_results = fits.open(overscan_results_file)
    datasec = overscan_results[0].header['DATASEC']
    amp_geom = parse_geom_kwd(datasec)
    xmax = amp_geom['xmax']

    prihdu = fits.PrimaryHDU()

    ## Sensor deferred charge parameters
    cti_results = np.zeros(16)
    drift_size_results = np.zeros(16)
    drift_tau_results = np.full(16, np.nan)
    drift_threshold_results = np.zeros(16)
    trap_size_results = np.zeros(16)
    dfactor_results = np.zeros(16)
    trap_tau_results = np.full(16, np.nan)

    overscan_mcmc_hdul = []

    for amp in range(1, 17):

        meanrows_data = overscan_results[amp].data['MEANROW']
        flux_array = overscan_results[amp].data['FLUX']

        ## High signal fitting
        high_indices = (20000. < flux_array) & (flux_array < 140000.)
        high_flux_array = flux_array[high_indices]
        high_meanrows_data = meanrows_data[high_indices, xmax+1:xmax+11]
        pixel_array = np.arange(2, 12)

        guess = [-6, 0.0, 1.7]
        bounds = [(-6, -6), (0.0, None), (0.5, 3.0)]
        args = (high_meanrows_data, pixel_array, high_flux_array, xmax, biasdrift_model)
                 

        hf_results = scipy.optimize.minimize(overscan_model_error, guess, args=args,
                                             bounds=bounds, method='SLSQP')

        _, A, drift_tau = hf_results.x

        ## Get low signal overscan data
        low_indices = flux_array < 10000.
        low_flux_array = flux_array[low_indices]
        low_meanrows_data = meanrows_data[low_indices, xmax:xmax+5]
        pixel_array = np.arange(1, 6)

        ## Initial least squares minimization
        guess = (-6.0, 30.0, 0.6)
        bounds = [(-6.3, -5.3), (0.0, None), (0.001, 3.0)]
        args = (low_meanrows_data, pixel_array, low_flux_array, xmax, trap_model)
        
        lf_results = scipy.optimize.minimize(overscan_model_error, guess, args=args,
                                         bounds=bounds, method='SLSQP')
        ctiexp0, C0, trap_tau0 = lf_results.x
        trapsize0 = summed_charge(C0, trap_tau0, 10)

        ## Record high flux results
        drift_size_results[amp-1] = A/10000.
        drift_tau_results[amp-1] = drift_tau
                
        ## CTI only optimization
        if trapsize0 < 1.0:
            guess = (ctiexp0, A, drift_tau)
            bounds = [(-7, -5.3), (A, A), (drift_tau, drift_tau)] # keep previous results?
            args = (low_meanrows_data, pixel_array, low_flux_array, xmax, biasdrift_model)

            result = scipy.optimize.minimize(overscan_model_error, guess, args=args,
                                             bounds=bounds, method='SLSQP')
            ctiexp, A, drift_tau = result.x

            cti_results[amp-1] = 10**ctiexp

        ## Simulator optimization
        else:
            biasdrift_params = A/10000., drift_tau, 0.0
            ndim = 4
            
            ctiexp = np.random.normal(ctiexp0, scale=0.1, size=walkers)
            densityfactor = np.random.normal(0.1, scale=0.02, size=walkers)
            trap_tau = np.random.normal(trap_tau0, scale=0.05, size=walkers)
            trapsize = np.random.normal(trapsize0, scale=0.2, size=walkers)
            p0 = np.asarray([ctiexp, densityfactor, trap_tau, trapsize]).T

            constraints = {'ctiexp' : (-6.3, -5.3),
                           'densityfactor' : (0.001, 1.0),
                           'tau' : (0.001, 3.0),
                           'trapsize' : (0.0, 20.0)}

            fitter = TrapModelFitting(constraints, ITL_AMP_GEOM, num_oscan_pixels=5)

            sampler = emcee.EnsembleSampler(walkers, ndim, fitter.logprobability, threads=threads, 
                                        args=[low_flux_array, meanrows_data[low_indices, :], 6.5, 1, biasdrift_params])
            sampler.run_mcmc(p0, steps)
            mcmc_results = sampler.chain

            ## Add MCMC results to main results table
            results = mcmc_results.reshape(-1, ndim)
            cti_results[amp-1] = 10**np.median(mcmc_results[:, burnin:, 0])
            dfactor_results[amp-1] = np.median(mcmc_results[:, burnin:, 1])
            trap_tau_results[amp-1] = np.median(mcmc_results[:, burnin:, 2])
            trap_size_results[amp-1] = np.median(mcmc_results[:, burnin:, 3])
                                            
            ## Add TableHDU with MCMC chains
            hdr = fits.Header()
            hdr['AMP'] = amp
            hdr['WALKERS'] = walkers
            hdr['NSTEPS'] = steps
            col1 = fits.Column(name='CTIEXP', array=results[:, 0], format='E')
            col2 = fits.Column(name='TRAPSIZE', array=results[:, 3], format='E')
            col3 = fits.Column(name='TAU', array=results[:, 2], format='E')
            col4 = fits.Column(name='DFACTOR', array=results[:, 1], format='E')
            mcmc_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4], header=hdr)
            overscan_mcmc_hdul.append(mcmc_hdu)

    cti_col = fits.Column(name='CTI', array=cti_results, format='E')
    drift_size_col = fits.Column(name='DRIFT_SIZE', array=drift_size_results, format='E')
    drift_tau_col = fits.Column(name='DRIFT_TAU', array=drift_tau_results, format='E')
    drift_threshold_col = fits.Column(name='DRIFT_THRESHOLD', array=drift_threshold_results, format='E')
    trap_size_col = fits.Column(name='TRAP_SIZE', array=trap_size_results, format='E')
    trap_tau_col = fits.Column(name='TRAP_TAU', array=trap_tau_results, format='E')
    dfactor_col = fits.Column(name='TRAP_DFACTOR', array=dfactor_results, format='E')
    dc_hdu = fits.BinTableHDU.from_columns([cti_col, drift_size_col, drift_tau_col, drift_threshold_col,
                                            trap_size_col, trap_tau_col, dfactor_col])

    prihdu = fits.PrimaryHDU()
    hdul = fits.HDUList([prihdu, dc_hdu] + overscan_mcmc_hdul)

    hdul.writeto(os.path.join(output_dir, 
                              '{0}_overscan_mcmc_results.fits'.format(sensor_id)), 
                 overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('overscan_results_file', type=str)
    parser.add_argument('-w', '--walkers', type=int, default=10,
                        help='Number of random walkers.')
    parser.add_argument('-s', '--steps', type=int, default=500, 
                        help='Number of random steps to perform.')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of multiprocessing threads.')
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    args = parser.parse_args()

    sensor_id = args.sensor_id
    overscan_results_file = args.overscan_results_file
    walkers = args.walkers
    steps = args.steps
    threads = args.threads
    output_dir = args.output_dir

    main(sensor_id, overscan_results_file, walkers=walkers, 
         steps=steps, threads=threads, output_dir=output_dir)

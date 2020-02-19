import argparse
import os
import numpy as np
import emcee
from os.path import join
from astropy.io import fits
import corner
import time

from ctisim.fitting import OverscanFitting, TrapSimulatedModel
from ctisim.core import LinearTrap, LogisticTrap, FloatingOutputAmplifier
from ctisim.utils import ITL_AMP_GEOM, save_mcmc_results

RAFT_NAMES = ['R01', 'R02', 'R03', 
              'R10', 'R13', 'R14',
              'R20', 'R23', 'R24',
              'R31', 'R32', 'R33', 'R34',
              'R41', 'R42', 'R43']

CCD_NAMES = ['S00', 'S01', 'S02',
             'S10', 'S11', 'S12',
             'S20', 'S21', 'S22',]

def main(sensor_id, amp, nsteps, nwalkers):

    ## Get flat field data
    overscan_results = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_results.fits'
    hdul = fits.open(overscan_results)

    data = hdul[amp].data
    signal_all = data['FLATFIELD_SIGNAL']
    indices = (signal_all<30000.)*(signal_all>10000.)
    signal_data = signal_all[indices]
    column_data = data['COLUMN_MEAN'][indices, 512:517]

    ## Get OutputAmplifier fit results
    amp_fit_results = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_fit_results.fits'
    hdul = fits.open(amp_fit_results)
    decay_times = hdul[1].data['DECAY_TIME']
    drift_scales = hdul[1].data['DRIFT_SIZE']
    output_amplifier = FloatingOutputAmplifier(1.0, drift_scales[amp-1]/10000., decay_times[amp-1], noise=0.0, offset=0.0)

    ## Low signal trapping
    mcmc_results = fits.open('/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/lsst-camera-dh/ctisim/examples/output/R20_S02_Amp9_lowtrap_mcmc.fits')

    cti_chain = mcmc_results[1].data
    trapsize_chain = mcmc_results[2].data
    emission_time_chain = mcmc_results[3].data
    scaling_chain = mcmc_results[4].data
    cti = 10**np.median(cti_chain[:, 500:])

    low_trap = LinearTrap(np.median(trapsize_chain[:, 500:]), np.median(emission_time_chain[:, 500:]), 1,
                          np.median(scaling_chain[:, 500:]))

    ## MCMC Fit
    ctiexp = np.log10(cti)
    params0 = (4., 0.5, 12500., 0.0025)
    constraints = [(0, 10.), (0.01, 1.0), (11000., 15000.), (0.0001, 0.004)]

    overscan_fitting = OverscanFitting(params0, constraints, TrapSimulatedModel, start=1, stop=5)

    scale = (0.5, 0.05, 500., 0.001)
    pos = overscan_fitting.initialize_walkers(scale, nwalkers)
    print(pos.shape)
    error = 7.0/np.sqrt(2000.)
    args = (signal_data, column_data, error, ctiexp, ITL_AMP_GEOM, 
            LogisticTrap, output_amplifier)
    kwargs = {'traps' : low_trap}

    sampler = emcee.EnsembleSampler(nwalkers, 4, overscan_fitting.logprobability, args=args, kwargs=kwargs)
    sampler.run_mcmc(pos, nsteps)

    outfile = '{0}_Amp{1}_medtrap_mcmc.fits'.format(sensor_id, amp)
    save_mcmc_results(sensor_id, amp, sampler.chain, outfile, LogisticTrap)

    samples = sampler.chain.reshape((-1, 5))
    fig = corner.corner(samples)
    fig.savefig('{0}_Amp{1}_medtrap_triangle.png'.format(sensor_id, amp))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str, help='Sensor id (e.g. R20_S02)')
    parser.add_argument('amp', type=int, help='Amplifier number (1-16)')
    parser.add_argument('nwalkers', type=int,
                        help='Number of walkers (must be greater than 8).')
    parser.add_argument('nsteps', type=int,
                        help='Number of steps for each chain.')
    args = parser.parse_args()

    main(args.sensor_id, args.amp, args.nsteps, args.nwalkers)

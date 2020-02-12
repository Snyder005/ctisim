import argparse
import os
import numpy as np
import emcee
from os.path import join
from astropy.io import fits
import corner
import time

from ctisim.fitting import OverscanFitting, SimulatedTrapModel
from ctisim.core import LinearTrap, FloatingOutputAmplifier
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

    a = time.time()
    ## Get flat field data
    overscan_results = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_results.fits'
    hdul = fits.open(overscan_results)

    data = hdul[amp].data
    signal_all = data['FLATFIELD_SIGNAL']
    indices = signal_all<10000.
    signal_data = signal_all[indices]
    column_data = data['COLUMN_MEAN'][indices, 512:516]

    ## Get OutputAmplifier fit results
    amp_fit_results = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_fit_results.fits'
    hdul = fits.open(amp_fit_results)
    decay_times = hdul[1].data['DECAY_TIME']
    drift_scales = hdul[1].data['DRIFT_SIZE']
    output_amplifier = FloatingOutputAmplifier(1.0, drift_scales[amp-1]/10000., decay_times[amp-1], noise=0.0, offset=0.0)

    ## MCMC Fit
    params0 = [-6.0, 3.5, 0.4, 0.08]
    constraints = [(-7, -5.3), (0., 10.), (0.01, 1.0), (0.001, 1.0)]
    overscan_fitting = OverscanFitting(params0, constraints, SimulatedTrapModel, start=1, stop=4)

    scale = (0.4, 0.5, 0.05, 0.005)
    pos = overscan_fitting.initialize_walkers(scale, nwalkers)
    error = 7.0/np.sqrt(2000.)

    args = (signal_data, column_data, error, ITL_AMP_GEOM, LinearTrap, output_amplifier)
    sampler = emcee.EnsembleSampler(nwalkers, 4, overscan_fitting.logprobability, args=args)
    sampler.run_mcmc(pos, nsteps)

    outfile = '{0}_{1}_trap_mcmc_results.fits'.format(sensor_id, amp)
    save_mcmc_results(sensor_id, 7, sampler.chain, outfile, LinearTrap)

    import corner
    samples = sampler.chain.reshape((-1, 4))
    fig = corner.corner(samples)
    fig.savefig('{0}_{1}_triangle.png'.format(sensor_id, amp))
    b = time.time()
    print(b-a)

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

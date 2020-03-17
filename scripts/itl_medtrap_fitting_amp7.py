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
from ctisim.utils import ITL_AMP_GEOM, E2V_AMP_GEOM, save_mcmc_results, OverscanParameterResults

RAFT_NAMES = ['R01', 'R02', 'R03', 
              'R10', 'R13', 'R14',
              'R20', 'R23', 'R24',
              'R31', 'R32', 'R33', 'R34',
              'R41', 'R42', 'R43']

CCD_NAMES = ['S00', 'S01', 'S02',
             'S10', 'S11', 'S12',
             'S20', 'S21', 'S22',]

def main(sensor_id, amp, nsteps, nwalkers, output_dir='./'):

    ## Config variables
    start = 1
    stop= 5
    ccd_type = 'itl'
    min_signal = 10000.
    max_signal = 30000.
    read_noise = 7.0

    ## Get flat field data
    overscan_results_file = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_results.fits'
    hdulist = fits.open(overscan_results_file)

    ## CCD geometry info
    if ccd_type == 'itl':
        ncols = ITL_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width
    elif ccd_type == 'e2v':
        ncols = E2V_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width

    signals_all = hdulist[amp].data['FLATFIELD_SIGNAL']
    data_all = hdulist[amp].data['COLUMN_MEAN'][:, ncols+start-1:ncols+stop]
    indices = (signals_all<max_signal)*(signals_all>min_signal)
    signals = signals_all[indices]
    data = data_all[indices]

    ## Get OutputAmplifier fit results
    parameter_results_file = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_parameter_results.fits'
    parameter_results = OverscanParameterResults.from_fits(parameter_results_file)
    output_amplifier = parameter_results.single_output_amplifier(amp, 1.0)

    ## Low signal trap
    mcmc_results = fits.open('/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/lsst-camera-dh/ctisim/examples/output/R20_S02_Amp{0}_lowtrap_mcmc.fits'.format(amp))

    trapsize = np.median(mcmc_results[2].data[:, 500:])
    emission_time = np.median(mcmc_results[3].data[:, 500])
    scaling = np.median(mcmc_results[4].data[:, 500])
    cti = parameter_results.cti_results[amp]

    low_trap = LinearTrap(trapsize, emission_time, 1, scaling)

    ## MCMC Fit
    ctiexp = np.log10(cti)
#    params0 = (4., 0.5, 12500., 0.0025)
#    constraints = [(0, 10.), (0.01, 1.0), (11000., 15000.), (0.0001, 0.004)]
    params0 = (37., 0.5, 17700., 0.001)
    constraints = [(30., 45.), (0.01, 1.0), (15000., 19000.), (0.0001, 0.004)]

    overscan_fitting = OverscanFitting(params0, constraints, TrapSimulatedModel, 
                                       start=start, stop=stop)

    scale = (0.5, 0.05, 500., 0.001)
    pos = overscan_fitting.initialize_walkers(scale, nwalkers)
    args = (signals, data, read_noise/np.sqrt(2000.), ctiexp, ITL_AMP_GEOM, 
            LogisticTrap, output_amplifier)
    kwargs = {'traps' : low_trap}

    sampler = emcee.EnsembleSampler(nwalkers, 4, overscan_fitting.logprobability, 
                                    args=args, kwargs=kwargs)
    sampler.run_mcmc(pos, nsteps)

    outfile = join(output_dir, '{0}_Amp{1}_medtrap_mcmc.fits'.format(sensor_id, amp))
    save_mcmc_results(sensor_id, amp, sampler.chain, outfile, LogisticTrap)

    samples = sampler.chain.reshape((-1, 5))
    fig = corner.corner(samples)
    fig.savefig(join(output_dir, '{0}_Amp{1}_medtrap_triangle.png'.format(sensor_id, amp)))

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

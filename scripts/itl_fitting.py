import numpy as np
from astropy.io import fits
import argparse
import emcee
import os

from lsst.eotest.sensor.AmplifierGeometry import parse_geom_kwd

from ctisim import ITL_AMP_GEOM
from ctisim.core import LogisticTrap
from ctisim.mcmc import TrapModelFitting
from ctisim.utils import save_mcmc_results

def main(sensor_id, amp, overscan_results_file, walkers=8, steps=500, burn_in=100,
         threads=1, output_dir='./', model='logistic'):

    assert steps > burn_in, "steps must be greater than burnin"

    ## Get amplifier results
    overscan_results = fits.open(overscan_results_file)
    datasec = overscan_results[0].header['DATASEC']
    amp_info = parse_geom_kwd(datasec)
    xmax = amp_info['xmax']
    data_all = overscan_results[amp].data['MEANROW']
    signals_all = overscan_results[amp].data['FLUX']

    ## Signal regime for trap
    low = 15000.
    high = 40000.

    indices = (signals_all > low)*(signals_all < high)
    signals = signals_all[indices]
    data = data_all[indices, :]

    params0 = [-6, 50, 1.8, 25000., 0.5]
    constraints = [(-6.3, -5.3),
                   (0.0, 200.0),
                   (0.1, 3.0),
                   (15000., 40000.),
                   (0.1, 1.0)]
    scale_list = [0.1, 10, 0.1, 5000., 0.05]
    ndim = len(params0)
    
    fitter = TrapModelFitting(params0, constraints, ITL_AMP_GEOM,
                              trap_type = LogisticTrap)
    p0 = fitter.initialize_walkers(scale_list, walkers)

    ## Perform MCMC optimization
    sampler = emcee.EnsembleSampler(walkers, ndim, fitter.logprobability, threads=threads,
                                    args=[signals, data, 6.5, 1])
    sampler.run_mcmc(p0, steps)

    ## Save MCMC chain to FITs file
    outfile = os.path.join(output_dir, 
                           '{0}_Amp{1}_mcmc_results.fits'.format(sensor_id, amp))
    save_mcmc_results(sensor_id, amp, sampler.chain, outfile, LogisticTrap)
    
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
    parser.add_argument('--model', type=str, default='logistic')
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

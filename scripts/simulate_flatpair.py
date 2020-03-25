from ctisim import ITL_AMP_GEOM
from ctisim import SerialTrap, OutputAmplifier, ImageSimulator
from astropy.io import fits
import numpy as np
import argparse
import os

def main(signal, eotest_results, mcmc_results, template_file, output_dir='./'):

    amp_geom = ITL_AMP_GEOM

    ## Noise and gain from eotest results
    with fits.open(eotest_results) as hdulist:
        data = hdulist[1].data
        noise = data['TOTAL_NOISE']
        gains = data['GAIN']

    offsets = {i : np.random.normal(27000.0, 2000.0) for i in range(1, 17)}
    output_amps = {i : OutputAmplifier(gains[i-1], noise[i-1], offsets[i]) for i in range(1, 17)}
    
    for i in range(2):

        imsim = ImageSimulator.from_amp_geom(amp_geom, output_amps)
        imsim.update_parameters(mcmc_results)
        imsim.flatfield_exp(signal)

        outfile = os.path.join(output_dir, 'Sim_flat_flat{0}_{1:0.1f}_sim.fits'.format(i, signal))
        imarr_results = imsim.simulate_readout(template_file, outfile=outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('signal', type=float, help='Desired flat field signal level.')
    parser.add_argument('eotest_results', type=str)
    parser.add_argument('mcmc_results', type=str)
    parser.add_argument('template_file', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    signal = args.signal
    eotest_results = args.eotest_results
    mcmc_results = args.mcmc_results
    template_file = args.template_file
    output_dir = args.output_dir
    main(signal, eotest_results, mcmc_results, template_file, output_dir=output_dir)

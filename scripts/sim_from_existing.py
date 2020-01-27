from ctisim import LinearTrap, LogisticTrap, OutputAmplifier, ImageSimulator
from astropy.io import fits
import numpy as np
import argparse
import os
from os.path import join

def main(infile, mcmc_results, output_dir='./', bias_frame=None):

    ## Output filename
    basename = os.path.basename(infile)[:-5]
    outfile = join(output_dir, '{0}_processed.fits'.format(basename))
    print(outfile)

    output_amps = {}
    traps_dict = {}
    cti_dict = {}

    mcmc_results = fits.open('../examples/data/ITL_overscan_mcmc_results.fits')
    mcmc_data = mcmc_results[1].data

    for amp in range(1, 17):

        ## Trap parameters
        cti = mcmc_data['CTI'][amp-1]
        cti_dict[amp] = cti
        size = mcmc_data['TRAP_SIZE'][amp-1]
        if size == 0.0:
            traps_dict[amp] = None
        else:
            emission_time = mcmc_data['TRAP_TAU'][amp-1]
            scaling = mcmc_data['TRAP_DFACTOR'][amp-1]
            trap = LinearTrap(size, emission_time, 1, scaling, 0.0)
            if amp == 7:
                new_trap = LogisticTrap(40, 0.5, 1, 18000., 0.0010)
                traps_dict[amp] = [trap, new_trap]
            else:
                traps_dict[amp] = trap

        ## Bias hysteresis parameters
        drift_scale = mcmc_data['DRIFT_SIZE'][amp-1]
        decay_time = mcmc_data['DRIFT_TAU'][amp-1]
        threshold = mcmc_data['DRIFT_THRESHOLD'][amp-1]
        output_amps[amp] = OutputAmplifier(1.0, 6.5, offset=0.0, drift_scale=drift_scale,
                                           decay_time=decay_time, threshold=threshold)
    
    imsim = ImageSimulator.image_from_fits(infile, output_amps, cti_dict=cti_dict,
                                           traps_dict=traps_dict, 
                                           bias_frame=bias_frame)
    imarr_results = imsim.simulate_readout(infile, 
                                           outfile=outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Existing FITs file image.')
    parser.add_argument('mcmc_results', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    infile = args.infile
    mcmc_results = args.mcmc_results
    output_dir = args.output_dir
    main(infile, mcmc_results, output_dir=output_dir)

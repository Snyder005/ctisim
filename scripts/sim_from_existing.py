from ctisim import SerialTrap, OutputAmplifier, ImageSimulator
from astropy.io import fits
import numpy as np
import argparse
import os
from os.path import join

def main(infile, mcmc_results, output_dir='./', bias_frame=None):

    output_amps = {i : OutputAmplifier(1.0, 0.0) for i in range(1, 17)}
    
    imsim = ImageSimulator.image_from_fits(infile, output_amps, bias_frame=bias_frame)
    imsim.update_parameters(mcmc_results)
    imarr_results = imsim.simulate_readout(infile, 
                                           outfile=join(output_dir, 'example_image.fits'))

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

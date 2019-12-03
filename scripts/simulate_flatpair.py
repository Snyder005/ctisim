from ctisim import ITL_AMP_GEOM
from ctisim import SerialRegister, SerialTrap, ReadoutAmplifier, ImageSimulator
from astropy.io import fits
import numpy as np
import argparse
import os

def main(signal, eotest_results, mcmc_results, template_file, output_dir='./'):

    amp_geom = ITL_AMP_GEOM

    length = amp_geom['ncols'] + amp_geom['num_serial_prescan']
    serial_registers = SerialRegister.from_mcmc_results(mcmc_results, length=length)
    
    ## Random offsets
    offsets = {i : np.random.normal(27000.0, 2000.0) for i in range(1, 17)}

    ## Initialize image simulator
    readout_amps = ReadoutAmplifier.from_eotest_results(eotest_results, mcmc_results=mcmc_results, offsets=offsets)
    
    for i in range(2):

        imsim = ImageSimulator.from_amp_geom(amp_geom, readout_amps, serial_registers)
        imsim.flatfield_exp(signal)

        outfile = os.path.join(output_dir, 'Sim_flat_flat{0}_{1:0.1f}_sim.fits'.format(i, signal))
        imarr_results = imsim.serial_readout(template_file, outfile=outfile, overwrite=True)

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

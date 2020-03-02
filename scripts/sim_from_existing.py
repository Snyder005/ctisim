from ctisim import LinearTrap, LogisticTrap, FloatingOutputAmplifier, ImageSimulator
from astropy.io import fits
import numpy as np
import argparse
import os
from os.path import join

def main(infile, output_dir='./', bias_frame=None):

    ## Output filename
    basename = os.path.basename(infile)[:-5]
    outfile = join(output_dir, '{0}_processed.fits'.format(basename))
    print(outfile)

    output_amps = {}
    traps_dict = {}
    cti_dict = {}

    ## Overscan Results
    oscan_results = fits.open('/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/Data/BOT/6790D_linearity/R20/S02/R20_S02_overscan_fit_results.fits')
    decay_times = oscan_results[1].data['DECAY_TIME']
    drift_scales = oscan_results[1].data['DRIFT_SIZE']

    for amp in range(1, 17):

        try:
            output_amps[amp] = FloatingOutputAmplifier(1.0, drift_scales[amp-1]/10000.,
                                                       decay_times[amp-1], noise=0.0, offset=0.0)
        except:
            output_amps[amp] = FloatingOutputAmplifier(1.0, 2.4/10000., 2.0, noise=0.0, offset=0.0)

        ## Add low signal traps
        try:
            low_results = fits.open('/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/lsst-camera-dh/ctisim/examples/output/R20_S02_Amp{0}_lowtrap_mcmc.fits'.format(amp))

            cti_chain = low_results[1].data
            trapsize_chain = low_results[2].data
            emission_time_chain = low_results[3].data
            scaling_chain = low_results[4].data
            cti_dict[amp] = 10**np.median(cti_chain[:, 500:])
            traps_dict[amp] = [LinearTrap(np.median(trapsize_chain[:, 500:]),
                                          np.median(emission_time_chain[:, 500:]), 1,
                                          np.median(scaling_chain[:, 500:]))]
        except:
            traps_dict[amp] = None
            cti_dict[amp] = 1.E-6 ## Change this with actual CTI values

        ## Add high signal traps
        try:
            high_results = fits.open('/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/lsst-camera-dh/ctisim/examples/output/R20_S02_Amp{0}_medtrap_mcmc.fits'.format(amp))

            trapsize_chain = high_results[1].data
            emission_time_chain = high_results[2].data
            f0_chain = high_results[3].data
            k_chain = high_results[4].data
            
            traps_dict[amp].append(LogisticTrap(np.median(trapsize_chain[:, 500:]),
                                                np.median(emission_time_chain[:, 500:]), 1,
                                                np.median(f0_chain[:, 500:]),
                                                np.median(k_chain[:, 500:])))
        except:
            pass

    imsim = ImageSimulator.from_image_fits(infile, output_amps, cti=cti_dict,
                                           traps=traps_dict, 
                                           bias_frame=bias_frame)
    imarr_results = imsim.simulate_readout(infile, 
                                           outfile=outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Existing FITs file image.')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    infile = args.infile
    output_dir = args.output_dir
    main(infile, output_dir=output_dir)

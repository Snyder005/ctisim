import numpy as np
import argparse
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import warnings
import os

from ctisim.matrix import cti_inverse_operator, electronics_operator, trap_operator
from ctisim import LinearTrap, LogisticTrap, SplineTrap
from ctisim import ImageSimulator
from ctisim import BaseOutputAmplifier, FloatingOutputAmplifier
from lsst.eotest.sensor import MaskedCCD
from lsst.eotest.fitsTools import fitsWriteto
import lsst.eotest.image_utils as imutils

def main(sensor_id, infile, output_dir='./', cti=None, do_trapping=False, 
         do_electronics=False):

    processed_file = os.path.join(output_dir, 
                                  '{0}_processed.fits'.format(sensor_id))
    corrected_file = os.path.join(output_dir, 
                                  '{0}_corrected.fits'.format(sensor_id))

    ## CTI parameters
    if cti is None:
        do_cti = False
        cti = 0.0
    else:
        do_cti = True
    cti_dict = {amp : cti for amp in range(1, 17)}
    print(cti_dict)

    ## Trapping parameters
    if do_trapping:
        traps = [LinearTrap(4.0, 0.4, 1, 0.08),
                 LogisticTrap(1000.0, 0.4, 1, 60000., 0.0002)]
    else:
        traps = None
    traps_dict = {amp : traps for amp in range(1, 17)}
    print(traps_dict)

    ## Electronics parameters
    if do_electronics:
        output_amps = {amp : FloatingOutputAmplifier(1.0, 0.0002, 2.4) for amp in range(1, 17)}
    else:
        output_amps = {amp : BaseOutputAmplifier(1.0) for amp in range(1, 17)}
    print(output_amps)

    ## Process infile
    imsim = ImageSimulator.from_image_fits(infile, output_amps, 
                                           cti=cti_dict, traps=traps_dict)
    imarr_results = imsim.image_readout(infile, outfile=processed_file)

    ccd = MaskedCCD(processed_file)
    hdulist = fits.HDUList()
    with fits.open(processed_file) as template:
        hdulist.append(template[0])
        hdulist[0].header['ORIGFILE'] = hdulist[0].header['FILENAME']
        hdulist[0].header['FILENAME'] = corrected_file

        for amp in imutils.allAmps(processed_file):

            imarr = ccd.bias_subtracted_image(amp).getImage().getArray()

            ## Electronics Correction
            if do_electronics:
                E = electronics_operator(imarr, scale, decay_time, num_previous_pixels=15)
            else:
                corrected_imarr = imarr

            ## Trap Correction
            if do_trapping:
                T = trap_operator(imarr, *traps)
                corrected_imarr = corrected_imarr - (1-cti)*T
            else:
                pass

            ## CTI Correction
            if do_cti:
                Dinv_cti = cti_inverse_operator(cti_results[amp], imarr.shape[1])
                for i in range(imarr.shape[0]):

                    corrected_imarr[i, :] = Dinv_cti*corrected_imarr[i, :]
            else:
                pass

            hdulist.append(fits.ImageHDU(data=corrected_imarr,
                                     header=template[amp].header))
            with warnings.catch_warnings():
                for warning in (UserWarning, AstropyWarning,
                                AstropyUserWarning):
                    warnings.filterwarnings('ignore', category=warning,
                                            append=True)
                fitsWriteto(hdulist, corrected_file, overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('--cti', type=float, default=None)
    parser.add_argument('--do_trapping', action='store_true')
    parser.add_argument('--do_electronics', action='store_true')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, output_dir=args.output_dir,
         cti=args.cti, do_trapping=args.do_trapping,
         do_electronics=args.do_electronics)

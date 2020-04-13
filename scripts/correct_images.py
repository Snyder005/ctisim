import argparse
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import warnings
import os
import pickle

from lsst.eotest.sensor import MaskedCCD
from lsst.eotest.fitsTools import fitsWriteto
from ctisim.utils import OverscanParameterResults
from ctisim.matrix import electronics_operator, trap_operator

def main(sensor_id, infile, main_dir, gain_file=None, output_dir='./'):

    ## Get existing parameter results
    param_file = os.path.join(main_dir, 
                              '{0}_parameter_results.fits'.format(sensor_id))
    param_results = OverscanParameterResults.from_fits(param_file)
    cti_results = param_results.cti_results
    drift_scales = param_results.drift_scales
    decay_times = param_results.decay_times
    thresholds = param_results.thresholds

    ## Get gains
    if gain_file is not None:
        with open(gain_file, 'rb') as f:
            gain_results = pickle.load(f)
            gains = gain_results.get_amp_gains(sensor_id)
    else:
        gains = {i : 1.0 for i in range(1, 17)}

    ## Output filename
    base = os.path.splitext(os.path.basename(infile))[0]
    outfile = os.path.join(output_dir, '{0}_corrected.fits'.format(base))

    ## Correct image
    bias_frame = os.path.join(main_dir, '{0}_superbias.fits'.format(sensor_id))
    ccd = MaskedCCD(infile, bias_frame=bias_frame)
    hdulist = fits.HDUList()
    with fits.open(infile) as template:
        hdulist.append(template[0])

        for amp in range(1, 17):

            imarr = ccd.bias_subtracted_image(amp).getImage().getArray()*gains[amp]

            ## Electronics Correction
            if drift_scales[amp] > 0.:
                E = electronics_operator(imarr, drift_scales[amp], 
                                         decay_times[amp], 
                                         thresholds[amp],
                                         num_previous_pixels=15)
                corrected_imarr = imarr - E
            else:
                corrected_imarr = imarr

            ## Trap Correction
            trap_file = os.path.join(main_dir, 
                                     '{0}_amp{1}_traps.pkl'.format(sensor_id, amp))
            spltrap = pickle.load(open(trap_file, 'rb'))
            T = trap_operator(imarr, spltrap)
            corrected_imarr = corrected_imarr - (1-cti_results[amp])*T

            hdulist.append(fits.ImageHDU(data=corrected_imarr,
                                     header=template[amp].header))
            with warnings.catch_warnings():
                for warning in (UserWarning, AstropyWarning,
                                AstropyUserWarning):
                    warnings.filterwarnings('ignore', category=warning,
                                            append=True)
                fitsWriteto(hdulist, outfile, overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    parser.add_argument('--gain_file', '-g', type=str, default=None)
    args = parser.parse_args()

    main(args.sensor_id, args.infile, args.main_dir, 
         gain_file=args.gain_file, output_dir=args.output_dir)


        

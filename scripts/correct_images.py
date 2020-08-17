import argparse
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import warnings
from os.path import join, splitext
import pickle

from lsst.eotest.sensor import MaskedCCD
from lsst.eotest.fitsTools import fitsWriteto
from ctisim.utils import OverscanParameterResults
from ctisim.correction import electronics_operator, trap_operator

def main(sensor_id, infile, main_dir, gain_file=None, output_dir='./', no_bias=False):

    ## Get existing parameter results
    param_file = join(main_dir, 
                              '{0}_parameter_results.fits'.format(sensor_id))
    param_results = OverscanParameterResults.from_fits(param_file)
    cti_results = param_results.cti_results
    drift_scales = param_results.drift_scales
    decay_times = param_results.decay_times
    trap_files = [join(main_dir, 
                       '{0}_amp{1}_trap.pkl'.format(sensor_id, i)) for i in range(1, 17)]

    ## Get gains
    if gain_file is not None:
        with open(gain_file, 'rb') as f:
            gain_results = pickle.load(f)
            gains = gain_results.get_amp_gains(sensor_id)
    else:
        gains = {i : 1.0 for i in range(1, 17)}

    ## Output filename
    base = splitext(os.path.basename(infile))[0]
    outfile = join(output_dir, '{0}_corrected.fits'.format(base))

    ## Include bias frame for calibration
    if no_bias:
        bias_frame = None
    else:
        bias_frame = join(main_dir, '{0}_superbias.fits'.format(sensor_id))
    ccd = MaskedCCD(infile, bias_frame=bias_frame)

    hdulist = fits.HDUList()
    with fits.open(infile) as template:
        hdulist.append(template[0])

        ## Perform correction amp by amp
        for amp in range(1, 17):

            imarr = ccd.bias_subtracted_image(amp).getImage().getArray()*gains[amp]

            ## Electronics Correction
            if drift_scales[amp] > 0.:
                Linv = electronics_inverse_operator(imarr, drift_scales[amp], 
                                                    decay_times[amp],
                                                    num_previous_pixels=15)
                corrected_imarr = imarr - Linv
            else:
                corrected_imarr = imarr

            ## Trap Correction
            spltrap = pickle.load(open(trap_files[amp], 'rb'))
            Tinv = trap_inverse_operator(corrected_imarr, spltrap)
            corrected_imarr = corrected_imarr - (1-cti_results[amp])*Tinv

            ## Reassemble HDUList
            hdulist.append(fits.ImageHDU(data=corrected_imarr/gains[amp],
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
    parser.add_argument('--no_bias', '-n', action='store_true')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, args.main_dir, 
         gain_file=args.gain_file, output_dir=args.output_dir,
         no_bias=args.no_bias)


        

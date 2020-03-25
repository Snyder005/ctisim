import numpy as np
import argparse
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import warnings
import os
from scipy.sparse.linalg import inv

from ctisim.matrix import cti_operator
from lsst.eotest.sensor import MaskedCCD
from lsst.eotest.fitsTools import fitsWriteto
import lsst.eotest.image_utils as imutils

def trap_operator_new(pixel_signals, trapsize1, scaling, trapsize2, f0, k, tau):
    
    def f(pixel_signals):

        return np.minimum(trapsize1, pixel_signals*scaling) + trapsize2/(1.+np.exp(-k*(pixel_signals-f0)))
    
    r = np.exp(-1/tau)
    S_estimate = pixel_signals + f(pixel_signals)
    
    C = f(S_estimate)
    R = np.zeros(C.shape)
    R[:, 1:] = f(S_estimate)[:,:-1]*(1-r)
    R[:, 2:] += np.maximum(0, (f(S_estimate[:, :-2])-f(S_estimate[:, 1:-1]))*r*(1-r))
    T = R - C
    
    return T

def main(sensor_id, infiles, bias_frame=None, output_dir='./'):

    outfile = os.path.join(output_dir, 'test.fits')
    gains = {i : 1.0 for i in range(1, 17)}

    cti_results = {i: 0.0 for i in range(1, 17)}
    trapsize1, scaling = 4.0, 0.08
#    trapsize2, f0, k = 40.0, 17500., 0.001, 
    trapsize2, f0, k = 1000.0, 60000., 0.0002
    tau = 0.4

    for i, infile in enumerate(infiles):

        outfile = os.path.join(output_dir, '{0}_{1:03d}_corrected.fits'.format(sensor_id, i))

        ccd = MaskedCCD(infile, bias_frame=bias_frame)

        hdulist = fits.HDUList()
        with fits.open(infile) as template:
            hdulist.append(template[0])
            hdulist[0].header['ORIGFILE'] = hdulist[0].header['FILENAME']
            hdulist[0].header['FILENAME'] = outfile

            for amp in imutils.allAmps(infile):

                imarr = ccd.bias_subtracted_image(amp).getImage().getArray()
                imarr *= gains[amp]

                ## Trap Correction
                corrected_imarr = np.zeros(imarr.shape)
                T = trap_operator_new(imarr, trapsize1, scaling, trapsize2, f0, k, tau)
                corrected_imarr = imarr - (1-cti_results[amp])*T

                ## CTI Correction
#                D_cti = cti_operator(cti_results[amp], imarr.shape[1])
#                Dinv_cti = inv(D_cti)
#                for i in range(imarr.shape[0]):

#                    corrected_imarr[i, :] = Dinv_cti*imarr[i, :]

                corrected_imarr /= gains[amp]
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
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.infiles, output_dir=args.output_dir)

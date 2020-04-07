import argparse
import errno
import os
import numpy as np
from os.path import join
import scipy
from astropy.io import fits
from lmfit import Minimizer, Parameters

from ctisim import ITL_AMP_GEOM, E2V_AMP_GEOM
from ctisim.fitting import SimpleModel
from ctisim.utils import OverscanParameterResults

RAFT_NAMES = ['R01', 'R02', 'R03', 
              'R10', 'R13', 'R14',
              'R20', 'R23', 'R24',
              'R31', 'R32', 'R33', 'R34',
              'R41', 'R42', 'R43']

CCD_NAMES = ['S00', 'S01', 'S02',
             'S10', 'S11', 'S12',
             'S20', 'S21', 'S22',]
def main(directory):

    ## Config variables
    start = 2
    stop = 20
    ccd_type = 'itl'
    max_signal = 160000.
    error = 7.0/np.sqrt(2000.)

    num_failures = 0
    for raft_name in RAFT_NAMES:

        raft_output_dir = join(directory, raft_name)
        
        for ccd_name in CCD_NAMES:

            ## Get existing overscan analysis results
            ccd_output_dir = join(directory, raft_name, ccd_name)
            sensor_id = '{0}_{1}'.format(raft_name, ccd_name)
            try:
                hdulist = fits.open(join(ccd_output_dir, 
                                         '{0}_overscan_results.fits'.format(sensor_id)))

            except FileNotFoundError:
                continue

            cti_results = {i : 0.0 for i in range(1, 17)}
            drift_scales = {i : 0.0 for i in range(1, 17)}
            decay_times = {i : 0.0 for i in range(1, 17)}

            ## CCD geometry info
            if ccd_type == 'itl':
                ncols = ITL_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width
            if ccd_type == 'e2v':
                ncols = E2V_AMP_GEOM.nx + E2V_AMP_GEOM.prescan_width

            for amp in range(1, 17):

                ## Signals
                all_signals = hdulist[amp].data['FLATFIELD_SIGNAL']
                signals = all_signals[all_signals<max_signal]

                ## Data
                data = hdulist[amp].data['COLUMN_MEAN'][all_signals<max_signal, 
                                                        start:stop+1]

                params = Parameters()
                params.add('ctiexp', value=-6, min=-7, max=-5, vary=False)
                params.add('trapsize', value=0.0, min=0., max=10., vary=False)
                params.add('scaling', value=0.08, min=0, max=1.0, vary=False)
                params.add('emissiontime', value=0.4, min=0.1, max=1.0, vary=False)
                params.add('driftscale', value=0.0006, min=0., max=0.001)
                params.add('decaytime', value=1.5, min=0.1, max=4.0)
                params.add('threshold', value=5000.0, min=0.0, max=150000.)

                model = SimpleModel()

                minner = Minimizer(model.difference, params, 
                                   fcn_args=(signals, data, error, ncols),
                                   fcn_kws={'start' : start, 'stop' : stop})
                result = minner.minimize()

                if result.success:

                    try:
                        cti = 10**result.params['ctiexp']
                    except KeyError:
                        cti = result.params['cti']
                    drift_scale = result.params['driftscale']
                    decay_time = result.params['decaytime']
                    cti_results[amp] = cti
                    drift_scales[amp] = drift_scale
                    decay_times[amp] = decay_time
                else:
                    num_failures += 1
                    print(sensor_id, amp)
                    print(fit_results)

            outfile = os.path.join(ccd_output_dir, 
                                   '{0}_parameter_results.fits'.format(sensor_id))
            parameter_results = OverscanParameterResults(sensor_id, cti_results, 
                                                         drift_scales, decay_times)
            parameter_results.write_fits(outfile, overwrite=True)

    print('There were {0} failures in the overscan fit'.format(num_failures))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='File path to base directory of overscan results.')
    args = parser.parse_args()

    main(args.directory)


import argparse
import errno
import os
import numpy as np
from os.path import join
import scipy
from astropy.io import fits

from ctisim import ITL_AMP_GEOM
from ctisim.fitting import OverscanFitting, BiasDriftModel
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

    start = 3
    stop = 15

    num_failures = 0
    for raft_name in RAFT_NAMES:

        raft_output_dir = join(directory, raft_name)
        
        for ccd_name in CCD_NAMES:

            ccd_output_dir = join(directory, raft_name, ccd_name)
            sensor_id = '{0}_{1}'.format(raft_name, ccd_name)

            try:
                hdul = fits.open(join(ccd_output_dir, 
                                      '{0}_overscan_results.fits'.format(sensor_id)))
            except FileNotFoundError:
                continue

            cti_results = {i : 0.0 for i in range(1, 17)}
            drift_scales = {i : 0.0 for i in range(1, 17)}
            decay_times = {i : np.nan for i in range(1, 17)}

            for i in range(1, 17):

                ncols = ITL_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width

                noise = 7.5/np.sqrt(2000.)
                ## Offset hysteresis fitting
                hdu_data = hdul[i].data
                signal_all = hdu_data['FLATFIELD_SIGNAL']
                oscan_data_all = hdu_data['COLUMN_MEAN'][:, ncols+start-1:ncols+stop]

                indices = (signal_all < 140000.)*(signal_all > 20000.)
                signals = signal_all[indices]
                oscandata = oscan_data_all[indices]

                params0 = [-7.0, 2.0, 3.0]
                constraints = [(-7., -7.), (0., 10.), (0.01, 4.0)]
                fitting_task = OverscanFitting(params0, constraints, BiasDriftModel, 
                                               start=start, stop=stop)
                fit_results = scipy.optimize.minimize(fitting_task.negative_loglikelihood, 
                                                      params0, 
                                                      args=(signals, oscandata, noise, ncols),
                                                      bounds=constraints, method='SLSQP')

                success = fit_results.success
                if success:
                    ctiexp, drift_scale, decay_time = fit_results.x
                    cti_results[i-1] = 10**ctiexp
                    drift_scales[i-1] = drift_scale
                    decay_times[i-1] = decay_time
                else:
                    num_failures += 1
                    print(sensor_id, i)
                    print(fit_results)

            outfile = join(ccd_output_dir,
                           '{0}_overscan_fit_results.fits'.format(sensor_id))
            results = OverscanParameterResults(sensor_id, cti_results, drift_scales, 
                                               decay_times)
            results.write_fits(outfile, overwrite=True)
    print('There were {0} failures in the overscan fit'.format(num_failures))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='File path to base directory of overscan results.')
    args = parser.parse_args()

    main(args.directory)

import argparse
import errno
import os
import numpy as np
from os.path import join
import scipy
from astropy.io import fits

from ctisim import ITL_AMP_GEOM
from ctisim.fitting import OverscanFitting, BiasDriftSimpleModel
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

    start = 1
    stop = 3

    num_failures = 0
    for raft_name in ['R20']:
        
        raft_output_dir = join(directory, raft_name)
        
        for ccd_name in CCD_NAMES[:1]:

            ccd_output_dir = join(directory, raft_name, ccd_name)
            sensor_id = '{0}_{1}'.format(raft_name, ccd_name)

            try:
                hdul = fits.open(join(ccd_output_dir, 
                                      '{0}_overscan_results.fits'.format(sensor_id)))
            except FileNotFoundError:
                continue

            try:
                oscan_fit_results = fits.open('{0}_overscan_fit_results.fits'.format(sensor_id))
            except FileNotFoundError:
                continue

            for i in range(1, 17):

                ncols = ITL_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width

                noise = 7.5/np.sqrt(2000.)
                hdu_data = hdul[i].data
                signal_all = hdu_data['FLATFIELD_SIGNAL']
                oscan_data_all = hdu_data['COLUMN_MEAN'][:, ncols+start-1:ncols+stop]

                indices = (signal_all < 5000.)
                signals = signal_all[indices]
                oscandata = oscan_data_all[indices]

                ## Get previous high signal fit results
                drift_scale = oscan_fit_results[1].data['DRIFT_SCALE'][i-1]
                decay_time = oscan_fit_results[1].data['DECAY_TIME'][i-1]

                params0 = [-6.0, drift_scale, decay_time]
                constraints = [(-6.8, -5.3), (drift_scale, drift_scale), (decay_time, decay_time)]
                fitting_task = OverscanFitting(params0, constraints, BiasDriftSimpleModel, 
                                               start=start, stop=stop)
                fit_results = scipy.optimize.minimize(fitting_task.negative_loglikelihood, 
                                                      params0, 
                                                      args=(signals, oscandata, noise, ncols),
                                                      bounds=constraints, method='SLSQP')

                success = fit_results.success
                if success:
                    ctiexp, drift_scale, decay_time = fit_results.x
                    print(sensor_id, i, ctiexp)
#                    oscan_fit_results[1].data['CTI'][ = 10**ctiexp
                else:
                    num_failures += 1
                    print(sensor_id, i)
                    print(fit_results)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='File path to base directory of overscan results.')
    args = parser.parse_args()

    main(args.directory)

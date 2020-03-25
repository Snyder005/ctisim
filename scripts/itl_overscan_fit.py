import argparse
import errno
import os
import numpy as np
from os.path import join
import scipy
from astropy.io import fits

from ctisim import ITL_AMP_GEOM, E2V_AMP_GEOM
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

    ## Config variables
    start = 3
    stop = 15
    ccd_type = 'itl'
    max_signal = 140000.
    min_signal = 20000.
    read_noise = 7.0


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

                signals_all = hdulist[amp].data['FLATFIELD_SIGNAL']
                data_all = hdulist[amp].data['COLUMN_MEAN'][:, start:stop+1]
                indices = (signals_all < max_signal)*(signals_all>min_signal)
                signals = signals_all[indices]
                data = data_all[indices]

                params0 = [-7., 2.0, 2.5]
                constraints = [(-7., -7.), (0., 10.), (0.01, 4.0)]

                fitting_task = OverscanFitting(params0, constraints, BiasDriftSimpleModel,
                                               start=start, stop=stop)
                fit_results = scipy.optimize.minimize(fitting_task.negative_loglikelihood,
                                                      params0,
                                                      args=(signals, data, read_noise/np.sqrt(2000.), ncols),
                                                      bounds=constraints, method='SLSQP')

                if fit_results.success:
                    ctiexp, drift_scale, decay_time = fit_results.x
                    cti_results[amp] = 10**ctiexp
                    drift_scales[amp] = drift_scale/10000.
                    decay_times[amp] = decay_time
                else:
                    num_failures += 1
                    print(sensor_id, amp)
                    print(fit_results)

            outfile = os.path.join(ccd_output_dir, '{0}_parameter_results.fits'.format(sensor_id))
            parameter_results = OverscanParameterResults(sensor_id, cti_results, drift_scales, decay_times)
            parameter_results.write_fits(outfile, overwrite=True)

    print('There were {0} failures in the overscan fit'.format(num_failures))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='File path to base directory of overscan results.')
    args = parser.parse_args()

    main(args.directory)


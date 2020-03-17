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

    ## Config variables
    start = 1
    stop = 4
    ccd_type = 'itl'
    max_signal = 5000.
    read_noise = 7.0

    num_failures = 0
    for raft_name in ['R20']:
        
        raft_output_dir = join(directory, raft_name)
        
        for ccd_name in ['S02']:

            ## Get existing overscan analysis results
            ccd_output_dir = join(directory, raft_name, ccd_name)
            sensor_id = '{0}_{1}'.format(raft_name, ccd_name)
            try:
                hdulist = fits.open(join(ccd_output_dir, 
                                      '{0}_overscan_results.fits'.format(sensor_id)))
            except FileNotFoundError:
                continue
            try:
                parameter_results_file = os.path.join(ccd_output_dir, 
                                                      '{0}_parameter_results.fits'.format(sensor_id))
                parameter_results = OverscanParameterResults.from_fits(parameter_results_file)
            except FileNotFoundError:
                continue

            cti_results = {i : 0.0 for i in range(1, 17)}

            ## CCD geometry info
            if ccd_type == 'itl':
                ncols = ITL_AMP_GEOM.nx + ITL_AMP_GEOM.prescan_width
            if ccd_type == 'e2v':
                ncols = E2V_AMP_GEOM.nx + E2V_AMP_GEOM.prescan_width
    
            for amp in range(1, 17):

                signals_all = hdulist[amp].data['FLATFIELD_SIGNAL']
                data_all = hdulist[amp].data['COLUMN_MEAN'][:, start:stop+1]
                indices = (signals_all < max_signal)
                signals = signals_all[indices]
                data = data_all[indices]

                output_amplifier = parameter_results.single_output_amplifier(amp, 1.)
                drift_scale = output_amplifier.scale*10000.
                decay_time = output_amplifier.decay_time

                params0 = [-6., drift_scale, decay_time]
                constraints = [(-6.8, -5.5), (drift_scale, drift_scale), (decay_time, decay_time)]

                fitting_task = OverscanFitting(params0, constraints, BiasDriftSimpleModel,
                                               start=start, stop=stop)
                fit_results = scipy.optimize.minimize(fitting_task.negative_loglikelihood,
                                                      params0,
                                                      args=(signals, data, read_noise/np.sqrt(2000.), ncols),
                                                      bounds=constraints, method='SLSQP')

                if not fit_results.success:
                    num_failures += 1
                    print(sensor_id, amp)
                    print(fit_results)
                else:
                    ctiexp, drift_scale, decay_time = fit_results.x
                    if  ctiexp > -5.6:
                        print(sensor_id, amp)
                        print("Flagged for MCMC.")
                    else:
                        cti_results[amp] = 10**ctiexp
                        print(ctiexp, drift_scale, decay_time)

            parameter_results.cti_results = cti_results
            parameter_results.write_fits(parameter_results_file, overwrite=True)

    print('There were {0} failures in the overscan fit'.format(num_failures))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='File path to base directory of overscan results.')
    args = parser.parse_args()

    main(args.directory)

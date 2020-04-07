import argparse

from ctisim import ITL_AMP_GEOM
from ctisim.fitting import SimpleModel
from ctisim.utils import OverscanParameterResults

def main(sensor_id):

    start = 1
    stop = 3
    max_signal = 10000.
    error = 7.0/np.sqrt(2000.)
    
    ## Get electronics parameters
    param_results_file = join(directory, 
                              '{0}_parameter_results.fits'.format(sensor_id))
    param_results = OverscanParameterResults.from_fits(param_results_file)
    cti_results = {amp : 0.0 for amp in range(1, 17)}
    drift_scales = param_results.drift_scales
    decay_times = param_results.decay_times
    thresholds = param_results.thresholds

    hdulist = fits.open(join(ccd_output_dir, 
                             '{0}_overscan_results.fits'.format(sensor_id)))

    for amp in range(1, 17):

        ## Signals
        all_signals = hdulist[amp].data['FLATFIELD_SIGNAL']
        signals = all_signals[all_signals<max_signal]

        ## Data
        data = hdulist[amp].data['COLUMN_MEAN'][all_signals<max_signal, 
                                                start:stop+1]

        params = Parameters()
        params.add('ctiexp', value=-6, min=-7, max=-5, vary=True)
        params.add('trapsize', value=0.0, min=0., max=10., vary=True)
        params.add('scaling', value=0.08, min=0, max=1.0, vary=True)
        params.add('emissiontime', value=0.4, min=0.1, max=1.0, vary=True)
        params.add('driftscale', value=drift_scales[amp], min=0., max=0.002, vary=False)
        params.add('decaytime', value=decay_times[amp], min=0.1, max=4.0, vary=False)
        params.add('threshold', value=thresholds[amp], min=0.0, max=150000., vary=False)

        model = SimpleModel()

        minner = Minimizer(model.difference, params, 
                           fcn_args=(signals, data, error, ncols),
                           fcn_kws={'start' : start, 'stop' : stop})
        result = minner.minimize()
        
    

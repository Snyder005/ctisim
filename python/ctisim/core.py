import numpy as np
import copy
from astropy.io import fits

class SegmentModelParams:

    def __init__(self, amp):

        self.amp = amp
        self.cti = 0.0
        self.drift_size = 0.0
        self.drift_tau = np.nan
        self.drift_threshold = 0.0
        self.trap_size = 0.0
        self.trap_location = np.nan
        self.density_factor = 0.0
        self.trap_tau = np.nan
        
        self.walkers = 0
        self.steps = 0
        self.mcmc_results = None
        
    def add_mcmc_results(self, mcmc_results, burn_in=0):

        walkers, steps, _ = mcmc_results.shape

        self.walkers = walkers
        self.steps = steps
        self.cti = 10**np.median(mcmc_results[:, burn_in:, 0])
        self.density_factor = np.median(mcmc_results[:, burn_in:, 1])
        self.trap_tau = np.median(mcmc_results[:, burn_in:, 2])
        self.trap_size = np.median(mcmc_results[:, burn_in:, 3])

        self.mcmc_results = mcmc_results

    def update_params(self, **param_results):

        try:
            self.cti = param_results['cti']
        except KeyError:
            pass
    
        ## Add bias drift results
        if param_results.get('drift_size'):
            try:
                self.drift_tau = param_results['drift_tau']
                self.drift_threshold = param_results['drift_threshold']
            except KeyError:
                self.drift_tau = np.nan
                self.drift_threshold = 0.0
                raise KeyError("Must include drift_tau and drift_threshold values " + \
                               "when updating drift_size.")
            else:
                self.drift_size = param_results['drift_size']

        ## Upate trap results
        if param_results.get('trap_size'):
            try:
                self.trap_location = param_results['trap_location']
                self.density_factor = param_results['density_factor']
                self.trap_tau = param_results['trap_tau']
            except KeyError:
                self.trap_location = np.nan
                self.density_factor = 0.0
                self.trap_tau = np.nan
                raise KeyError("Must include density_factor and trap_tau values " + \
                               "when updating trap_size.")
            else:
                self.trap_size = param_results['trap_size']

    def create_table_hdu(self):

        hdr = fits.Header()
        hdr['AMP'] = self.amp
        hdr['WALKERS'] = self.walkers
        hdr['STEPS'] = self.steps

        results = self.mcmc_results.reshape((-1, 4)) # see what error raised and add try/except
        cols = [fits.Column(name='CTIEXP', array=results[:, 0], format='E'),
                fits.Column(name='TRAPSIZE', array=results[:, 3], format='E'),
                fits.Column(name='TAU', array=results[:, 2], format='E'),
                fits.Column(name='DFACTOR', array=results[:, 1], format='E')]

        mcmc_hdu = fits.BinTableHDU.from_columns(cols, header=hdr)

        return mcmc_hdu
        
class SensorModelParams:

    def __init__(self):

        self.segment_params = {i : SegmentModelParams(i) for i in range(1, 17)}

    def add_segment_mcmc_results(self, amp, mcmc_results, burn_in=0):

        self.segment_params[amp].add_mcmc_results(mcmc_results, burn_in=burn_in)

    def update_segment_params(self, amp, **param_results):

            self.segment_params[amp].update_params(**param_results)

    def write_fits(self, outfile, overwrite=True):
        
        prihdu = fits.PrimaryHDU()

        cti_results = np.zeros(16)
        drift_size_results = np.zeros(16)
        drift_tau_results = np.zeros(16)
        drift_threshold_results = np.zeros(16)
        trap_size_results = np.zeros(16)
        trap_tau_results = np.zeros(16)
        dfactor_results = np.zeros(16)
        location_results = np.zeros(16)

        for i in range(16):
            cti_results[i] = self.segment_params[i+1].cti
            drift_size_results[i] = self.segment_params[i+1].drift_size
            drift_tau_results[i] = self.segment_params[i+1].drift_tau
            drift_threshold_results[i] = self.segment_params[i+1].drift_threshold
            trap_size_results[i] = self.segment_params[i+1].trap_size
            trap_tau_results[i] = self.segment_params[i+1].trap_tau
            dfactor_results[i] = self.segment_params[i+1].density_factor
            location_results[i] = self.segment_params[i+1].trap_location
        
        cols = [fits.Column(name='CTI', array=cti_results, format='E'),
                fits.Column(name='DRIFT_SIZE', array=drift_size_results, format='E'),
                fits.Column(name='DRIFT_TAU', array=drift_tau_results, format='E'),
                fits.Column(name='DRIFT_THRESHOLD', array=drift_threshold_results, format='E'),
                fits.Column(name='TRAP_SIZE', array=trap_size_results, format='E'),
                fits.Column(name='TRAP_TAU', array=trap_tau_results, format='E'),
                fits.Column(name='TRAP_DFACTOR', array=dfactor_results, format='E'),
                fits.Column(name='TRAP_LOCATION', array=location_results, format='E')]
        print('test')
        hdu = fits.BinTableHDU.from_columns(cols)
        
        hdulist = fits.HDUList([prihdu, hdu])
        for i in range(1, 17):
            if self.segment_params[i].mcmc_results is not None:
                hdulist.append(self.segment_params[i].create_table_hdu())

        hdulist.writeto(outfile, overwrite=overwrite)

class SerialTrap:
    """Represents a serial register trap.

    Attributes:
        density_factor (float): Fraction of pixel signal exposed to trap.
        emission_time (float): Trap emission time constant [1/transfers].
        trap_size (float): Size of charge trap [e-].
        location (int): Serial pixel location of trap.
    """
    
    def __init__(self, size, scaling, emission_time, threshold, pixel):
        
        self.size = size
        self.scaling = scaling
        self.emission_time = emission_time
        self.threshold = threshold
        self.pixel = pixel

class OutputAmplifier:
    """Object representing the readout amplifier of a single channel.

    Attributes:
        noise (float): Value of read noise [e-].
        offset (float): Bias offset level [e-].
        gain (float): Value of amplifier gain [e-/ADU].
        do_bias_drift (bool): Specifies inclusion of bias drift.
        drift_size (float): Strength of bias drift exponential.
        drift_tau (float): Decay time constant for bias drift.
        drift_threshold (float): Cut-off threshold for bias drift.
    """
    
    def __init__(self, gain, noise, offset=0.0, drift_scale=0.0, 
                 decay_time=np.nan, threshold=0.0):

        self.gain = gain
        self.noise = noise
        self.offset = offset
        self.drift_scale = drift_scale
        self.decay_time = decay_time
        self.threshold = threshold

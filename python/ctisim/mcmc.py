import numpy as np

from ctisim import SerialTrap, OutputAmplifier, SegmentSimulator

class TrapModelFitting:
    """Object to control MCMC parameter fitting of a serial trapping model."""
    
    def __init__(self, constraints, amp_geom, num_oscan_pixels=5):
        
        self.constraints = constraints
        self.amp_geom = amp_geom
        self.num_oscan_pixels = num_oscan_pixels
        
    def logprior(self, trap_params):
        """Calculate log prior for given parameters and constraints."""
        
        ## Parameter constraint min/max limits
        ctiexp_l, ctiexp_h = self.constraints['ctiexp']
        df_l, df_h = self.constraints['densityfactor']
        tau_l, tau_h = self.constraints['tau']
        trapsize_l, trapsize_h = self.constraints['trapsize']
        
        ## Log prior calculation
        ctiexp, df, tau, trapsize = trap_params
        if (ctiexp>ctiexp_l)*(ctiexp<ctiexp_h)*(tau>tau_l)*(tau<tau_h)*(trapsize>trapsize_l)*(trapsize<trapsize_h)*(df>df_l)*(df<df_h):
            return 0.0
        else:
            return -np.inf
            
    def loglikelihood(self, trap_params, flux_array, data_rows, error, 
                      trap_locations, biasdrift_params=None):
        """Calculate log likelihood for model with given parameters."""
        
        ctiexp, scaling, emission_time, size = trap_params
        cti = 10.**ctiexp
        model_params = [cti, scaling, emission_time, size, 0.0]
        
        start = self.amp_geom.prescan_width + self.amp_geom.nx
        stop = start+self.num_oscan_pixels
        model_rows = self.simplified_model(model_params, flux_array, trap_locations, 
                                           self.amp_geom, biasdrift_params=biasdrift_params)
        
        model = model_rows[:, start:stop]
        data = data_rows[:, start:stop]
        inv_sigma2 = 1./(error**2.)
        
        return -0.5*(np.sum(inv_sigma2*(data-model)**2.))
        
    def logprobability(self, trap_params, flux_array, meanrows, error, 
                       trap_locations, biasdrift_params=None):
        """Calculate sum of log likelihood and log prior."""
        
        lp = self.logprior(trap_params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            result = lp + self.loglikelihood(trap_params, flux_array, meanrows, error, 
                                             trap_locations, biasdrift_params=biasdrift_params)
            return result
        
    @staticmethod
    def simplified_model(trap_params, flux_array, pixel, amp_geom, 
                         biasdrift_params=None):
        """Generate simulated data for given serial trap parameters."""
    
        nx = amp_geom.nx
        prescan_width = amp_geom.prescan_width
        serial_overscan_width = amp_geom.serial_overscan_width
        parallel_overscan_width = 0

        cti, scaling, emission_time, size, threshold = trap_params        
        trap = SerialTrap(size, scaling, emission_time, threshold, pixel)

        if biasdrift_params is not None:
            drift_scale, decay_time, drift_threshold = biasdrift_params
            output_amplifier = OutputAmplifier(1.0, 0.0, offset=0.0, drift_scale=drift_scale,
                                               decay_time=decay_time, threshold=drift_threshold)
        else:
            output_amplifier = OutputAmplifier(1.0, 0.0, offset=0.0)

        imarr = np.zeros((len(flux_array), nx))
        flat = SegmentSimulator(imarr, prescan_width, output_amplifier, cti=cti, traps=trap)
        flat.ramp_exp(flux_array)

        return flat.simulate_readout(serial_overscan_width=serial_overscan_width, 
                                      parallel_overscan_width=parallel_overscan_width,
                                      do_trapping=True, do_bias_drift=True)

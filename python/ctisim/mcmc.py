import numpy as np

from ctisim import SerialTrap, SerialRegister, ReadoutAmplifier, SegmentSimulator

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
            
    def loglikelihood(self, trap_params, flux_array, data_rows, error, trap_locations, biasdrift_params=None):
        """Calculate log likelihood for model with given parameters."""
        
        ctiexp, densityfactor, tau, trapsize = trap_params
        cti = 10.**ctiexp
        model_params = [cti, densityfactor, tau, trapsize]
        
        start = self.amp_geom['num_serial_prescan'] + self.amp_geom['ncols']
        stop = start+self.num_oscan_pixels
        model_rows = self.simplified_model(model_params, flux_array, trap_locations, self.amp_geom,
                                          biasdrift_params=biasdrift_params)
        
        model = model_rows[:, start:stop]
        data = data_rows[:, start:stop]
        inv_sigma2 = 1./(error**2.)
        
        return -0.5*(np.sum(inv_sigma2*(data-model)**2.))
        
    def logprobability(self, trap_params, flux_array, meanrows, error, trap_locations, biasdrift_params=None):
        """Calculate sum of log likelihood and log prior."""
        
        lp = self.logprior(trap_params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            result = lp + self.loglikelihood(trap_params, flux_array, meanrows, error, 
                                             trap_locations, biasdrift_params=biasdrift_params)
            return result
        
    @staticmethod
    def simplified_model(trap_params, flux_array, trap_location, amp_geom, biasdrift_params=None):
        """Generate simulated data for given serial trap parameters."""
    
        cti, df, tau, trap_size = trap_params

        ncols = amp_geom['ncols']
        num_serial_prescan = amp_geom['num_serial_prescan']
        num_serial_overscan = amp_geom['num_serial_overscan']
        num_parallel_overscan = 0

        trap = SerialTrap(df, tau, trap_size, location=trap_location)
        serial_register = SerialRegister(ncols+num_serial_prescan, cti, trap=trap)
        readout_amplifier = ReadoutAmplifier(0.0, gain=1.0, offset=0.0, biasdrift_params=biasdrift_params)
        flat = SegmentSimulator((len(flux_array), ncols), num_serial_prescan)
        flat.ramp_exp(flux_array)

        return readout_amplifier.serial_readout(flat, serial_register, 
                                                num_serial_overscan=num_serial_overscan, 
                                                num_parallel_overscan=num_parallel_overscan)

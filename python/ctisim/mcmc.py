import numpy as np

from ctisim import SerialTrap, SerialRegister, ReadoutAmplifier, SegmentSimulator

class ModelFitting:
    
    def __init__(self, constraints, amp_geom, num_points=4):
        
        self.constraints = constraints
        self.amp_geom = amp_geom
        self.num_points = 5
        
    def logprior(self, params):
        
        ctiexp_l, ctiexp_h = self.constraints['ctiexp']
        df_l, df_h = self.constraints['densityfactor']
        tau_l, tau_h = self.constraints['tau']
        trapsize_l, trapsize_h = self.constraints['trapsize']
        
        ctiexp, df, tau, trapsize = params
        
        if (ctiexp>ctiexp_l)*(ctiexp<ctiexp_h)*(tau>tau_l)*(tau<tau_h)*(trapsize>trapsize_l)*(trapsize<trapsize_h)*(df>df_l)*(df<df_h):
            return 0.0
        else:
            return -np.inf
            
    def loglikelihood(self, params, fluxes, meanrows, error, trap_locations, biasdrift_params=None):
        
        ctiexp, densityfactor, tau, trapsize = params
        cti = 10.**ctiexp
        model_params = [cti, densityfactor, tau, trapsize]
        
        start = self.amp_geom['num_serial_prescan'] + self.amp_geom['ncols']
        stop = start+self.num_points
        modelrows = self.simplified_model(model_params, fluxes, trap_locations, self.amp_geom,
                                          biasdrift_params=biasdrift_params)
        
        model = modelrows[:, start:stop]
        data = meanrows[:, start:stop]
        inv_sigma2 = 1./(float(error)**2)
        
        return -0.5*(np.sum(inv_sigma2*(data-model)**2.))
        
    def logprobability(self, params, fluxes, meanrows, error, trap_locations, biasdrift_params=None):
        
        lp = self.logprior(params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.loglikelihood(params, fluxes, meanrows, error, trap_locations, biasdrift_params=biasdrift_params)
        
    @staticmethod
    def simplified_model(params, flux_list, trap_locations, amp_geom, biasdrift_params=None):
    
        cti, df, tau, trap_size = params

        ncols = amp_geom['ncols']
        num_serial_prescan = amp_geom['num_serial_prescan']
        num_serial_overscan = amp_geom['num_serial_overscan']
        num_parallel_overscan = 0

        trap = SerialTrap(df, tau, trap_size, locations=trap_locations)
        serial_register = SerialRegister(ncols+num_serial_prescan, cti, trap=trap)
        readout_amplifier = ReadoutAmplifier(0.0, gain=1.0, offset=0.0, biasdrift_params=biasdrift_params)
        flat = SegmentSimulator((len(flux_list), ncols), num_serial_prescan)
        flat.ramp_exp(flux_list)

        return readout_amplifier.serial_readout(flat, num_serial_overscan, 
                                                num_parallel_overscan, serial_register)

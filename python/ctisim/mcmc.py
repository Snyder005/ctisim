import numpy as np
from ctisim import OutputAmplifier, SegmentSimulator, LinearTrap, LogisticTrap

class TrapModelFitting:

    def __init__(self, params0, constraints, amp_geom, trap_type=LinearTrap, 
                 num_oscan_pixels=5, biasdrift_params=None):

        self.params0 = params0
        self.constraints = constraints
        if self.logprior(params0) == -np.inf:
            raise ValueError("Initial parameters lie outside constrained bounds.")

        self.start = amp_geom.nx + amp_geom.prescan_width
        self.stop = self.start + num_oscan_pixels
        self.amp_geom = amp_geom
        self.trap_type = trap_type

        ## Create OutputAmplifier object
        if biasdrift_params is not None:
            drift_scale, decay_time, drift_threshold = biasdrift_params
            self.output_amplifier = OutputAmplifier(1.0, 0.0, offset=0.0, drift_scale=drift_scale,
                                               decay_time=decay_time, threshold=drift_threshold)
            self.do_bias_drift = True
        else:
            self.output_amplifier = OutputAmplifier(1.0, 0.0, offset=0.0)
            self.do_bias_drift = False

    def initialize_walkers(self, scale_list, walkers):
        """Initialize a group of MCMC walkers."""

        params_list = []
        for i, param0 in enumerate(self.params0):

            low, high = self.constraints[i]
            params = np.clip(np.random.normal(param0, scale=scale_list[i], size=walkers),
                             low, high)
            params_list.append(params)

        p0 = np.asarray(params_list).T

        return p0

    def logprior(self, params):
        """Calculate log prior for given parameters and constraints."""

        for i, param in enumerate(params):
            low, high = self.constraints[i]
            if not (param>low)*(param<high):
                return -np.inf
        
        return 0.0

    def loglikelihood(self, params, signals, data, error, trap_pixel, 
                      biasdrift_params=None):
        """Calculate log likelihood for model."""

        model = self.simplified_model(params, signals, trap_pixel)

        inv_sigma2 = 1./(error**2.)
        diff = (model-data)[self.start:self.stop]

        return -0.5*(np.sum(inv_sigma2*(diff)**2.))
        
    def logprobability(self, params, signals, data, error, trap_pixel,
                       biasdrift_params=None):
        """Calculate log probability for given parameters and model."""

        lp = self.logprior(params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            result = lp + self.loglikelihood(params, signals, data, error, trap_pixel, 
                                             biasdrift_params=biasdrift_params)
            return result

    def simplified_model(self, params, signals, trap_pixel):

        ## Create SerialTrap object
        cti = 10**params[0]
        size = params[1]
        emission_time = params[2]
        trap = self.trap_type(size, emission_time, trap_pixel, *params[3:])
            
        ## Ramp exposure across signal levels
        imarr = np.zeros((len(signals), self.amp_geom.nx))
        ramp = SegmentSimulator(imarr, self.amp_geom.prescan_width, self.output_amplifier, 
                                cti=cti, traps=trap)
        ramp.ramp_exp(signals)

        return ramp.simulate_readout(serial_overscan_width=self.amp_geom.serial_overscan_width,
                                     parallel_overscan_width=0,
                                     do_bias_drift=self.do_bias_drift)

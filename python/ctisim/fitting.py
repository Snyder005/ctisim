# -*- coding: utf-8 -*-
"""Simple deferred charge models.

This submodule contains function definitions to reproduce the overscan results
due to a variety of different simplified deferred charge models.

Todo:

"""
import numpy as np
from ctisim import SegmentSimulator

class BaseSimpleModel:
    """Base analytic model for EPER."""

    def __init__(self, ctiexp, num_transfers):
        self.cti = 10**ctiexp
        self.num_transfers = num_transfers

    def results(self, signals, start=1, stop=10):

        if start<1:
            raise ValueError("Start pixel must be 1 or greater.")
        if start >= stop:
            raise ValueError("Start pixel must be less than stop pixel.")

        x = np.arange(start, stop+1)
        model_results = np.zeros((signals.shape[0], x.shape[0]))
        for i, signal in enumerate(signals):

            model_results[i, :] = self.overscan_pixels(signal, x)

        return model_results

    def overscan_pixels(self, signal, x):
        raise NotImplementedError

class FixedLossSimpleModel(BaseSimpleModel):
    """Analytic EPER model with fixed loss and CTI."""

    def __init__(self, params, num_transfers):
        super().__init__(params[0], num_transfers)
        self.size = params[1]
        self.tau = params[2]

    def overscan_pixels(self, signal, x):

        A = self.size*(np.exp(1/self.tau) - 1)
        r = A*np.exp(-x/self.tau) + (self.cti**x)*self.num_transfers*signal

        return r

class BiasDriftSimpleModel(BaseSimpleModel):
    """Analytic EPER model with proportional loss and CTI."""

    def __init__(self, params, num_transfers):  
        super().__init__(params[0], num_transfers)
        self.scale = params[1]/10000.
        self.tau = params[2]

    def overscan_pixels(self, signal, x):

        r = (self.scale*signal)*np.exp(-x/self.tau) + (self.cti**x)*self.num_transfers*signal

        return r

class CTISimpleModel(BaseSimpleModel):
    """Analytic EPER model with only CTI."""

    def __init__(self, params, num_transfers):
        
        if not isinstance(params, list):
            params = [params]
        super().__init__(params[0], num_transfers)

    def overscan_pixels(self, signal, x):
        """Model proportional loss from CTI."""

        r = (self.cti**x)*self.num_transfers*signal

        return r

class BaseSimulatedModel:
    """Base model to handle simulating varying trap and CTI parameters."""

    def __init__(self, ctiexp, params, amp_geom, trap_type, output_amplifier, trap_pixel=1):

        self.cti = 10**ctiexp
        self.amp_geom = amp_geom
        self.output_amplifier = output_amplifier
        self.last_pix = amp_geom.prescan_width + amp_geom.nx
        self.trap = trap_type(params[0], params[1], trap_pixel, *params[2:])

    def results(self, signals, start=1, stop=10, **kwargs):

        if start<1:
            raise ValueError("Start pixel must be 1 or greater.")
        if start >= stop:
            raise ValueError("Start pixel must be less than stop pixel.")
        imarr = np.zeros((signals.shape[0], self.amp_geom.nx))
        
        ## Add additional fixed parameter traps
        try:
            traps = kwargs['traps']
            if isinstance(traps, list):
                traps.append(self.trap)
            else:
                traps = [traps, self.trap]
        except KeyError:
            traps = self.trap

        ## Simulate ramp readout
        ramp = SegmentSimulator(imarr, self.amp_geom.prescan_width, self.output_amplifier,
                                cti=self.cti, traps=traps)
        ramp.ramp_exp(signals)
        model_results = ramp.simulate_readout(serial_overscan_width=self.amp_geom.serial_overscan_width,
                                              parallel_overscan_width=0, **kwargs)

        return model_results[:, self.last_pix+start-1:self.last_pix+stop]

class JointSimulatedModel(BaseSimulatedModel):
    """Simulated model with varying trap and CTI parameters."""

    def __init__(self, params, amp_geom, trap_type, output_amplifier, trap_pixel=1):

        super().__init__(params[0], params[1:], amp_geom, trap_type, 
                         output_amplifier, trap_pixel=trap_pixel)

class TrapSimulatedModel(BaseSimulatedModel):
    """Simulated model with varying trap parameters and fixed CTI."""

    def __init__(self, params, ctiexp, amp_geom, trap_type, output_amplifier, trap_pixel=1):

        super().__init__(ctiexp, params, amp_geom, trap_type, 
                         output_amplifier, trap_pixel=trap_pixel)

class OverscanFitting:

    def __init__(self, params0, constraints, overscan_model, start=1, stop=10):

        self.params0 = params0
        self.constraints = constraints
        if self.logprior(params0) == -np.inf:
            raise ValueError("Initial parameters lie outside constrained bounds.")
        self.overscan_model = overscan_model

        if start<1:
            raise ValueError("Start pixel must be 1 or greater")
        if start >= stop:
            raise ValueError("Start pixel must be less than stop pixel.")
        self.start = start
        self.stop = stop

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
            if not (param>=low)*(param<=high):
                return -np.inf
        
        return 0.0

    def loglikelihood(self, params, signals, data, error, *args, **kwargs):
        """Calculate log likelihood for model."""

        model = self.overscan_model(params, *args)
        model_pixels = model.results(signals, start=self.start, stop=self.stop, **kwargs)

        inv_sigma2 = 1./(error**2.)
        diff = model_pixels-data

        return -0.5*(np.sum(inv_sigma2*(diff)**2.))
        
    def logprobability(self, params, signals, data, error, *args, **kwargs):
        """Calculate log probability for given parameters and model."""

        lp = self.logprior(params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            result = lp + self.loglikelihood(params, signals, data, error, *args, **kwargs)
            if np.isnan(result):
                print(params)
            return result

    def negative_loglikelihood(self, params, signals, data, error, *args, **kwargs):
        
        return -self.loglikelihood(params, signals, data, error, *args, **kwargs)

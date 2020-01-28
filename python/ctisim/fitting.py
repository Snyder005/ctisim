# -*- coding: utf-8 -*-
"""Simple deferred charge models.

This submodule contains function definitions to reproduce the overscan results
due to a variety of different simplified deferred charge models.

Todo:
    * Trap Model Fitting and Overscan Fitting look very similar.  Modify to make
      use of python inheritance.  Then the optimizer of choice (minimize or mcmc)
      can be used using with the specific class (logprobability).

"""
import numpy as np
from ctisim import OutputAmplifier, SegmentSimulator, LinearTrap, LogisticTrap

class BaseSimpleModel:

    def __init__(self, ctiexp, num_transfers):
        self.cti = 10**ctiexp
        self.num_transfers = num_transfers

    def results(self, signals, start=1, stop=10):

        if start<1:
            raise ValueError("Start pixel must be 1 or greater.")

        x = np.arange(start, stop+1)
        model_results = np.zeros((signals.shape[0], x.shape[0]))
        for i, signal in enumerate(signals):

            model_results[i, :] = self.overscan_pixels(signal, x)

        return model_results

    def overscan_pixels(self, signal, x):
        raise NotImplementedError

class FixedLossModel(BaseSimpleModel):

    def __init__(self, params, num_transfers):
        super().__init__(params[0], num_transfers)
        self.size = params[1]
        self.tau = params[2]

    def overscan_pixels(self, signal, x):

        A = self.size*(np.exp(1/self.tau) - 1)
        r = A*np.exp(-x/self.tau) + (self.cti**x)*self.num_transfers*signal

        return r

class BiasDriftModel(BaseSimpleModel):

    def __init__(self, params, num_transfers):  
        super().__init__(params[0], num_transfers)
        self.scale = params[1]/10000.
        self.tau = params[2]

    def overscan_pixels(self, signal, x):

        r = (self.scale*signal)*np.exp(-x/self.tau) + (self.cti**x)*self.num_transfers*signal

        return r

class CTIModel(BaseSimpleModel):

    def __init__(self, params, num_transfers):
        
        if not isinstance(params, list):
            params = [params]
        super().__init__(params[0], num_transfers)

    def overscan_pixels(self, signal, x):
        """Model proportional loss from CTI."""

        r = (self.cti**x)*self.num_transfers*signal

        return r

class SimulatedTrapModel:

    def __init__(self, params, amp_geom, trap_type, output_amplifier, trap_pixel=1):

        self.cti = params[0]        
        self.amp_geom = amp_geom
        self.output_amplifier = output_amplifier
        self.last_pix = amp_geom.prescan_width + amp_geom.nx
        self.trap = trap_type(params[1], params[2], trap_pixel, *params[3:])

    def results(self, signals, start=1, stop=10, **kwargs):

        if start<1:
            raise ValueError("Start pixel must be 1 or greater.")
        imarr = np.zeros((signals.shape[0], self.amp_geom.nx))
        ramp = SegmentSimulator(imarr, self.amp_geom.prescan_width, self.output_amplifier,
                                cti=self.cti, traps=self.trap)
        ramp.ramp_exp(signals)

        model_results = ramp.simulate_readout(serial_overscan_width=self.amp_geom.serial_overscan_width,
                                              parallel_overscan_width=0, **kwargs)

        return model_results[:, self.last_pix+start-1:self.last_pix+stop]

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

        model = self.overscan_model(params, *args, **kwargs)
        model_pixels = model.results(signals, start=self.start, stop=self.stop)

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
            return result

    def negative_loglikelihood(self, params, signals, data, error, *args, **kwargs):
        
        return -self.loglikelihood(params, signals, data, error, *args, **kwargs)

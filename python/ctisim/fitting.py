# -*- coding: utf-8 -*-
"""Simple deferred charge models.

This submodule contains function definitions to reproduce the overscan results
due to a variety of different simplified deferred charge models.

ToDo:

"""
import numpy as np
from ctisim import SegmentSimulator
from ctisim import LinearTrap, LogisticTrap
from ctisim import BaseOutputAmplifier, FloatingOutputAmplifier

class OverscanModel:
    """Base object handling model/data fit comparisons."""

    def loglikelihood(self, params, signals, data, error, 
                      *args, **kwargs):
        """Calculate log likelihood of the model."""

        model_results = self.model_results(params, signals, 
                                           *args, **kwargs)

        inv_sigma2 = 1./(error**2.)
        diff = model_results-data

        return -0.5*(np.sum(inv_sigma2*(diff)**2.))

    def negative_loglikelihood(self, params, signals, data, error, 
                               *args, **kwargs):
        """Calculate negative log likelihood of the model."""
        
        ll = self.loglikelihood(params, signals, data, error, *args, **kwargs)

        return -ll
        

    def rms_error(self, params, signals, data, error, *args, **kwargs):
        """Calculate RMS error between model and data."""

        model_results = self.model_results(params, signals, *args, **kwargs)

        inv_sigma2 = 1./(error**2.)
        diff = model_pixels-data

        rms = np.sqrt(np.mean(np.square(diff)))

        return rms

    def difference(self, params, signals, data, error, *args, **kwargs):
        """Calculate the flattened difference array between model and data."""

        model_results = self.model_results(params, signals, *args, **kwargs)

        inv_sigma2 = 1./(error**2.)
        diff = (model_results-data).flatten()

        return diff
    
class SimpleModel(OverscanModel):
    """Simple analytic overscan model."""
        
    @staticmethod
    def model_results(params, signals, num_transfers, start=1, stop=10):
        
        v = params.valuesdict()        
        try:
            v['cti'] = 10**v['ctiexp']
        except KeyError:
            pass
        
        x = np.arange(start, stop+1)
        res = np.zeros((signals.shape[0], x.shape[0]))

        for i, s in enumerate(signals):
            res[i, :] = (np.minimum(v['trapsize'], s*v['scaling'])*(np.exp(1/v['emissiontime'])-1.)*np.exp(-x/v['emissiontime'])
                         + s*num_transfers*v['cti']**x
                         + v['driftscale']*np.maximum(0, s-v['threshold'])*np.exp(-x/v['decaytime']))
                                            
        return res
    
class SimulatedModel(OverscanModel):
    """Simulated overscan model."""
        
    @staticmethod
    def model_results(params, signals, num_transfers, amp_geom, **kwargs):
        
        v = params.valuesdict() 
        
        start = kwargs.pop('start', 1)
        stop = kwargs.pop('stop', 10)
        trap_type = kwargs.pop('trap_type', None)
        
        ## Electronics effect optimization
        try:
            output_amplifier = FloatingOutputAmplifier(1.0, 
                                                       v['driftscale'], 
                                                       v['decaytime'],
                                                       v['threshold'])
        except KeyError:
            output_amplifier = BaseOutputAmplifier(1.0)
            
        ## CTI optimization
        try:
            v['cti'] = 10**v['ctiexp']
        except KeyError:
            pass
        
        ## Trap type for optimization
        if trap_type is None:
            trap = None
        elif trap_type == 'linear':
            trap = LinearTrap(v['trapsize'], v['emissiontime'], 1, 
                              v['scaling'])
        elif trap_type == 'logistic':
            trap = LogisticTrap(v['trapsize'], v['emissiontime'], 1, 
                                v['f0'], v['k'])
            
        ## Optional fixed traps
        try:
            fixed_traps = kwargs['fixed_traps']
            if isinstance(traps, list):
                traps.append(trap)
            else:
                traps = [fixed_traps, trap]
        except KeyError:
            traps = trap

        ## Simulate ramp readout
        imarr = np.zeros((signals.shape[0], amp_geom.nx))
        ramp = SegmentSimulator(imarr, amp_geom.prescan_width, output_amplifier,
                                cti=v['cti'], traps=trap)
        ramp.ramp_exp(signals)
        model_results = ramp.readout(serial_overscan_width=amp_geom.serial_overscan_width,
                                     parallel_overscan_width=0, **kwargs)
        
        ncols = amp_geom.prescan_width + amp_geom.nx

        return model_results[:, ncols+start-1:ncols+stop]

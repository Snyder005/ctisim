# -*- coding: utf-8 -*-
"""Core classes and objects.

This submodule contains the core classes for use in the deferred charge simulations.
"""

import numpy as np
import copy
from astropy.io import fits

class SerialTrap:
    """Represents a serial register trap.

    This is a base class which all serial trap class variations inherit from.
    All serial trap classes use the same emission function, but must re-implement
    their specific trapping functions.

    Attributes:
        density_factor (float): Fraction of pixel signal exposed to trap.
        emission_time (float): Trap emission time constant [1/transfers].
        trap_size (float): Size of charge trap [e-].
        location (int): Serial pixel location of trap.
    """
    
    def __init__(self, size, emission_time, pixel):
        
        self.size = size
        self.emission_time = emission_time
        self.pixel = pixel
        self._trap_array = None
        self._trapped_charge = None

    @property
    def trap_array(self):
        return self._trap_array

    @property
    def trapped_charge(self):
        return self._trapped_charge

    def initialize(self, ny, nx, prescan_width):
        """Initialize trapping arrays for simulated readout."""

        if self.pixel > nx+prescan_width:
            raise ValueError('Trap location {0} must be less than {1}'.format(self.pixel,
                                                                              nx+prescan_width))

        self._trap_array = np.zeros((ny, nx+prescan_width))
        self._trap_array[:, self.pixel] = self.size
        self._trapped_charge = np.zeros((ny, nx+prescan_width))

    def release_charge(self):
        """Release charge through exponential decay."""
        
        released_charge = self._trapped_charge*(1-np.exp(-1./self.emission_time))
        self._trapped_charge -= released_charge

        return released_charge

    def trap_charge(self):
        """Capture charge according to trapping function and parameters."""
        raise NotImplementedError

class LinearTrap(SerialTrap):

    def __init__(self, size, emission_time, pixel, scaling, threshold):

        super().__init__(size, emission_time, pixel)
        self.scaling = scaling
        self.threshold = threshold

    def trap_charge(self, free_charge):
        """Perform charge capture using a linear function."""
        
        captured_charge = np.clip((free_charge-self.threshold)*self.scaling,
                                  self.trapped_charge, self._trap_array) - self.trapped_charge
        self._trapped_charge += captured_charge

        return captured_charge

    @staticmethod
    def parameter_keywords():
        """Return keywords for trap model parameters."""

        return ['scaling', 'threshold']

class LogisticTrap(SerialTrap):

    def __init__(self, size, emission_time, pixel, f0, k):

        super().__init__(size, emission_time, pixel)
        self.f0 = f0
        self.k = k

    def trap_charge(self, free_charge):
        """Perform charge capture using a logistic function."""

        captured_charge = np.clip(self._trap_array/(1.+np.exp(-self.k*(free_charge-self.f0))),
                                  self.trapped_charge, None) - self.trapped_charge
        self._trapped_charge += captured_charge

        return captured_charge

    @staticmethod
    def parameter_keywords():
        """Return keywords for trap model parameters."""

        return ['f0', 'k']

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

    def offset_drift(self, drift, signal):
        """Calculate bias value hysteresis."""

        new_drift = np.maximum(self.drift_scale*(signal - self.threshold), 
                               np.zeros(signal.shape))
        
        return np.maximum(new_drift, drift*np.exp(-1/self.decay_time))
        

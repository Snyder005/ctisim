# -*- coding: utf-8 -*-
"""Core classes and objects.

This submodule contains the core classes for use in the deferred charge simulations.

To Do:
    * Change the initialize. Traps should not carry a pixel array dependent on
      unknown amplifier geometry.
    * Modify so that each trap has a trapping function that can be used for the
      trapping operator in the correction scheme.
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
        
        if size <= 0.0:
            raise ValueError('Trap size must be greater than 0.')
        self.size = size

        if emission_time <= 0.0:
            raise ValueError('Emission time must be greater than 0.')
        if np.isnan(emission_time):
            raise ValueError('Emission time must be real-valued number, not NaN')
        self.emission_time = emission_time

        if not isinstance(pixel, int):
            raise ValueError('Pixel must be type int.')
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

    def trap_charge(self, free_charge):
        """Perform charge capture using a logistic function."""

        captured_charge = np.clip(self.f(free_charge), self.trapped_charge, 
                                  self._trap_array) - self.trapped_charge
        self._trapped_charge += captured_charge

        return captured_charge

    def capture(self):
        """Trap capture function."""

        raise NotImplementedError

class LinearTrap(SerialTrap):

    parameter_keywords = ['scaling']
    model_type = 'linear'

    def __init__(self, size, emission_time, pixel, scaling):

        super().__init__(size, emission_time, pixel)
        self.scaling = scaling

    def f(self, pixel_signals):
        """Calculate charge trapping function."""

        return pixel_signals*self.scaling

class LogisticTrap(SerialTrap):

    parameter_keywords = ['f0', 'k']
    model_type = 'logistic'

    def __init__(self, size, emission_time, pixel, f0, k):

        super().__init__(size, emission_time, pixel)
        self.f0 = f0
        self.k = k

    def f(self, pixel_signals):
        
        return self.size/(1.+np.exp(-self.k*(pixel_signals-self.f0)))

class SplineTrap(SerialTrap):

    parameter_keywords = None
    model_type = 'spline'

    def __init__(self, interpolant, emission_time, pixel):

        super().__init__(self, 200000., emission_time, pixel)
        self.f = interpolant

class BaseOutputAmplifier:

    do_local_offset = False

    def __init__(self, gain, noise=0.0, global_offset=0.0):

        self.gain = gain
        self.noise = noise
        self.global_offset = global_offset

class FloatingOutputAmplifier(BaseOutputAmplifier):
    """Object representing the readout amplifier of a single channel.

    Attributes:
        noise (float): Value of read noise [e-].
        offset (float): Bias offset level [e-].
        gain (float): Value of amplifier gain [e-/ADU].
        do_bias_drift (bool): Specifies inclusion of bias drift.
        drift_size (float): Strength of bias drift exponential.
        drift_tau (float): Decay time constant for bias drift.
    """
    do_local_offset = True
    
    def __init__(self, gain, scale, decay_time, noise=0.0, offset=0.0):

        super().__init__(gain, noise, offset)
        self.update_parameters(scale, decay_time)

    def local_offset(self, old, signal):
        """Calculate local offset hysteresis."""

        new = np.maximum(self.scale*signal, np.zeros(signal.shape))
        
        return np.maximum(new, old*np.exp(-1/self.decay_time))

    def update_parameters(self, scale, decay_time):
        """Update parameter values, if within acceptable values."""

        if scale <= 0.0:
            raise ValueError("Hysteresis scale must be greater than or equal to 0.")
        self.scale = scale
        if decay_time <= 0.0:
            raise ValueError("Decay time must be greater than 0.")
        if np.isnan(decay_time):
            raise ValueError("Decay time must be real-valued number, not NaN.")
        self.decay_time = decay_time

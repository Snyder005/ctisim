# ctisim

 `ctisim` is a module for performing a simplified simulation of serial readout of CCD sensors (specifically LSST CCD sensors) and model the resulting "deferred charge" effects.  The following effects are modeled:
* Proportional charge loss due to transfer inefficiency that occurs at each serial pixel transfer.
* Fixed charge loss due to charge trapping at specific regions in the serial register.
* Exponential bias hysteresis in the output amplifier that mimics a deferred charge signal.

Additionally `ctisim` includes code to generate simulated CCD diagnostic segment and full-frame images, such as:
* Flat field, or uniform illumination images.
* Ramp images, or images with increasing illumination across rows.
* <sup>55</sup>Fe soft x-ray exposures.
* Simulated star images (not fully implemented).

# Dependencies

The `ctisim` is written for Python 3.7.3 and is designed to be used with the [LSST DM Stack Version 18](https://pipelines.lsst.io/).  Installation and usage of the stack will include most of the necessary core dependencies.  For usage outside of the stack the following non-standard library are required.  

* NumPy 1.17.0 - Array operations.
* Astropy 3.1.2 - Standard FITs Image/Table file interface.
* Galsim 2.1.5 - For simulation of source images with appropriate sensor level effects.

The following additional dependencies are used:

* emcee 3.0.0 - Markov Chain Monte Carlo ensemble sampler for model fitting.



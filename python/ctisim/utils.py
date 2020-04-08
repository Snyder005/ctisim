"""Utility objects and functions.

This submodule contains a convenience and utility objects, classes and functions
for use with the `ctisim` module.  


Attributes:
    ITL_AMP_GEOM (lsst.eotest.sensor.AmplifierGeometry): Segment geometry parameters
        for LSST ITL CCD sensors. 
    E2V_AMP_GEOM (lsst.eotest.sensor.AmplifierGeometry): Segment geometry parameters
        for LSST E2V CCD sensors.

To Do:
    * Modify OverscanParameterResults to more closely mimic EOTestResults from eotest.
    * Confirm global change of drift_size to drift_scale.
"""

import numpy as np
from astropy.io import fits
from lsst.eotest.sensor.AmplifierGeometry import AmplifierGeometry, amp_loc

from ctisim import FloatingOutputAmplifier

ITL_AMP_GEOM = AmplifierGeometry(prescan=3, nx=509, ny=2000, 
                                 detxsize=4608, detysize=4096,
                                 amp_loc=amp_loc['ITL'], vendor='ITL')
"""AmplifierGeometry: Amplifier geometry parameters for LSST ITL CCD sensors."""

E2V_AMP_GEOM = AmplifierGeometry(prescan=10, nx=512, ny=2002,
                                 detxsize=4688, detysize=4100,
                                 amp_loc=amp_loc['E2V'], vendor='E2V')
"""AmplifierGeometry: Amplifier geometry parameters for LSST E2V CCD sensors."""

class OverscanParameterResults:

    def __init__(self, sensor_id, cti_results, drift_scales, decay_times,
                 thresholds):

        self.sensor_id = sensor_id
        self.cti_results = cti_results
        self.drift_scales = drift_scales
        self.decay_times = decay_times
        self.thresholds = thresholds

    @classmethod
    def from_fits(cls, infile):

        with fits.open(infile) as hdulist:
            
            sensor_id = hdulist[0].header['SENSORID']
            data = hdulist[1].data

            cti_results = cls.asdict(data['CTI'])
            drift_scales = cls.asdict(data['DRIFT_SCALE'])
            decay_times = cls.asdict(data['DECAY_TIME'])
            thresholds = cls.asdict(data['THRESHOLD'])
        results = cls(sensor_id, cti_results, drift_scales, decay_times, 
                      thresholds)


        return results

    def single_output_amplifier(self, ampnum, gain, noise=0.0, offset=0.0):
        """Return a single output amplifier, given stored fit parameters."""
        
        drift_scale = self.drift_scales[ampnum]
        decay_time = self.decay_times[ampnum]
        output_amplifier = FloatingOutputAmplifier(gain, drift_scale, 
                                                   decay_time,
                                                   threshold,
                                                   noise=noise, 
                                                   offset=offset)

        return output_amplifier

    def all_output_amplifiers(self, gain_dict, noise_dict, offset_dict):
        """Return all output amplifier, given stored fit parameters."""


        output_amplifiers = {}
        for ampnum in range(1, 17):
            output_amplifiers[ampno] = self.single_output_amplifier(ampnum, 
                                                                    gain_dict[ampnum], 
                                                                    noise_dict[ampnum], 
                                                                    offset_dict[ampnum])

        return output_amplifiers
        
    def write_fits(self, outfile, **kwargs):

        hdr = fits.Header()
        hdr['SENSORID'] = self.sensor_id
        prihdu = fits.PrimaryHDU(header=hdr)
        
        cti_results = self.cti_results
        drift_scales = self.drift_scales
        decay_times = self.decay_times
        thresholds = self.thresholds

        cols = [fits.Column(name='AMPLIFIER', array=np.arange(1, 17), format='I'),
                fits.Column(name='CTI', array=self.asarray(cti_results), format='E'),
                fits.Column(name='DRIFT_SCALE', array=self.asarray(drift_scales), format='E'),
                fits.Column(name='DECAY_TIME', array=self.asarray(decay_times), format='E'),
                fits.Column(name='THRESHOLD', array=self.asarray(thresholds), format='E')]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdulist = fits.HDUList([prihdu, hdu])
        hdulist.writeto(outfile, **kwargs)
        
    @staticmethod
    def asarray(param_dict):
        
        param_array = np.asarray([param_dict[ampnum] for ampnum in range(1, 17)])

        return param_array

    @staticmethod
    def asdict(param_array):

        param_dict = {ampnum : param_array[ampnum-1] for ampnum in range(1, 17)}

        return param_dict

def calculate_cti(imarr, last_pix_num, num_overscan_pixels=1):
    """Calculate the serial CTI of an image array.

    Convenience function to calculate the serial charge transfer inefficiency 
    using the extended pixel edge response method.  The number of overscan pixels 
    to use in the calculation can be optionally specified.

    Args:
        imarr (numpy.ndarray): Image pixel data.
        last_pix_num (int): Last image pixel index.
        num_overscan_pixels (int): Overscan pixels to use to calcualte CTI.

    Returns:
        float: CTI result.
    """
    
    last_pix = np.mean(imarr[:, last_pix_num])

    overscan = np.mean(imarr[:, last_pix_num+1:], axis=0)
    cti = np.sum(overscan[:num_overscan_pixels])/(last_pix*last_pix_num)
                           
    return cti

def save_mcmc_results(sensor_id, amp, chain, outfile, trap_type):
    """Save the MCMC model fitting results to a FITs file.

    Convenience function to save the chain results of the `emcee` module's 
    implementation of Markov Chain Monte Carlo, as a FITs file.  The chain 
    results for each parameter are saves as individual FITs file Image HDUs.

    Args:
        sensor_id (str): Identifier for the CCD sensor.
        amp (int): Output amplifier number.
        chain (numpy.ndarray): MCMC results.
        outfile (str): Output filename.
        model (str): Type of trap model used.
    """

    walkers, steps, ndim = chain.shape

    hdr = fits.Header()
    hdr['SENSORID'] = sensor_id
    hdr['AMP'] = amp
    hdr['TYPE'] = trap_type.model_type
    hdr['STEPS'] = steps
    hdr['WALKERS'] = walkers

    prihdu = fits.PrimaryHDU(header=hdr)

    ctiexp_hdu = fits.ImageHDU(data=chain[:, :, 0], name='CTIEXP')
    trapsize_hdu = fits.ImageHDU(data=chain[:, :, 1], name='TRAPSIZE')
    emission_time_hdu = fits.ImageHDU(data=chain[:, :, 2], name='TAU')
    
    hdulist = fits.HDUList([prihdu, ctiexp_hdu, trapsize_hdu, emission_time_hdu])

    for i in range(3, ndim):
        name = trap_type.parameter_keywords[i-3]
        param_hdu = fits.ImageHDU(data=chain[:, :, i], name=name)
        hdulist.append(param_hdu)

    hdulist.writeto(outfile, overwrite=True)

    

    

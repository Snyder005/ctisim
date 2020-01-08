"""Utility objects and functions.

This submodule contains a convenience and utility objects, classes and functions
for use with the `ctisim` module.  


Attributes:
    ITL_AMP_GEOM (lsst.eotest.sensor.AmplifierGeometry): Segment geometry parameters
        for LSST ITL CCD sensors. 
    E2V_AMP_GEOM (lsst.eotest.sensor.AmplifierGeometry): Segment geometry parameters
        for LSST E2V CCD sensors.
"""

import numpy as np
from astropy.io import fits
from lsst.eotest.sensor.AmplifierGeometry import AmplifierGeometry, amp_loc

ITL_AMP_GEOM = AmplifierGeometry(prescan=3, nx=509, ny=2000, 
                                 detxsize=4608, detysize=4096,
                                 amp_loc=amp_loc['ITL'], vendor='ITL')
"""AmplifierGeometry: Amplifier geometry parameters for LSST ITL CCD sensors."""

E2V_AMP_GEOM = AmplifierGeometry(prescan=10, nx=512, ny=2002,
                                 detxsize=4688, detysize=4100,
                                 amp_loc=amp_loc['E2V'], vendor='E2V')
"""AmplifierGeometry: Amplifier geometry parameters for LSST E2V CCD sensors."""

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

def save_mcmc_results(sensor_id, amp, chain, outfile, model='linear'):
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

    steps, walkers, ndim = chain.shape

    hdr = fits.Header()
    hdr['SENSORID'] = sensor_id
    hdr['AMP'] = amp
    hdr['TYPE'] = model
    hdr['STEPS'] = steps
    hdr['WALKERS'] = walkers

    prihdu = fits.PrimaryHDU(header=hdr)

    ctiexp_hdu = fits.ImageHDU(data=chain[:, :, 0], name='CTIEXP')
    trapsize_hdu = fits.ImageHDU(data=chain[:, :, 1], name='TRAPSIZE')
    emission_time_hdu = fits.ImageHDU(data=chain[:, :, 2], name='TAU')
    hdulist = fits.HDUList([prihdu, ctiexp_hdu, trapsize_hdu, emission_time_hdu])

    for i in range(3, ndim):
        chain = sampler.chain[:, :, i]
        name = trap_type.parameter_keywords()[i-3]
        param_hdu = fits.ImageHDU(data=chain, name=name)
        hdulist.append(param_hdu)

    hdulist.writeto(outfile, overwrite=True)

def save_ccd_results(sensor_id, cti_results, drift_size_results, drift_tau_results,
                     drift_threshold_results, outfile):
    """Save optimization results for all CCD amplifiers.

    Convenience function to save the optimization results for CTI and output
    amplifier bias drift parameters, as a FITs file.

    Args:
        sensor_id (str): Identifier for the CCD sensor.
        cti_results (numpy.ndarray): CTI optimization results per amplifier.
        drift_size_results (numpy.ndarray): Drift size results per amplifier.
        drift_tau_results (numpy.ndarray): Drift decay time results per amplifier.
        drift_threshold_results (numpy.ndarray): Drift threshold results per amplifier.
        outfile (str): Output filename.
    """

    hdr = fits.Header()
    hdr['SENSORID'] = sensor_id
    prihdu = fits.PrimaryHDU(header=hdr)

    cols = [fits.Column(name='AMPLIFIER', array=np.arange(1, 17), format='I'),
            fits.Column(name='CTI', array=cti_results, format='E'),
            fits.Column(name='DRIFT_SIZE', array=drift_size_results, format='E'),
            fits.Column(name='DRIFT_TAU', array=drift_tau_results, format='E'),
            fits.Column(name='DRIFT_THRESHOLD', array=drift_threshold_results, format='E')]

    hdu = fits.BinTableHDU.from_columns(cols)
    hdulist = fits.HDUList([prihdu, hdu])
    hdulist.writeto(outfile, overwrite=True)

    

    

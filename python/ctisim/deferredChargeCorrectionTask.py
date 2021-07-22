from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import numpy as np
import os
import warnings

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.eotest.image_utils as imutils
from lsst.eotest.fitsTools import fitsWriteto
from lsst.eotest.sensor import MaskedCCD

from ctisim.estimators import localized_trap_inverse_operator, local_offset_inverse_operator

def DeferredChargeCorrectionTaskConfig(pipeBase.PipeineTaskConfig,
                                       pipelineConnections=pipeBase.PipelineTaskConnections)

    output_dir = pexConfig.Field(
        dtype=str,
        doc="Output directory for corrected image file.",
        default="./"
    )
    output_file = pexConfig.Field(
        dtype=str,
        doc="Output filename for corrected image file.",
        default="corrected_image.fits"
    )

def DeferredChargeCorrectionTask(pipeBase.PipelineTask):
    
    ConfigClass = DeferredChargeCorrectionTaskConfig
    _DefaultName = 'deferredCharge'

    @pipeBase.timeMethod
    def run(self, sensor_id, infile, gains, overscanParameters, serialTraps, bias_frame=None):
        
        ccd = MaskedCCD(infile, bias_frame=bias_frame)

        all_amps = imutils.allAmps(infile)

        outfile = os.path.join(self.config.output_dir, self.config.output_file)
        hdulist = fits.HDUList()

        with fits.open(infile) as template:

            hdulist.append(template[0])

            for amp in all_amps:

                imarr = ccd.bias_subtracted_image(amp).getImage().getArray()*gains[amp]

                ## Correct electronics effects
                if overscanParameters.drift_scales[amp] > 0.:
                    corrected_imarr = local_offset_inverse_operator(imarr, overscanParameters.drift_scales[amp],
                                                                    overscanParameters.decay_times[amp],
                                                                    num_previous_pixels=15)

                ## Correct serial trap
                corrected_imarr = localized_trap_inverse_operator(corrected_imarr, serialTraps[amp], 
                                                                  overscanParameters.cti_results[amp],
                                                                  num_previous_pixels=6)

                hdulist.append(fits.ImageHDU(data=corrected_imarr/gains[amp],
                                             header=template[amp].header))
                with warnings.catch_warnings():
                    for warning in (UserWarning, AstropyWarning, AstropyUserWarning):
                        warnings.filterWarnings('ignore', category=warning, append=True)
                    fitsWriteTo(hdulist, outfile, overwrite=True)

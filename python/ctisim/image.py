# -*- coding: utf-8 -*-
"""Image simulation classes.

This submodule contains tools for simulating LSST segment and full CCD images.
The LSST DM stack and the  `galsim` module is used to simulate sensor PSF
and create realistic images.  Simulated image objects use deferred charge
simulating tools within the full module to simulate the effects of serial readout.

To Do:
    * Modify ramp function to use different kwargs such as low/high or list of signals.
    * Clean up update_parameter functions in here and in utils.
"""

import os
import galsim
import warnings
import copy
import numpy as np
import multiprocessing
from astropy.io import fits

from ctisim.core import FloatingOutputAmplifier, SerialTrap
from lsst.eotest.sensor.MaskedCCD import MaskedCCD

class ImageSimulator:

    def __init__(self, ny, nx, prescan_width, serial_overscan_width, 
                 parallel_overscan_width, segments):

        ## Verify and assign segments and geometry
        for i in range(1, 17):
            segment = segments[i]
            assert segment.prescan_width == prescan_width
            assert segment.ny == ny
            assert segment.nx == nx
        self.segments = segments
        self.ny = ny
        self.nx = nx
        self.prescan_width = prescan_width
        self.serial_overscan_width = serial_overscan_width
        self.parallel_overscan_width = parallel_overscan_width

    @classmethod
    def from_image_fits(cls, infile, output_amplifiers, bias_frame=None,
                        linearity_correction=None, cti=None, traps=None):
        """Initialize from existing FITs file."""

        ## Geometry information from infile
        ccd = MaskedCCD(infile, bias_frame=bias_frame, 
                        linearity_correction=linearity_correction)
        prescan_width = ccd.amp_geom.prescan_width
        ny = ccd.amp_geom.ny
        nx = ccd.amp_geom.nx
        serial_overscan_width = ccd.amp_geom.serial_overscan_width
        parallel_overscan_width = ccd.amp_geom.naxis2 - ccd.amp_geom.ny

        if cti is None:
            cti = {i : 0.0 for i in range(1, 17)}
        if traps is None:
            traps = {i : None for i in range(1, 17)}

        segments = {}
        for i in range(1, 17):

            output_amplifier = output_amplifiers[i]
            gain = output_amplifier.gain

            imarr = ccd.unbiased_and_trimmed_image(i).getImage().getArray()*gain
            segments[i] = SegmentSimulator(imarr, prescan_width, output_amplifier, 
                                           cti=cti[i], traps=traps[i])

        image = cls(ny, nx, prescan_width, serial_overscan_width, 
                    parallel_overscan_width, segments)

        return image

    @classmethod
    def from_amp_geom(cls, amp_geom, output_amplifiers, cti=None,
                      traps=None):

        ny = amp_geom.ny
        nx = amp_geom.nx
        prescan_width = amp_geom.prescan_width
        serial_overscan_width = int(amp_geom.serial_overscan_width)
        parallel_overscan_width = int(amp_geom.naxis2 - amp_geom.ny)

        if cti is None:
            cti = {i : 0.0 for i in range(1, 17)}

        if traps is None:
            traps = {i : None for i in range(1, 17)}

        segments = {}
        for i in range(1, 17):
            output_amplifier = output_amplifiers[i]

            segments[i] =  SegmentSimulator.from_amp_geom(amp_geom, output_amplifier, 
                                                          cti=cti[i], traps=traps[i])

        image = cls(ny, nx, prescan_width, serial_overscan_width, 
                    parallel_overscan_width, segments)

        return image

    def fe55_exp(self, num_fe55_hits, stamp_length=6, psf_fwhm=0.00016, 
                 hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 exposure.

        This method simulates a Fe55 soft x-ray CCD image using the Galsim module.  
        Fe55 x-ray hits are randomly generated as postage stamps and positioned 
        randomly on each of the segment images.

        Args:
            num_fe55_hits (int): Number of Fe55 x-ray hits to perform.
            stamp_length (int): Side length of desired Fe55 postage stamp.
            random_seed (float): Random number generator seed.
            psf_fwhm (float): FWHM of sensor PSF.
            hit_flux (int): Total flux per Fe55 x-ray hit.
            hit_hlr (float): Half-light radius of Fe55 x-ray hits.
        """
        for i in range(1, 17):
            self.segments[i].fe55_exp(num_fe55_hits, stamp_length=stamp_length, 
                                      random_seed=None, psf_fwhm=psf_fwhm, 
                                      hit_flux=hit_flux, hit_hlr=hit_hlr)

    def flatfield_exp(self, signal, noise=True):
        """Simulate a flat field exposure.

        This method simulates a flat field CCD image with given signal level.
        The simulated image can be generated with or with out shot noise.

        Args:
            signal (float): Signal level of the flat field.
            noise (bool): Specifies inclusion of shot noise.
        """
        for i in range(1, 17):            
            self.segments[i].flatfield_exp(signal, noise=noise)

    def segment_readout(self, segarr_dict, amp, do_bias_drift=True):
        """Simulate readout of a single segment.

        This method is to facilitate the use of multiprocessing when reading out 
        an entire image (16 segments). 

        Args:
            segarr_dict ('dict' of 'numpy.array'): Dictionary of array results.
            amp (int): Amplifier number.
        """
        im = self.segments[amp].simulate_readout(serial_overscan_width=self.serial_overscan_width,
                                                 parallel_overscan_width=self.parallel_overscan_width,
                                                 do_bias_drift=do_bias_drift)
        segarr_dict[amp] = im
            
    def simulate_readout(self, template_file, bitpix=32, outfile='simulated_image.fits', 
                         use_multiprocessing=False, **kwargs):
        """Perform the serial readout of all CCD segments.

        This method simulates the serial readout for each segment of the CCD,
        in accordance to each segments ReadoutAmplifier and SerialRegister objects.
        Using a provided template file, an output file is generated that matches
        existing FITs image files.

        Args:
            template_file (str): Filepath to existing FITs file to use as template.
            bitpix (int): Representation of output array data type.
            outfile (str): Filepath for desired output data file.
            do_multiprocessing (bool): Specifies usage of multiprocessing module.
            kwds ('dict'): Keyword arguments for Astropy `HDUList.writeto()`.

        Returns:
            List of NumPy arrays.
        """
        output = fits.HDUList()
        output.append(fits.PrimaryHDU())

        ## Segment readout using single or multiprocessing
        if use_multiprocessing:
            manager = mp.Manager()
            segarr_dict = manager.dict()
            job = [mp.Process(target=self.segment_readout, 
                              args=(segarr_dict, amp, do_bias_drift),
                              kwargs=kwargs) for amp in range(1, 17)]

            _ = [p.start() for p in job]
            _ = [p.join() for p in job]

        else:
            segarr_dict = {}
            for amp in range(1, 17):
                self.segment_readout(segarr_dict, amp, do_bias_drift, **kwargs)

        ## Write results to FITs file
        with fits.open(template_file) as template:
            output[0].header.update(template[0].header)
            output[0].header['FILENAME'] = os.path.basename(outfile)
            for amp in range(1, 17):
                imhdu = fits.ImageHDU(data=segarr_dict[amp], header=template[amp].header)
                self.set_bitpix(imhdu, bitpix)
                output.append(imhdu)
            for i in (-3, -2, -1):
                output.append(template[i])
            output.writeto(outfile, overwrite=True)
            
        return segarr_dict

    def update_all_parameters(self, parameter_results):
        """Update CTI and output amplifier parameters for all segments."""

        for ampnum in range(1, 17):
            update_segment_parameters(ampnum, parameter_results)

    def update_segment_parameters(self, ampnum, parameter_results):
        """Update CTI and output amplifier parameters for a single segment."""

        cti = parameter_results.cti_results[ampnum]
        drift_size = parameter_results.drift_sizes[ampnum]
        decay_time = parameter_results.decay_times[ampnum]
        threshold = parameter_results.thresholds[ampnum]

        self.segments[i].cti = cti
        self.segments[i].output_amplifier.update_parameters(drift_scale, decay_time, threshold)
    
    @staticmethod
    def set_bitpix(hdu, bitpix):
        """Set desired data type (bitpix) for HDU image array.

        Args:
            hdu (fits.ImageHDU): ImageHDU to modify.
            bitpix (int): Representation of data type.
        """
        dtypes = {16: np.int16, -32: np.float32, 32: np.int32}
        for keyword in 'BSCALE BZERO'.split():
            if keyword in hdu.header:
                del hdu.header[keyword]
        if bitpix > 0:
            my_round = np.round
        else:
            def my_round(x): return x
        hdu.data = np.array(my_round(hdu.data), dtype=dtypes[bitpix])

class SegmentSimulator:
    """Controls the creation of simulated segment images.

    Attributes:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        num_serial_prescan (int): Number of serial prescan pixels.
        image (numpy.array): NumPy array containg the image pixels.
    """

    def __init__(self, imarr, prescan_width, output_amplifier, cti=0.0, traps=None):

        ## Image array geometry
        self.prescan_width = prescan_width
        self.ny, self.nx = imarr.shape

        self.segarr = np.zeros((self.ny, self.nx+prescan_width))
        self.segarr[:, prescan_width:] = imarr

        ## Serial readout information
        self.output_amplifier = output_amplifier
        if isinstance(cti, np.ndarray):
            raise ValueError("cti must be single value, not an array.")
        self.cti = cti
        
        self.serial_traps = None
        self.do_trapping = False
        if traps is not None:
            if not isinstance(traps, list):
                traps = [traps]
            for trap in traps:
                self.add_trap(trap)

    @classmethod
    def from_amp_geom(cls, amp_geom, output_amplifier, cti=0.0, traps=None):
        """Create SegmentSimulator object from AmplifierGeometry object.

        This method takes an existing AmplifierGeometry object and uses this to
        create a SegmentSimulator object with the desired image shape.  

        Args:
            amp_geom (AmplifierGeometry): Amplifier geometry information.
            output_amplifier (OutputAmplifier): Output amplifier for the segment.
            cti (float): CTI value for the segment.
            traps (list of SerialTrap): Traps to include in the serial register.
        """

        prescan_width = amp_geom.prescan_width
        imarr = np.zeros((amp_geom.ny, amp_geom.nx))

        segment = cls(imarr, prescan_width, output_amplifier, cti=cti, traps=traps)

        return segment

    def add_trap(self, serial_trap):
        """Add a trap to the serial register.

        Args:
            serial_trap (SerialTrap): Serial trap to include in serial register.
        """

        try:
            self.serial_traps.append(serial_trap)
        except AttributeError:
            self.serial_traps = [serial_trap]
            self.do_trapping = True

    def fe55_exp(self, num_fe55_hits, stamp_length=6, random_seed=None, psf_fwhm=0.00016, 
                 hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 exposure.

        This method simulates a Fe55 soft x-ray segment image using the Galsim module.  
        Fe55 x-ray hits are randomly generated as postage stamps and positioned 
        randomly on the segment image.

        Args:
            num_fe55_hits (int): Number of Fe55 x-ray hits to simulate.
            stamp_length (int): Side length of desired Fe55 hit postage stamp.
            random_seed (float): Random number generator seed.
            psf_fwhm (float): FWHM of sensor PSF.
            hit_flux (int): Total flux per Fe55 x-ray hit.
            hit_hlr (float): Half-light radius of Fe55 x-ray hits.
        """
        for i in range(num_fe55_hits):
            
            stamp = self.sim_fe55_hit(random_seed=random_seed, stamp_length=stamp_length,
                                      psf_fwhm=psf_fwhm, hit_flux=hit_flux, hit_hlr=hit_hlr).array
            sy, sx = stamp.shape

            y0 = np.random.randint(0, self.ny-sy)
            x0 = np.random.randint(self.prescan_width,
                                   self.nx+self.prescan_width-sx)

            self.segarr[y0:y0+sy, x0:x0+sx] += stamp

    def flatfield_exp(self, signal, noise=True):
        """Simulate a flat field exposure.

        This method simulates a flat field segment image with given signal level.
        The simulated image can be generated with or with out shot noise.

        Args:
            signal (float): Signal level of the flat field.
            noise (bool): Specifies inclusion of shot noise.
        """
        if noise:
            flat = np.random.poisson(signal, size=(self.ny, self.nx))
        else:
            flat = np.ones((self.ny, self.nx))*signal
        self.segarr[:, self.prescan_width:] += flat

    def ramp_exp(self, signal_list):
        """Simulate an image with varying flux illumination per row.

        This method simulates a segment image where the signal level increases
        along the horizontal direction, according to the provided list of
        signal levels.

        Args:
            signal_list ('list' of 'float'): List of signal levels.

        Raises:
            ValueError: If number of signal levels does not equal the number of rows.
        """
        if len(signal_list) != self.ny:
            raise ValueError
            
        ramp = np.tile(signal_list, (self.nx, 1)).T
        self.segarr[:, self.prescan_width:] += ramp

    def reset(self):
        """Reset segment image to zeros."""

        self.array[:, self.prescan_width:] = 0.0

    def simulate_readout(self, serial_overscan_width=10, parallel_overscan_width=0,
                         **kwargs):
        """Simulate serial readout of the segment image.

        This method performs the serial readout of a segment image given the
        appropriate SerialRegister object and the properties of the ReadoutAmplifier.
        Additional arguments can be provided to account for the number of 
        desired overscan transfers  The result is a simulated final segment image,
        in ADU.

        Args:
            segment (SegmentSimulator): Simulated segment image to process.
            serial_register (SerialRegister): Serial register to use during readout.
            num_serial_overscan (int): Number of serial overscan pixels.
            num_parallel_overscan (int): Number of parallel overscan pixels.

        Returns:
            NumPy array.
        """
        ## Create output array
        iy = int(self.ny + parallel_overscan_width)
        ix = int(self.nx + self.prescan_width + serial_overscan_width)
        image = np.random.normal(loc=self.output_amplifier.global_offset, 
                                 scale=self.output_amplifier.noise, 
                                 size=(iy, ix))
        free_charge = copy.deepcopy(self.segarr)

        ## Keyword override toggles
        if kwargs.get('no_trapping', False):
            do_trapping = False
        else:
            do_trapping = self.do_trapping
        if kwargs.get('no_local_offset', False):
            do_local_offset = False
        else:
            do_local_offset = self.output_amplifier.do_local_offset
        if kwargs.get('no_cti', False):
            cti = 0.0
        else:
            cti = self.cti
        
        offset = np.zeros(self.ny)
        cte = 1 - cti
        if do_trapping:
            for trap in self.serial_traps:
                trap.initialize(self.ny, self.nx, self.prescan_width)
            
        for i in range(ix):

             ## Trap capture
            if do_trapping:
                for trap in self.serial_traps:
                    captured_charge = trap.trap_charge(free_charge)
                    free_charge -= captured_charge

            ## Pixel-to-pixel proportional loss
            transferred_charge = free_charge*cte
            deferred_charge = free_charge*cti

            ## Pixel transfer and readout
            if do_local_offset:
                offset = self.output_amplifier.local_offset(offset, 
                                                            transferred_charge[:, 0])
                image[:iy-parallel_overscan_width, i] += transferred_charge[:, 0] + offset
            else:
                image[:iy-parallel_overscan_width, i] += transferred_charge[:, 0]
            free_charge = np.pad(transferred_charge, ((0, 0), (0, 1)), 
                                 mode='constant')[:, 1:] + deferred_charge

            ## Trap emission
            if do_trapping:
                for trap in self.serial_traps:
                    released_charge = trap.release_charge()
                    free_charge += released_charge

        return image/float(self.output_amplifier.gain)
        
    @staticmethod
    def sim_fe55_hit(random_seed=None, stamp_length=6, psf_fwhm=0.00016,
                     hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 postage stamp.

        A single Fe55 x-ray hit is simulated using Galsim.  This simulates
        charge spreading due to sensor effects (the sensor PSF).  The
        result is a postage stamp containing the Fe55 x-ray hit.

        Args:
            random_seed (float): Random number generator seed.
            stamp_length (int): Side length of desired Fe55 postage stamp.
            psf_fwhm (float): FWHM of sensor PSF.
            hit_flux (int): Total flux per Fe55 x-ray hit.
            hit_hlr (float): Half-light radius of Fe55 x-ray hits.

        Returns:
            NumPy array.
        """
        
        ## Set image parameters
        pixel_scale = 0.2
        sy = sx = stamp_length
        psf_fwhm = psf_fwhm
        gal_flux = hit_flux
        gal_hlr = hit_hlr
        gal_e = 0.0
        dy, dx = np.random.rand(2)-0.5

        ## Set galsim parameters
        gsparams = galsim.GSParams(folding_threshold=1.e-2,
                                   maxk_threshold=2.e-3,
                                   xvalue_accuracy=1.e-4,
                                   kvalue_accuracy=1.e-4,
                                   shoot_accuracy=1.e-4,
                                   minimum_fft_size=64)
        
        if random_seed is not None:
            rng = galsim.UniformDeviate(random_seed)
        else:
            rng = galsim.UniformDeviate(0)
        
        ## Generate stamp with Gaussian image
        image = galsim.ImageF(sy, sx, scale=pixel_scale)
        psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=gsparams)
        gal = galsim.Gaussian(half_light_radius=1, gsparams=gsparams)       
        gal = gal.withFlux(gal_flux)
        gal = gal.dilate(gal_hlr)
        final = galsim.Convolve([gal, psf])
        sensor = galsim.sensor.SiliconSensor(rng=rng, diffusion_factor=1)
        stamp = final.drawImage(image, method='phot', rng=rng,
                                offset=(dx,dy),sensor=sensor)

        return stamp

    @staticmethod
    def sim_star(flux, psf_fwhm, stamp_length=40, random_seed=None):
        """Simulate a star postage stamp."""

        ## Set image parameters
        pixel_scale = 0.2
        sy =  sx = stamp_length
        psf_fwhm = psf_fwhm
        dy, dx = np.random.rand(2)-0.5

        if random_seed is not None:
            rng = galsim.UniformDeviate(random_seed)
        else:
            rng = galsim.UniformDeviate(0)

        ## Generate stamp with PSF image
        image = galsim.ImageF(sy, sx, scale=pixel_scale)
        psf = galsim.Kolmogorov(fwhm=psf_fwhm, scale_unit=galsim.arcsec)
        psf = psf.withFlux(flux)
        sensor = galsim.sensor.SiliconSensor(rng=rng, diffusion_factor=1)
        stamp = psf.drawImage(image, rng=rng, offset=(dx, dy), sensor=sensor)

        return stamp


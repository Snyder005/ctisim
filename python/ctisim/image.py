import os
import galsim
import numpy as np
import multiprocessing
from astropy.io import fits

class SegmentSimulator:
    """Controls the creation of simulated segment images.

    Attributes:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        num_serial_prescan (int): Number of serial prescan pixels.
        image (numpy.array): NumPy array containg the image pixels.
    """

    def __init__(self, shape, num_serial_prescan):

        self.nrows, self.ncols = shape
        self.num_serial_prescan = num_serial_prescan
        self._imarr = np.zeros((self.nrows, self.ncols+self.num_serial_prescan), 
                               dtype=np.float32)
        
    @classmethod
    def from_amp_geom(cls, amp_geom):
        """Initialize a SegmentSimulator object from a amp geometry dictionary.

        This method is a convenience function to initialize a SegmentSimulator
        object from a dictionary containing the necessary segment geometry
        information.

        Args:
            amp_geom ('dict' of 'int'): Parameters defining geometry of a segment.

        Returns:
            SegmentSimulator.
        """
        num_serial_prescan = amp_geom['num_serial_prescan']
        nrows = amp_geom['nrows']
        ncols = amp_geom['ncols']
        
        return cls((nrows, ncols), num_serial_prescan)

    @property
    def image(self):
        """Return current segment image."""

        return self._imarr

    def reset(self):
        """Reset segment image to zeros."""

        self._imarr = np.zeros((self.nrows, self.ncols+self.num_serial_prescan), 
                               dtype=np.float32)
        
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
        if len(signal_list) != self.nrows:
            raise ValueError
            
        ramp = np.tile(signal_list, (self.ncols, 1)).T
        self._imarr[:, self.num_serial_prescan:] += ramp
        
    def flatfield_exp(self, signal, noise=True):
        """Simulate a flat field exposure.

        This method simulates a flat field segment image with given signal level.
        The simulated image can be generated with or with out shot noise.

        Args:
            signal (float): Signal level of the flat field.
            noise (bool): Specifies inclusion of shot noise.
        """
        if noise:
            flat = np.random.poisson(signal, size=(self.nrows, self.ncols))
        else:
            flat = np.ones((self.nrows, self.ncols))*signal
        self._imarr[:, self.num_serial_prescan:] += flat

    def fe55_exp(self, num_fe55_hits, stamp_length=6, random_seed=None, psf_fwhm=0.00016, 
                 hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 exposure.

        This method simulates a Fe55 soft x-ray segment image using the Galsim module.  
        Fe55 x-ray hits are randomly generated as postage stamps and positioned 
        randomly on the segment image.

        Args:
            num_fe55_hits (int): Number of Fe55 x-ray hits to perform.
            stamp_length (int): Side length of desired Fe55 postage stamp.
            random_seed (float): Random number generator seed.
            psf_fwhm (float): FWHM of sensor PSF.
            hit_flux (int): Total flux per Fe55 x-ray hit.
            hit_hlr (float): Half-light radius of Fe55 x-ray hits.
        """
        for i in range(num_fe55_hits):
            
            stamp = self.sim_fe55_hit(random_seed=random_seed, stamp_length=stamp_length,
                                      psf_fwhm=psf_fwhm, hit_flux=hit_flux, hit_hlr=hit_hlr).array
            sy, sx = stamp.shape

            y0 = np.random.randint(0, self.nrows-sy)
            x0 = np.random.randint(self.num_serial_prescan,
                                   self.ncols+self.num_serial_prescan-sx)

            self._imarr[y0:y0+sy, x0:x0+sx] += stamp
        
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

class ImageSimulator:
    """Represent an 16-channel LSST image.

    Attributes:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        num_serial_overscan (int): Number of serial overscan pixels.
        num_parallel_overscan (int): Number of parallel overscan pixels.
        readout_amplifiers ('dict' of 'ReadoutAmplifier'): Dictionary
            of ReadoutAmplifier objects for each segment.
        serial_registers ('dict' of 'SerialRegister'): Dictionary
            of SerialRegister objects for each segment.
        segments ('dict' of 'SegmentSimulator'): Dictionary
            of SegmentSimulator objects for each segment.
    """
    
    def __init__(self, shape, num_serial_prescan, num_serial_overscan, 
                 num_parallel_overscan, output_amplifiers, serial_registers):
        
        self.nrows, ncols = shape
        self.num_serial_overscan = num_serial_overscan
        self.num_parallel_overscan = num_parallel_overscan
        self.output_amplifiers = output_amplifiers
        self.serial_registers = serial_registers

        self.segments = {i : SegmentSimulator(shape, num_serial_prescan) for i in range(1, 17)}
        
    @classmethod
    def from_amp_geom(cls, amp_geom, output_amplifiers, serial_registers):
        """Initialize an ImageSimulator object from amplifier geometry dictionary.

        Args:
            amp_geom ('dict' of 'int'): Parameters defining geometry of a segment. 
            readout_amplifiers ('dict' of 'ReadoutAmplifier'): Dictionary
                of ReadoutAmplifier objects for each segment.
            serial_registers ('dict' of 'SerialRegister'): Dictionary
                of SerialRegister objects for each segment.

        Returns:
            ImageSimulator.
        """
        nrows = amp_geom['nrows']
        ncols = amp_geom['ncols']
        num_serial_prescan = amp_geom['num_serial_prescan']
        num_serial_overscan = amp_geom['num_serial_overscan']
        num_parallel_overscan = amp_geom['num_parallel_overscan']
        
        return cls((nrows, ncols), num_serial_prescan, num_serial_overscan, 
                   num_parallel_overscan, output_amplifiers, serial_registers)
        
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
            
    def serial_readout(self, template_file, bitpix=32, outfile='simulated_image.fits', 
                       do_multiprocessing=False, **kwds):
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
        if do_multiprocessing:
            manager = mp.Manager()
            segarr_dict = manager.dict()
            job = [mp.Process(target=self.segment_readout, 
                              args=(segarr_dict, amp)) for amp in range(1, 17)]

            _ = [p.start() for p in job]
            _ = [p.join() for p in job]

        else:
            segarr_dict = {}
            for amp in range(1, 17):
                self.segment_readout(segarr_dict, amp)

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
            output.writeto(outfile, **kwds)
            
        return segarr_dict

    def segment_readout(self, segarr_dict, amp):
        """Simulate readout of a single segment.

        This method is to facilitate the use of multiprocessing when reading out 
        an entire image (16 segments). 

        Args:
            segarr_dict ('dict' of 'numpy.array'): Dictionary of array results.
            amp (int): Amplifier number.
        """

        im = self.output_amplifiers[amp].serial_readout(self.segments[amp], self.serial_registers[amp],
                                                        num_serial_overscan=self.num_serial_overscan,
                                                        num_parallel_overscan=self.num_parallel_overscan)
        segarr_dict[amp] = im
    
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

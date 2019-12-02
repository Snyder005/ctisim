import galsim
import numpy as np
import os
import copy
import multiprocessing as mp
from astropy.io import fits

class SerialTrap:
    """Represents a serial register trap.

    Attributes:
        density_factor (float): Fraction of pixel signal exposed to trap.
        emission_time (float): Trap emission time constant [1/transfers].
        trap_size (float): Size of charge trap [e-].
        location (int): Serial pixel location of trap.
    """
    
    def __init__(self, density_factor, emission_time, trap_size, location):
        
        self.density_factor = density_factor
        self.emission_time = emission_time
        self.trap_size = trap_size
        self.location = location

class SerialRegister:
    """Object representing a serial register of a segment.

    Attributes:
        length (int): Length of serial register [pixels].
        trap (SerialTrap, optional): Serial register trap.
        cti (float, optional): Value of proportional loss due to CTI.
        serial_register (numpy.array): Array holding charge trap values.
    """
    
    def __init__(self, length, cti=0.0, trap=None):
        
        self.length = length
        self.trap = None
        self.cti = cti
        self.serial_register = np.zeros(length)
        
        if trap is not None:
            self.add_trap(trap)
       
    @classmethod
    def from_mcmc_results(cls, mcmc_results, length, mean_func=np.mean, burnin=500):
        """Initialize SerialRegister object from MCMC results file."""

        raise NotImplementedError
        
    def add_trap(self, trap):
        """Add charge trapping to trap locations in serial register.

        This method checks that the SerialTrap is compliant, before adding
        to the charge trapping array.

        Args:
            trap (SerialTrap): SerialTrap to add to serial register.

        Raises:
            ValueError: If trap location is outside serial register dimensions.
        """         
        if trap.location < self.length:
            self.serial_register[trap.location] = trap.trap_size
        else:
            raise ValueError("Serial trap locations must be less than {0}".format(self.length))
        self.trap = trap
                
    def make_readout_array(self, nrows):
        """Create tiled charge trapping array.

        This method extends the charge trapping array along the y-direction
        for use in serial readout of a segment image. This is to facilitate
        performing serial transfer operations on every row of the image at once.

        Args:
            nrows (int): Length of y-dimension to tile charge trapping array.

        Returns:
            NumPy array.
        """
        trap_array = np.tile(self.serial_register, (nrows, 1))
        
        return trap_array

class ReadoutAmplifier:
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
    
    def __init__(self, noise, gain=1.0, offset=0.0, biasdrift_params=None):
        
        self.noise = noise
        self.offset = offset
        self.gain = gain
        self.do_bias_drift = False
        
        if biasdrift_params is not None:
            self.add_bias_drift(biasdrift_params)
    
    @classmethod
    def from_eotest_results(cls, eotest_results, offsets=None):
        """Initialize a dictionary of ReadoutAmplifier objects from eotest results.

        This method is a convenience function for initializing a series of 
        ReadoutAmplifier objects from existing electro-optical test results.
        The appropriate segment gain and read noise values are taken from
        the test results and used to create each ReadoutAmplifier.

        Args:
            eotest_results (str): FITs file containing sensor eotest results.
            offsets ('dict' of 'float'): Dictionary of bias offset levels.

        Returns:
            Dictionary of 'ReadoutAmplifier' objects.
        """
        if offsets is None:
            offsets = {amp : 0.0 for amp in range(1, 17)}
            
        readout_amps = {}
        with fits.open(eotest_results) as hdulist:
            
            for ampno in range(1, 17):
                
                offset = offsets[ampno]
                noise = hdulist[1].data['READ_NOISE'][ampno-1]
                gain = hdulist[1].data['GAIN'][ampno-1]
                
                readout_amp = cls(noise, gain=gain, offset=offset)
                readout_amps[ampno] = readout_amp
                
        return readout_amps
    
    def add_bias_drift(self, bias_drift_params):
        """Add parameters for bias drift.

        This method sets the parameters to perform bias drifting during
        serial readout.

        Args:
            bias_drift_params ('tuple' of 'float'): Bias drift parameters.
        """
        self.drift_size = bias_drift_params[0]
        self.drift_tau = bias_drift_params[1]
        self.drift_threshold = bias_drift_params[2]
        self.do_bias_drift = True
                
    def serial_readout(self, segment, serial_register, num_serial_overscan=10, 
                       num_parallel_overscan=0):
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
        nrows = segment.nrows
        ncols = segment.ncols + segment.num_serial_prescan
        iy, ix = nrows, ncols+num_serial_overscan

        image = np.random.normal(loc=self.offset, scale=self.noise, 
                                 size=(iy+num_parallel_overscan, ix))
        free_charge = copy.deepcopy(segment.image)
        trapped_charge = np.zeros((nrows, ncols))
        cti = serial_register.cti
        cte = 1 - cti
        
        trap_array = serial_register.make_readout_array(nrows)
        
        drift = np.zeros(nrows)
        
        for i in range(ix):
            
            ## Capture charge (if traps exist)
            if serial_register.trap is not None:
                captured_charge = np.clip(free_charge*serial_register.trap.density_factor, 
                                          trapped_charge, trap_array) - trapped_charge
                trapped_charge += captured_charge
                free_charge -= captured_charge
    
            ## Pixel-to-pixel proportional loss
            transferred_charge = free_charge*cte
            deferred_charge = free_charge*cti
            
            ## Perform readout (with optional bias drift)
            if self.do_bias_drift:
                new_drift = np.maximum(self.drift_size*(transferred_charge[:, 0]-self.drift_threshold), np.zeros(nrows))
                drift = np.maximum(new_drift, drift*np.exp(-1/self.drift_tau))                
                image[:iy, i] += transferred_charge[:, 0] + drift
            else:
                image[:iy, i] += transferred_charge[:, 0]
            free_charge = np.pad(transferred_charge, ((0, 0), (0, 1)), mode='constant')[:, 1:] + deferred_charge
                        
            ## Release charge (if traps exist)
            if serial_register.trap is not None:
                released_charge = trapped_charge*(1-np.exp(-1./serial_register.trap.emission_time))
                trapped_charge -= released_charge        
                free_charge += released_charge
            
        return image/float(self.gain)

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
                 num_parallel_overscan, readout_amplifiers, serial_registers):
        
        self.nrows, ncols = shape
        self.num_serial_overscan = num_serial_overscan
        self.num_parallel_overscan = num_parallel_overscan
        self.readout_amplifiers = readout_amplifiers
        self.serial_registers = serial_registers

        self.segments = {i : SegmentSimulator(shape, num_serial_prescan) for i in range(1, 17)}
        
    @classmethod
    def from_amp_geom(cls, amp_geom, readout_amplifiers, serial_registers):
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
                   num_parallel_overscan, readout_amplifiers, serial_registers)
        
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

        im = self.readout_amplifiers[amp].serial_readout(self.segments[amp], self.serial_registers[amp],
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

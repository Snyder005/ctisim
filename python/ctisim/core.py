import galsim
import numpy as np
import os
import copy
from astropy.io import fits

galsim.meta_data.share_dir = '/nfs/slac/g/ki/ki19/lsst/snyder18/LSST/lsst_stack/stack/miniconda3-4.5.12-1172c30/Linux64/galsim/2.1.4.lsst+6/share/galsim/'

class SerialTrap:
    """Represents a serial register trap."""
    
    def __init__(self, density_factor, emission_time, trap_size, locations):
        
        self.density_factor = density_factor
        self.emission_time = emission_time
        self.trap_size = trap_size

        if not isinstance(locations, list):
            locations = [locations]
        self.locations = locations

class SerialRegister:
    """Object representing a serial register of a segment."""
    
    def __init__(self, length, cti=0.0, trap=None):
        
        self.length = length
        self.trap = None ## modify to contain more than one trap
        self.cti = cti
        self.serial_register = np.zeros(length)
        
        if trap is not None:
            self.add_trap(trap)
       
    @classmethod
    def from_mcmc_results(cls, mcmc_results, length, mean_func=np.mean, burnin=500):

        raise NotImplementedError
        
    def add_trap(self, trap):
                    
        for l in trap.locations:
            if l < self.length:
                self.serial_register[l] = trap.trap_size
            else:
                raise ValueError("Serial trap locations must be less than {0}".format(self.length))
        self.trap = trap
                
    def make_readout_array(self, nrows):
            
        trap_array = np.tile(self.serial_register, (nrows, 1))
        
        return trap_array

class ReadoutAmplifier:
    """Object representing the readout amplifier of a single channel."""
    
    def __init__(self, noise, gain=1.0, offset=0.0, biasdrift_params=None):
        
        self.noise = noise
        self.offset = offset
        self.gain = gain
        self.do_bias_drift = False
        
        if biasdrift_params is not None:
            self.add_bias_drift(biasdrift_params)
    
    @classmethod
    def from_eotest_results(cls, eotest_results, offsets=None):
        """Initialize ReadoutAmplifier objects from eotest results."""
        
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
        self.drift_A = bias_drift_params[0]
        self.drift_threshold = bias_drift_params[1]
        self.drift_tau = bias_drift_params[2]
        self.do_bias_drift = True
                
    def serial_readout(self, segment, num_serial_overscan, num_parallel_overscan, serial_register):
        
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
                captured_charge = np.clip(free_charge*serial_register.trap.density_factor, trapped_charge, trap_array) - trapped_charge
                trapped_charge += captured_charge
                free_charge -= captured_charge
    
            ## Pixel-to-pixel proportional loss
            transferred_charge = free_charge*cte
            deferred_charge = free_charge*cti
            
            if self.do_bias_drift:
                new_drift = np.maximum(self.drift_A*(transferred_charge[:, 0]-self.drift_threshold), np.zeros(nrows))
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

    def __init__(self, shape, num_serial_prescan):

        self.nrows, self.ncols = shape
        self.num_serial_prescan = num_serial_prescan
        self._imarr = np.zeros((self.nrows, self.ncols+self.num_serial_prescan), 
                               dtype=np.float32)
        
    @classmethod
    def from_amp_geom(cls, amp_geom):
        """Initialize a SegmentSimulator object from a amp geometry dictionary."""
        
        num_serial_prescan = amp_geom['num_serial_prescan']
        nrows = amp_geom['nrows']
        ncols = amp_geom['ncols']
        
        return cls((nrows, ncols), num_serial_prescan)

    @property
    def image(self):
        return self._imarr

    def reset(self):
        """Reset segment image to zeros."""

        self._imarr = np.zeros((self.nrows, self.ncols+self.num_serial_prescan), 
                               dtype=np.float32)
        
    def ramp_exp(self, flux_list):
        """Simulate an image with varying flux illumination per row."""
        
        if len(flux_list) != self.nrows:
            raise ValueError
            
        ramp = np.tile(flux_list, (self.ncols, 1)).T
        self._imarr[:, self.num_serial_prescan:] += ramp
        
    def flatfield_exp(self, flux, noise=True):
        """Simulate a flat field exposure."""

        if noise:
            flat = np.random.poisson(flux, size=(self.nrows, self.ncols))
        else:
            flat = np.ones((self.nrows, self.ncols))*flux
        self._imarr[:, self.num_serial_prescan:] += flat

    def fe55_exp(self, num_fe55_hits, stamp_length=6, random_seed=None, psf_fwhm=0.00016, 
                 hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 exposure."""

        for i in range(num_fe55_hits):
            
            stamp = self.sim_fe55_hit(random_seed=random_seed, stamp_length=stamp_length,
                                      psf_fwhm=psf_fwhm, hit_flux=hit_flux, hit_hlr=hit_hlr).array
            sy, sx = stamp.shape

            y0 = np.random.randint(0, self.nrows-sy)
            x0 = np.random.randint(self.num_serial_prescan,
                                   self.ncols+self.num_serial_prescan-sx)

            self._imarr[y0:y0+sy, x0:x0+sx] += stamp
            
    def star_exp(self, num_stars, flux, psf_fwhm, stamp_length=40, random_seed=None):
        """Simulate an exposure of a random star field."""
        
        for i in range(num_stars):
            
            stamp = self.sim_star(flux, psf_fwhm, stamp_length=stamp_length, random_seed=random_seed).array
            sy, sx = stamp.shape
            
            y0 = np.random.randint(0, self.nrows-sy)
            x0 = np.random.randint(self.num_serial_prescan,
                                   self.ncols+self.num_serial_prescan-sx)
            
            self._imarr[y0:y0+sy, x0:x0+sx] += stamp
        
    @staticmethod
    def sim_fe55_hit(random_seed=None, stamp_length=6, psf_fwhm=0.00016,
                     hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 postage stamp."""
        
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
    """Represent an 16-channel LSST image."""
    
    def __init__(self, shape, num_serial_prescan, num_serial_overscan, 
                 num_parallel_overscan, readout_amplifiers):
        
        self.num_serial_overscan = num_serial_overscan
        self.num_parallel_overscan = num_parallel_overscan
        self.readout_amplifiers = readout_amplifiers

        self.amps = {i : SegmentSimulator(shape, num_serial_prescan) for i in range(1, 17)}
        
    @classmethod
    def from_amp_geom(cls, amp_geom, readout_amplifiers):
        """Initialize an ImageSimulator object from amplifier geometry dictionary."""
        
        nrows = amp_geom['nrows']
        ncols = amp_geom['ncols']
        num_serial_prescan = amp_geom['num_serial_prescan']
        num_serial_overscan = amp_geom['num_serial_overscan']
        num_parallel_overscan = amp_geom['num_parallel_overscan']
        
        return cls((nrows, ncols), num_serial_prescan, num_serial_overscan, 
                   num_parallel_overscan, readout_amplifiers)
        
    def flatfield_exp(self, flux, noise=True):
        """Simulate a flat field exposure."""
        
        for i in range(1, 17):            
            self.amps[i].flatfield_exp(flux, noise=noise)

    def fe55_exp(self, num_fe55_hits, stamp_length=6, psf_fwhm=0.00016, 
                 hit_flux=1620, hit_hlr=0.004):
        """Simulate an Fe55 exposure."""
        
        for i in range(1, 17):
            self.amps[i].fe55_exp(num_fe55_hits, stamp_length=stamp_length, 
                                  random_seed=None, psf_fwhm=psf_fwhm, 
                                  hit_flux=hit_flux, hit_hlr=hit_hlr)
            
    def star_exp(self, num_stars, flux, psf_fwhm, stamp_length=40):
        """Simulate star field exposure."""
        
        for i in range(1, 17):
            self.amps[i].star_exp(num_stars, flux, psf_fwhm, stamp_length, 
                                  random_seed=None)
            
    def serial_readout(self, template_file, serial_registers, bitpix=32, 
                       outfile='simulated.fits', **kwds):

        output = fits.HDUList()
        output.append(fits.PrimaryHDU())
        
        imarr_list = []
        for i in range(1, 17):
            
            im = self.readout_amplifiers[i].serial_readout(self.amps[i], 
                                                           self.num_serial_overscan, 
                                                           self.num_parallel_overscan, 
                                                           serial_registers[i])
            imarr_list.append(im)
            output.append(fits.ImageHDU(data=im/self.readout_amplifiers[i].gain))

        ## Use template file to create output FITs file
        with fits.open(template_file) as template:
            output[0].header.update(template[0].header)
            output[0].header['FILENAME'] = os.path.basename(outfile)
            for i in range(1, 17):
                output[i].header.update(template[i].header)
                self.set_bitpix(output[i], bitpix)
            for i in (-3, -2, -1):
                output.append(template[i])
            output.writeto(outfile, **kwds)
            
        return imarr_list
    
    @staticmethod
    def set_bitpix(hdu, bitpix):
        dtypes = {16: np.int16, -32: np.float32, 32: np.int32}
        for keyword in 'BSCALE BZERO'.split():
            if keyword in hdu.header:
                del hdu.header[keyword]
        if bitpix > 0:
            my_round = np.round
        else:
            def my_round(x): return x
        hdu.data = np.array(my_round(hdu.data), dtype=dtypes[bitpix])

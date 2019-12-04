import numpy as np
import copy
from astropy.io import fits

class SegmentModelParams:

    def __init__(self, amp):

        self.amp = amp
        self.cti = 0.0
        self.drift_size = 0.0
        self.drift_tau = np.nan
        self.drift_threshold = 0.0
        self.trap_size = 0.0
        self.density_factor = 0.0
        self.trap_tau = np.nan
        
        self.walkers = 0
        self.steps = 0
        self.mcmc_results = None
        
    def add_mcmc_results(self, mcmc_results, burn_in=0):

        walkers, steps, _ = mcmc_results.shape

        self.walkers = walkers
        self.steps = steps
        self.cti = 10**np.median(mcmc_results[:, burn_in:, 0])
        self.density_factor = np.median(mcmc_results[:, burnin:, 1])
        self.trap_tau = np.median(mcmc_results[:, burnin:, 2])
        self.trap_size = np.median(mcmc_results[:, burnin:, 3])

        self.mcmc_results = mcmc_results

    def update_params(self, **param_results):

        try:
            self.cti = param_results['cti']
        except KeyError:
            pass
    
        ## Add bias drift results
        if param_results.get('drift_size'):
            try:
                self.drift_tau = param_results['drift_tau']
                self.drift_threshold = param_results['drift_threshold']
            except KeyError:
                raise KeyError("Must include drift_tau and drift_threshold values " + \
                               "when updating drift_size.")
            else:
                self.drift_size = param_results['drift_size']

        ## These need to be conditional
        if param_results.get('trap_size'):

            try:
                self.density_factor = param_results['density_factor']
                self.trap_tau = param_results['trap_tau']
            except KeyError:
                raise KeyError("Must include density_factor and trap_tau values " + \
                               "when updating trap_size.")

    def create_table_hdu(self):

        hdr = fits.Header()
        hdr['AMP'] = self.amp
        hdr['WALKERS'] = self.walkers
        hdr['STEPS'] = self.steps

        results = self.mcmc_results.flatten() # see what error raised and add try/except
        cols = [fits.Column(name='CTIEXP', array=results[:, 0], format='E'),
                fits.Column(name='TRAPSIZE', array=results[:, 3], format='E'),
                fits.Column(name='TAU', array=results[:, 2], format='E'),
                fits.Column(name='DFACTOR', array=results[:, 1], format='E')]

        mcmc_hdu = fits.BinTableHDU.from_columns(cols, header=hdr)

        return mcmc_hdu
        
class SensorModelParams:

    def __init__(self):

        self.segment_params = {i : SegmentModelParams(i) for i in range(1, 17)}

    def update_segment_params(self, amp, **param_results):

            self.segment_params[amp].update_params(**param_results)

    def write_fits(self, outfile, overwrite=True):
        
        prihdu = fits.PrimaryHDU()
        
        cols = [fits.Column(name='CTI', array=cti_results, format='E'),
                fits.Column(name='DRIFT_SIZE', array=drift_size_results, format='E'),
                fits.Column(name='DRIFT_TAU', array=drift_tau_results, format='E'),
                fits.Column(name='DRIFT_THRESHOLD', array=drift_threshold_results, format='E'),
                fits.Column(name='TRAP_SIZE', array=trap_size_results, format='E'),
                fits.Column(name='TRAP_TAU', array=trap_tau_results, format='E'),
                fits.Column(name='TRAP_DFACTOR', array=dfactor_results, format='E')]
        hdu = fits.BinTableHDU.from_columns(cols)
        
        hdulist = fits.HDUList([prihdu, hdu])
        for i in range(1, 17):
            if self.segment_params[i].mcmc_results is not None:
                hdulist.append(self.segment_params[i].create_table_hdu())

        hdulist.writeto(outfile, overwrite=overwrite)

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
    def from_mcmc_results(cls, mcmc_results, length):
        """Initialize SerialRegister object from MCMC results file.

        This method is a convenience function for initializing a series of 
        SerialRegister objects from existing MCMC optimization results.

        Args:
            mcmc_results (str): FITs file containing MCMC optimization results.
            length (int): Length of serial register [pixels].
        """
        serial_registers = {}
        with fits.open(mcmc_results) as hdulist:
        
            results = hdulist[1].data

            for amp in range(1, 17):
            
                trapsize = results['TRAP_SIZE'][amp-1]
                if trapsize > 0:
                    trap = SerialTrap(results['TRAP_DFACTOR'][amp-1],
                                      results['TRAP_TAU'][amp-1],
                                      trapsize,
                                      1)
                else:
                    trap = None
                serial_registers[amp] = cls(length, cti=results['CTI'][amp-1],
                                            trap=trap)

        return serial_registers
        
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
    
    def __init__(self, noise, gain=1.0, offset=0.0, biasdrift_params=None):
        
        self.noise = noise
        self.offset = offset
        self.gain = gain
        self.do_bias_drift = False
        
        if biasdrift_params is not None:
            self.add_bias_drift(biasdrift_params)
    
    @classmethod
    def from_eotest_results(cls, eotest_results, mcmc_results=None, offsets=None):
        """Initialize a dictionary of ReadoutAmplifier objects from eotest results.

        This method is a convenience function for initializing a series of 
        ReadoutAmplifier objects from existing electro-optical test results.
        The appropriate segment gain and read noise values are taken from
        the test results and used to create each ReadoutAmplifier.

        Args:
            eotest_results (str): FITs file containing sensor eotest results.
            mcmc_results (str): FITs file containing MCMC optimization results.
            offsets ('dict' of 'float'): Dictionary of bias offset levels.

        Returns:
            Dictionary of 'ReadoutAmplifier' objects.
        """
        if offsets is None:
            offsets = {i : 0.0 for i in range(1, 17)}

        ## Optionally add bias drift parameters
        if mcmc_results is not None:
            with fits.open(mcmc_results) as mcmc_hdulist:
                data = mcmc_hdulist[1].data
                biasdrift_params = {i+1 : (data['DRIFT_SIZE'][i], 
                                      data['DRIFT_TAU'][i], 
                                      data['DRIFT_THRESHOLD'][i]) for i in range(16)}
        else:
            biasdrift_params = {i : None for i in range(1, 17)}
            
        ## Get read noise and gain from eotest results
        readout_amps = {}
        with fits.open(eotest_results) as hdulist:
            
            for ampno in range(1, 17):
                
                offset = offsets[ampno]
                noise = hdulist[1].data['READ_NOISE'][ampno-1]
                gain = hdulist[1].data['GAIN'][ampno-1]
                
                readout_amp = cls(noise, gain=gain, offset=offset,
                                  biasdrift_params = biasdrift_params[ampno])
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

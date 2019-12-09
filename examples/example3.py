#!/usr/bin/env python
"""example3.py

This script is a more detailed version of `example1.py`; simulating the serial
readout of a flat field segment image with proportional loss from CTI as well
as fixed loss from a charge trap.

To run this script, write

    python example3.py <signal> <cti> <trap_size>

where <signal is the desired flat field illumination signal, <cti> is the 
desired value for CTI and <trap_size> is the size of the charge trapping.

The introduced class is:
    * SerialTrap(density_factor, emission_time, trap_size, location)

In this example a serial trap with some basic properties is introduced into the 
serial register, located in a serial pre-scan pixel.  (For more information 
regarding trap properties, consult `SerialTrap` docstrings.)  By changing the 
desired flat field signal level, one can observe the effect of the charge 
trapping on measurements of the CTI.  Recommended comparison:

    python example3.py 50000. 1.E-6 10.0
    python example3.py 5000. 1.E-6 10.0
"""

from ctisim import ITL_AMP_GEOM
from ctisim import SegmentSimulator, SerialRegister, OutputAmplifier, SerialTrap
from ctisim.utils import calculate_cti
import argparse

def main(signal, cti, trap_size):

    # Create simulated image segment with desired signal level
    segment = SegmentSimulator.from_amp_geom(ITL_AMP_GEOM)
    segment.flatfield_exp(signal)

    # Create SerialTrap object.  
    # For now only the `trap_size` parameter will be modified. 
    # The trap will be placed at pixel 1, in the serial prescan.
    density_factor = 0.1
    emission_time = 1.0 
    location = 1
    trap = SerialTrap(density_factor, emission_time, trap_size, location)

    # Create the serial register, now with a trap.
    length = segment.ncols + segment.num_serial_prescan
    serial_register = SerialRegister(length, cti=cti, trap=trap)

    # Create the output amplifier
    output_amplifier = OutputAmplifier(6.5)

    # Simulate readout
    num_oscan = ITL_AMP_GEOM['num_serial_overscan']
    seg_imarr = output_amplifier.serial_readout(segment, serial_register,
                                                num_serial_overscan=num_oscan)

    # Calculate CTI
    last_pix_num = ITL_AMP_GEOM['last_pixel_index']
    result = calculate_cti(seg_imarr, last_pix_num, num_overscan_pixels=2)
    print(result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demonstrate CTI and charge trapping.')
    parser.add_argument('signal', type=float, help='Flat field illumination signal [e-]')
    parser.add_argument('cti', type=float, help='Proportional loss from CTI.')
    parser.add_argument('trap_size', type=float, help='Size of charge trap.')
    args = parser.parse_args()

    signal = args.signal
    cti = args.cti
    trap_size = args.trap_size

    main(signal, cti, trap_size)

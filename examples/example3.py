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
from ctisim import SegmentSimulator, OutputAmplifier, SerialTrap
from ctisim.utils import calculate_cti
import argparse

def main(signal, cti, trap_size):

    amp_geom = ITL_AMP_GEOM
    serial_overscan_width = amp_geom.serial_overscan_width
    parallel_overscan_width = int(amp_geom.naxis2 - amp_geom.ny)
    last_pixel = amp_geom.nx + amp_geom.prescan_width - 1

    # Create the output amplifier
    output_amplifier = OutputAmplifier(1.0, 6.5)

    # Create simulated image segment with desired signal level
    segment = SegmentSimulator.from_amp_geom(amp_geom, output_amplifier, cti=cti)
    segment.flatfield_exp(signal)

    # Create SerialTrap object.  
    # For now only the `trap_size` parameter will be modified. 
    # The trap will be placed at pixel 1, in the serial prescan.
    size = trap_size
    scaling = 0.1
    emission_time = 1.0
    threshold = 0.0
    pixel = 1
    trap = SerialTrap(size, scaling, emission_time, threshold, pixel)

    segment.add_trap(trap)

    # Simulate readout
    seg_imarr = segment.simulate_readout(serial_overscan_width = serial_overscan_width,
                                         parallel_overscan_width = parallel_overscan_width,
                                         do_trapping = True, do_bias_drift = False)

    # Calculate CTI
    last_pix_num = amp_geom.nx + amp_geom.prescan_width - 1
    result = calculate_cti(seg_imarr, last_pixel, num_overscan_pixels=2)
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

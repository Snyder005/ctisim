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
from ctisim import SegmentSimulator, OutputAmplifier, LogisticTrap
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
    print("Simulating flat field image with signal: {0} electrons".format(signal))
    segment = SegmentSimulator.from_amp_geom(amp_geom, output_amplifier, cti=cti)
    segment.flatfield_exp(signal)

    # Create SerialTrap object.  
    # For now only the `trap_size` parameter will be modified. 
    # The trap will be placed at pixel 1, in the serial prescan.
    print("Adding linear serial trap of size: {0:.1f} electrons".format(size))
    size = trap_size
    k = 1.0
    emission_time = 1.0
    f0 = 10.0
    pixel = 1
    trap = LogisticTrap(size, emission_time, pixel, f0, k)

    segment.add_trap(trap)

    # Simulate readout
    print("Simulating readout with CTI: {0:.1E}".format(cti))
    seg_imarr = segment.simulate_readout(serial_overscan_width = serial_overscan_width,
                                         parallel_overscan_width = parallel_overscan_width,
                                         do_bias_drift = False)

    # Calculate CTI
    last_pix_num = amp_geom.nx + amp_geom.prescan_width - 1
    result = calculate_cti(seg_imarr, last_pixel, num_overscan_pixels=2)
    print("CTI from EPER result: {0:.4E}".format(result))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demonstrate CTI and logistic trapping.')
    parser.add_argument('signal', type=float, help='Flat field illumination signal [e-]')
    parser.add_argument('cti', type=float, help='Proportional loss from CTI.')
    parser.add_argument('trap_size', type=float, help='Size of charge trap.')
    args = parser.parse_args()

    signal = args.signal
    cti = args.cti
    trap_size = args.trap_size

    main(signal, cti, trap_size)

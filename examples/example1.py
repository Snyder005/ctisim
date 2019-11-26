#!/usr/bin/env python
"""example1.py

This script is a worked example of simulating the serial readout
of a flat field segment image, with proportional loss from charge transfer
inefficiency (CTI) occuring at each pixel transfer.

To run this script, write

    python example1.py <cti>

where <cti> is the desired value for CTI.

The demonstrated class usages are:
    * SegmentSimulator(shape, num_serial_prescan)
    * SerialRegister(length, cti)
    * ReadoutAmplifier(noise)

This example makes use of the `ITL_AMP_GEOM` utility dictionary,
which contains all the necessary pixel geometry information 
corresponding to an ITL CCD segment.
"""

from ctisim import ITL_AMP_GEOM
from ctisim import SegmentSimulator, SerialRegister, ReadoutAmplifier
from ctisim.utils import calculate_cti
import argparse

def main(cti):

    # Create a SegmentSimulator object using `from_amp_geom()` method.
    # This method constructs a SegmentSimulator from a dictionary
    # containing information on the segment geometry.
    segment = SegmentSimulator.from_amp_geom(ITL_AMP_GEOM)
    segment.flatfield_exp(50000.)

    # Create a SerialRegister object with the desired CTI value.
    # The length of the register will be the segment columns
    # plus any additional physical prescan pixels.
    length = segment.ncols + segment.num_serial_prescan
    serial_register = SerialRegister(length, cti)

    # Create a ReadoutAmplifier object with 6.5 electrons of noise.
    readout_amplifier = ReadoutAmplifier(6.5)

    # The `serial_readout()` method creates the final image.
    seg_imarr = readout_amplifier.serial_readout(segment, serial_register,
                                                 num_serial_overscan=10)

    ## Calculate CTI using the `calculate_cti()` utility function.
    last_pix_num = ITL_AMP_GEOM['last_pixel_index']
    result = calculate_cti(seg_imarr, last_pix_num, num_overscan_pixels=2)
    print(result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demonstrate CTI calculation')
    parser.add_argument('cti', type=float, help='Proportional loss from CTI.')
    args = parser.parse_args()

    cti = args.cti
    main(cti)

    
    

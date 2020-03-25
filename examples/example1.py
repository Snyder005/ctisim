#!/usr/bin/env python
"""example1.py

This script is a worked example of simulating the serial readout
of a flat field segment image, with proportional loss from charge transfer
inefficiency (CTI) occuring at each pixel transfer.

To run this script, write

    python example1.py <cti>

where <cti> is the desired value for CTI.

The demonstrated class usages are:
    * OutputAmplifier(gain, noise)
    * SegmentSimulator.from_amp_geom(amp_geom, output_amplifier, cti)

This example makes use of the `ITL_AMP_GEOM` utility dictionary,
which contains all the necessary pixel geometry information 
corresponding to an ITL CCD segment.
"""

from ctisim import ITL_AMP_GEOM
from ctisim import SegmentSimulator, BaseOutputAmplifier
from ctisim.utils import calculate_cti
import argparse

def main(cti):

    amp_geom = ITL_AMP_GEOM
    serial_overscan_width = amp_geom.serial_overscan_width
    last_pixel = amp_geom.nx + amp_geom.prescan_width - 1

    # Create a BaseOutputAmplifier object with 6.5 electrons of noise.
    output_amplifier = BaseOutputAmplifier(1.0, noise=6.5)

    # Create a SegmentSimulator object using `from_amp_geom()` method.
    # This method constructs a SegmentSimulator from a dictionary
    # containing information on the segment geometry.
    print("Simulating flat field image with signal: 50000.0 electrons")
    segment = SegmentSimulator.from_amp_geom(amp_geom, output_amplifier, cti=cti)
    segment.flatfield_exp(50000)

    # The `readout()` method creates the final image.
    print("Simulating readout with CTI: {0:.1E}".format(cti))
    seg_imarr = segment.readout(serial_overscan_width=serial_overscan_width)

    ## Calculate CTI using the `calculate_cti()` utility function.
    result = calculate_cti(seg_imarr, last_pixel, num_overscan_pixels=2)
    print("CTI from EPER result: {0:.4E}".format(result))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demonstrate CTI calculation')
    parser.add_argument('cti', type=float, help='Proportional loss from CTI.')
    args = parser.parse_args()

    cti = args.cti
    main(cti)

    
    

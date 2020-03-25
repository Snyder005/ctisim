import os
import argparse
from ctisim import LinearTrap, LogisticTrap, ImageSimulator
from ctisim import BaseOutputAmplifier

def main(sensor_id, infile, output_dir='./', bias_frame=None):

    outfile = os.path.join(output_dir, '{0}_processed.fits'.format(sensor_id))

    low_trap = LinearTrap(4.0, 0.4, 1, 0.08)
#    med_trap = LogisticTrap(40.0, 0.4, 1, 17500., 0.001)
    med_trap = LogisticTrap(1000.0, 0.4, 1, 60000., 0.0002)

    output_amps = {amp : BaseOutputAmplifier(1.0) for amp in range(1, 17)}
    traps_dict = {amp : [low_trap, med_trap] for amp in range(1, 17)}
    cti_dict = {amp : 0.0 for amp in range(1, 17)}

    imsim = ImageSimulator.from_image_fits(infile, output_amps, 
                                           cti=cti_dict, traps=traps_dict,
                                           bias_frame=bias_frame)
    imarr_results = imsim.image_readout(infile, outfile=outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str, help='Sensor identifier.')
    parser.add_argument('infile', type=str, help='Existing FITs file image.')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, output_dir=args.output_dir)

    

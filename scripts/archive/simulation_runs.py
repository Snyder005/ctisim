import argparse
from os.path import join

from ctisim import BaseOutputAmplifier, FloatingOutputAmplifier
from ctisim import LinearTrap, LogisticTrap
from ctisim import ImageSimulator

def main(sensor_id, infiles, output_dir='./'):

    ## CTI parameters
    cti = 1.E-6
    cti_dict = {amp : cti for amp in range(1, 17)}
    
    ## Trapping parameters
#    trap = LinearTrap(4.0, 0.4, 1, 0.08)
    trap = LogisticTrap(40.0, 0.4, 1, 17500, 0.001)
#    trap = None
    traps_dict = {amp : trap for amp in range(1, 17)}

    ## Electronic parameters
    output_amps = {amp : FloatingOutputAmplifier(1.0, 0.0002, 2.4, 0.0) for amp in range(1, 17)}
#    output_amps = {amp : BaseOutputAmplifier(1.0) for amp in range(1, 17)}

    for i, infile in enumerate(infiles):

        ## Process infile
        processed_file = join(output_dir, 
                              '{0}_{1:03d}_processed.fits'.format(sensor_id, i))
        imsim = ImageSimulator.from_image_fits(infile, output_amps, cti=cti_dict, 
                                               traps=traps_dict)
        imarr_results = imsim.image_readout(infile, outfile=processed_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.infiles, output_dir=args.output_dir)

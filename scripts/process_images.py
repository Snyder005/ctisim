import argparse
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
import warnings
import os
import pickle

from ctisim.utils import OverscanParameterResults
from ctisim import ImageSimulator

def main(sensor_id, infile, main_dir, gain_file=None, output_dir='./', include_noise=False):

    ## Get gains
    if gain_file is not None:
        with open(gain_file, 'rb') as f:
            gain_results = pickle.load(f)
            gains = gain_results.get_amp_gains(sensor_id)
    else:
        gains = {i : 1.0 for i in range(1, 17)}

    ## Optional noise
    if include_noise:
        noise_dict = {i : 6.5 for i in range(1, 17)}
    else:
        noise_dict = {i : 0.0 for i in range(1, 17)}

    offset_dict = {i : 0.0 for i in range(1, 17)}

    ## Get CTI and output amplifiers
    param_file = os.path.join(main_dir, 
                              '{0}_parameter_results.fits'.format(sensor_id))
    param_results = OverscanParameterResults.from_fits(param_file)
    cti_results = param_results.cti_results
    output_amplifiers = param_results.all_output_amplifiers(gains, 
                                                            noise_dict, 
                                                            offset_dict)

    ## Get traps
    traps = {}
    for i in range(1, 17):
        trap_file = os.path.join(main_dir, 
                                 '{0}_amp{1}_trap.pkl'.format(sensor_id, i))
        traps[i] = pickle.load(open(trap_file, 'rb'))

    ## Output filename
    base = os.path.splitext(os.path.basename(infile))[0]
    outfile = os.path.join(output_dir, '{0}_processed.fits'.format(base))

    image = ImageSimulator.from_image_fits(infile, output_amplifiers, 
                                           cti=cti_results, traps=traps)

    image.image_readout(infile, outfile=outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    parser.add_argument('--gain_file', '-g', type=str, default=None)
    parser.add_argument('--noise', '-n', action='store_true')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, args.main_dir, 
         gain_file=args.gain_file, output_dir=args.output_dir,
         include_noise=args.noise)


        

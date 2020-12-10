import numpy as np
import argparse
import subprocess

def main(sensor_id, main_dir, infiles, gain_file=None, output_dir='./', bias_frame=None):
    
    for i, infile in enumerate(infiles):

        command = ['bsub', '-W', '1:00', '-R', 'bullet', '-o', 
                   '/nfs/slac/g/ki/ki19/lsst/snyder18/log/logfile_correct_images_{0:03d}.log'.format(i), 
                   'python', 'correct_images.py', sensor_id, infile, main_dir,
                   '-o', output_dir]
        if gain_file is not None:
            command.append('-g')
            command.append(gain_file)
        if bias_frame is not None:
            command.append('-b')
            command.append(bias_frame)
        subprocess.check_output(command)
        print("Processing {0}, submitted to batch farm.".format(infile)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('--gain_file', '-g', type=str, default=None)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    parser.add_argument('--bias_frame', '-b', type=str, default=None)
    args = parser.parse_args()

    main(args.sensor_id, args.main_dir, args.infiles, 
         gain_file=args.gain_file, output_dir=args.output_dir,
         bias_frame=args.bias_frame)

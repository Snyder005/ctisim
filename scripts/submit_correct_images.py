import numpy as np
import argparse
import subprocess

def main(sensor_id, main_dir, infiles, output_dir='./'):
    
    for i, infile in enumerate(infiles):

        command = ['bsub', '-W', '1:00', '-R', 'bullet', '-o', 
                   '/nfs/slac/g/ki/ki19/lsst/snyder18/log/logfile_{0:03d}.log'.format(i), 
                   'python', 'correct_images.py', sensor_id, infile, main_dir,
                   '-o', output_dir]
        subprocess.check_output(command)
        print("Processing {0}, submitted to batch farm.".format(infile)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.main_dir, args.infiles, output_dir=args.output_dir)

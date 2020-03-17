import numpy as np
import argparse
import subprocess

def main(infiles, output_dir='./'):
    
    for i, infile in enumerate(infiles):

        command = ['bsub', '-W', '1:00', '-R', 'bullet', '-o', '/nfs/slac/g/ki/ki19/lsst/snyder18/log/logfile_{0:03d}.log'.format(i), 
                   'python', 'sim_from_existing.py', infile, '-o', output_dir]
        subprocess.check_output(command)
        print("Processing {0}, submitted to batch farm.".format(infile)) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    infiles = args.infiles
    output_dir = args.output_dir
    main(infiles, output_dir=output_dir)

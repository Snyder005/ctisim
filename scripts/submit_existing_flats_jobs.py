import numpy as np
import argparse
import subprocess

def main(infiles, mcmc_results, output_dir='./'):
    
    for i, infile in enumerate(infiles):

        command = ['bsub', '-W', '1:00', '-R', 'bullet', '-o', './logs/logfile_{0:03d}.log'.format(i), 
                   'python', 'sim_from_existing.py', infile, mcmc_results, '-o', output_dir]
        subprocess.check_output(command)
        print("Processing {0}, submitted to batch farm.".format(infile)) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mcmc_results', type=str)
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    infiles = args.infiles
    mcmc_results = args.mcmc_results
    output_dir = args.output_dir
    main(infiles, mcmc_results, output_dir=output_dir)

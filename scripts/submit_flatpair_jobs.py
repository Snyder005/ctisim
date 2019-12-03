import numpy as np
import argparse
import subprocess

def main(num_flatpairs, eotest_results, mcmc_results, template_file, output_dir='./'):

    signal_array = np.logspace(2, np.log10(150000), num_flatpairs)
    
    for i in range(num_flatpairs):

        signal = signal_array[i]
        command = ['bsub', '-W', '1:00', '-R', 'bullet', '-o', 'logfile.log', 'python', 
                   'simulate_flatpair.py', '{0:.1f}'.format(signal), eotest_results, 
                   mcmc_results, template_file, '-o', output_dir]
        subprocess.check_output(command)
        print("Flat pair {0:.1f} e-, submitted to batch farm.".format(signal)) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('num_flatpairs', type=int, 
                        help='Number of flat field acquisitions to simulate.')
    parser.add_argument('eotest_results', type=str)
    parser.add_argument('mcmc_results', type=str)
    parser.add_argument('template_file', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    num_flatpairs = args.num_flatpairs
    eotest_results = args.eotest_results
    mcmc_results = args.mcmc_results
    template_file = args.template_file
    output_dir = args.output_dir
    main(num_flatpairs, eotest_results, mcmc_results, template_file, output_dir=output_dir)

import sys
import os.path

from endorse.sa import sample, analyze
from endorse.bayes_orig import aux_functions
from endorse.bayes_orig import run_all as bayes_run_all
import endorse.Bukov2.sample_storage as sample_storage

import numpy as np
import csv


sensitivity_dirname = "sensitivity"
param_dirname = "parameters"
empty_hdf_dirname = "empty_hdfs"
pbs_job_dirname = "pbs_jobs"

sampled_data_dirname = "sampled_data"

class Analyze:
    def __init__(self, problem):
        self.problem = problem

    def single(self, output):
        return analyze.sobol(self.problem, output, calc_second_order=True, print_to_console=False)


def prepare_pbs_scripts(sens_config_dict, output_dir_in, np):
    endorse_root = sens_config_dict["rep_dir"]
    met = sens_config_dict["metacentrum"]

    pbs_dir = os.path.join(output_dir_in, pbs_job_dirname)
    aux_functions.force_mkdir(pbs_dir, force=True)

    def create_common_lines(id):
        name = met["name"] + "_" + id
        common_lines = [
            '#!/bin/bash',
            '#PBS -S /bin/bash',
            # '#PBS -l select=' + str(met["chunks"]) + ':ncpus=' + str(met["ncpus_per_chunk"]) + ':mem=' + met["memory"],
            # scratch: charon ssd: 346/20 = 17,3 GB
            '#PBS -l select=1:ncpus=1:mem=' + met["memory"] + ":scratch_local=17gb",
            # '#PBS -l place=scatter',
            '#PBS -l walltime=' + str(met["walltime"]),
            '#PBS -q ' + met["queue"],
            '#PBS -N ' + name,
            '#PBS -j oe',
            '#PBS -o ' + os.path.join(pbs_dir, name + '.out'),
            #'#PBS -e ' + os.path.join(pbs_dir, name + '.err'),
            '\n',
            'set -x',
            'export TMPDIR=$SCRATCHDIR',
            '\n# absolute path to output_dir',
            'output_dir="' + output_dir + '"',
            'workdir=$SCRATCHDIR'
            #'\n',
            #'SWRAP="' + met["swrap"] + '"',
            #'IMG_MPIEXEC="/usr/local/mpich_4.0.3/bin/mpirun"',
            #'SING_IMAGE="' + os.path.join(endorse_root, 'endorse.sif') + '"',
            #'\n',
            #'cd $output_dir',
            #'SCRATCH_COPY=$output_dir',
            #'python3 $SWRAP/smpiexec_prepare.py -i $SING_IMAGE -s $SCRATCH_COPY -m $IMG_MPIEXEC'
        ]
        return common_lines

    pbs_file_list = []
    for n in range(np):
        id = solver_id(n)
        sensitivity_dir = os.path.join("$workdir", sensitivity_dirname)
        csv_file = os.path.join(sensitivity_dir, param_dirname, "params_" + id + ".csv")
        sample_subdir = os.path.join(sensitivity_dir, "samples_" + id)
        sampled_data_out = os.path.join("$workdir", sampled_data_hdf(n))
        # prepare PBS script
        common_lines = create_common_lines(id)
        rsync_cmd = " ".join(["rsync -av",
                              #"--include " + os.path.join(sensitivity_dirname, empty_hdf_dirname, sampled_data_hdf(n)),
                              "--exclude *.h5",
                              "--exclude *.pdf",
                              "--exclude " + os.path.join(sensitivity_dirname, empty_hdf_dirname),
                              "--exclude " + os.path.join(sensitivity_dirname, param_dirname),
                              "--exclude " + os.path.join(sensitivity_dirname, pbs_job_dirname),
                              "$output_dir" + "/",
                              "$workdir"])
        lines = [
            *common_lines,
            '\n',
            rsync_cmd,
            ' '.join(['cp',
                      os.path.join(sensitivity_dirname, empty_hdf_dirname, sampled_data_hdf(n)),
                      "$workdir"]),
            'cd $workdir',
            'pwd',
            'ls -la',
            '\n# finally gather the full command',
            os.path.join(endorse_root, "bin", "endorse-bayes") + " "
                + ' '.join(["-t", "set", "-o", "$workdir", "-p", csv_file, "-x", sample_subdir, "-s", id]),
            # 'zip -r samples.zip solver_*', # avoid 'bash: Argument list too long'
            # 'find . -name "solver_*" -print0 | xargs -0 tar -zcvf samples.tar.gz',
            # 'find . -name "solver_*" -print0 | xargs -0 rm -r',
            # '\n' + ' '.join(['tar', '-zcvf', 'samples_' + id + '.tar.gz', sample_subdir]),
            # ' '.join(['rm', '-r', sample_subdir]),
            'ls -la',
            'mkdir -p $output_dir/sampled_data',
            'cp ' + sampled_data_out + ' $output_dir/sampled_data',
            #'cp -r sensitivity $output_dir',
            'clean_scratch',
            'echo "FINISHED"'
        ]
        pbs_file = os.path.join(pbs_dir, "pbs_job_" + id + ".sh")
        with open(pbs_file, 'w') as f:
            f.write('\n'.join(lines))
        pbs_file_list.append(pbs_file)

    return pbs_file_list


def solver_id(i):
    return str(i).zfill(2)


def sampled_data_hdf(i):
    return 'sampled_data_' + solver_id(i) + '.h5'


def prepare_sets_of_params(parameters, output_dir_in, n_processes, par_names):
    no_samples, no_parameters = np.shape(parameters)
    rows_per_file = no_samples // n_processes
    rem = no_samples % n_processes

    param_dir = os.path.join(output_dir_in, param_dirname)
    aux_functions.force_mkdir(param_dir, force=True)
    empty_hdf_dir = os.path.join(output_dir_in, empty_hdf_dirname)
    aux_functions.force_mkdir(empty_hdf_dir, force=True)

    sample_idx = 0
    off_start = 0
    off_end = 0
    for i in range(n_processes):
        off_start = off_end
        off_end = off_end + rows_per_file
        # add sample while there is still remainder after rows_per_file division
        if rem > 0:
            off_end = off_end + 1
            rem = rem - 1
        subset_matrix = parameters[off_start:off_end, :]

        param_file = os.path.join(param_dir, "params_" + solver_id(i) + ".csv")
        with open(param_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['idx', *par_names])
            for row in subset_matrix:
                writer.writerow([sample_idx, *row])
                sample_idx = sample_idx+1

        # Prepare HDF, write parameters
        output_file = os.path.join(empty_hdf_dir, sampled_data_hdf(i))
        n_params = parameters.shape[0]
        n_times = config_dict["sample_shape"][0]
        n_elements = config_dict["sample_shape"][1]
        sample_storage.create_chunked_dataset(output_file,
                                              shape=(n_params, n_times, n_elements),
                                              chunks=(1, 1, n_elements))
        sample_storage.append_new_dataset(output_file, "parameters", parameters)

    # for i, mat in enumerate(sub_parameters):
    #     output_file = f"parameters_{str(i+1).zfill(2)}.npy"
    #     np.save(output_file, mat)
    #     print(f"Saved {output_file}")



def conductivity(k0, eps, delta, gamma, sigma0, a, b, c, sigma_m):
    sigma_vm = 170/120*np.abs(sigma_m)
    sigma_tres = 55e6
    x = 0
    y = 30
    z = 30
    lin = (1 + 0.1*(a * x / 35 + b * y / 30 + c * z / 30))
    kr = 1/eps *k0
    k = kr + delta * np.exp(np.log((k0 - kr) / delta) * sigma_m / sigma0)
    return np.where(sigma_vm < sigma_tres, k, k * np.exp(gamma * (sigma_vm - sigma_tres)/sigma_tres)) * lin
    # if sigma_vm < sigma_tres:
    #     return k
    # else:
    #     return k * np.exp(gamma*(sigma_vm - sigma_tres))


def plot_conductivity(params):
    import matplotlib.pyplot as plt

    # init_sigma_m = -(42 + 19 + 14)*1e6/3
    sigma_mean = np.linspace(-120e6,7e6,200)
    for p in params:
        # if p[4] >= p[5]:
        #     continue
        # p[[4,5]] = p[[5,4]]
        init_sigma_m = -(p[1] + p[2] + p[3])/3
        cond = conductivity(p[4],p[5],p[6],p[7],init_sigma_m,p[8],p[9],p[10],sigma_mean)
        plt.plot(sigma_mean/1e6, cond)
        plt.scatter(init_sigma_m/1e6,p[4])

    plt.yscale('log')
    plt.savefig('conductivity.pdf')
    plt.close()

    plt.scatter(range(params.shape[0]), params[:,5])
    plt.savefig('eps.pdf')


if __name__ == "__main__":

    # default parameters
    output_dir = None

    len_argv = len(sys.argv)
    assert len_argv > 1, "Specify output dir."
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])

    # aux_functions.force_mkdir(output_dir, force=True)
    # shutil.copyfile("../test_data/config_sim_A04hm_V1_03.yaml", os.path.join(output_dir, "config.yaml"))
    # setup paths and directories
    config_dict = bayes_run_all.setup(output_dir, can_overwrite=False, clean=False)
    # add repository dir
    config_dict["rep_dir"] = os.path.abspath(os.path.join(config_dict["script_dir"], "../../.."))

    # Define the problem for SALib
    # Bayes Inversion borehole_V1/sim_A04hm_V1_04_20230713a
    params = config_dict["parameters"]
    problem = {
        'num_vars': len(params),
        'names': [p["name"] for p in params],
        'dists': [p["type"] for p in params],
        # available distributions:
        # unif - interval given by bounds
        # logunif,
        # triang - [lower_bound, upper_bound, mode_fraction]
        # norm,  bounds : [mean, std]
        # truncnorm, bounds : [lower_bound, upper_bound, mean, std_dev]
        # lognorm, bounds: [mean, std]  # mean and std of the log(X)
        'bounds': [p["bounds"] for p in params]
    }

    print(problem)

    # Generate Saltelli samples
    param_values = sample.saltelli(problem, config_dict["n_samples"], calc_second_order=config_dict["second_order_sa"])
    # param_values = sample.sobol(problem, n_samples, calc_second_order=True)
    print(param_values.shape)

    # plot_conductivity(param_values)
    # exit(0)

    sensitivity_dir = os.path.join(output_dir, sensitivity_dirname)
    aux_functions.force_mkdir(sensitivity_dir, force=True)

    # plan sample parameters a prepare them in CSV
    prepare_sets_of_params(param_values, sensitivity_dir, config_dict["n_processes"], problem["names"])
    # exit(0)

    # plan parallel sampling, prepare PBS jobs
    pbs_file_list = prepare_pbs_scripts(config_dict, sensitivity_dir, config_dict["n_processes"])

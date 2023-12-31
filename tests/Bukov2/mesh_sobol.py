"""
Collect HDF files from workers.

Usage:

python collect_hdf.py   workdir/<pattern>

e.g.
python collect_hdf.py   workdir/sampled_data_*.h5
"""
import sys
import logging

import h5py
import numpy as np
from pathlib import Path
from endorse import common
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse.Bukov2.bukov_common import memoize, file_result, load_cfg
from endorse.Bukov2 import sa_problem, sobol_fast
params_name="parameters"

#####################################š

@memoize
def chracterize_samples(workdir, in_dset):
    n_samples, n_times, n_els = in_dset.shape
    s_max = np.empty(n_samples)
    s_min = np.empty(n_samples)
    stride = 8
    for i_sample in range(0, n_samples, stride):
        print("  i_sample: ", i_sample)
        step = min(stride, n_samples - i_sample)
        frame = slice(i_sample, i_sample + step)
        sample = np.array(in_dset[frame, :, :])
        s_max[frame] = np.max(sample, axis=(1,2))
        s_min[frame] = np.min(sample, axis=(1,2))
    return s_max, s_min

@file_result("mesh_sobol_st.h5")
def mesh_sobol_st(workdir, out_file, in_file):
    workdir, cfg = load_cfg(workdir / "Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    n_params = problem['num_vars']
    with h5py.File(workdir / in_file, mode='r') as in_f:
        in_dset = in_f[dataset_name]
        n_samples, n_times, n_els = in_dset.shape
        with h5py.File(workdir / out_file, mode='w') as out_f:
            out_sobol_t = out_f.create_dataset('sobol_indices', (n_times, n_els, n_params), dtype='float64')
            out_mean = out_f.create_dataset('mean', (n_times, n_els), dtype='float64')
            out_std = out_f.create_dataset('std', (n_times, n_els), dtype='float64')
            out_max_sample = out_f.create_dataset('max_sample', (n_times, n_els), dtype='float64')
            out_med_sample = out_f.create_dataset('med_sample', (n_times, n_els), dtype='float64')
            
            print("Characterize ...")
            s_max, s_min = chracterize_samples(workdir, in_dset)
            samples_sorted = np.argsort(s_max - s_min)
            out_max_sample[:,:]  = in_dset[samples_sorted[-1], :, :]
            out_med_sample[:, :] = in_dset[samples_sorted[n_samples//2], :, :]
            for i_time in range(n_times):
                print("  sobol time: ", i_time)
                
                time_frame = np.empty((in_dset.shape[0], in_dset.shape[2]))
                time_frame[:, :] = in_dset[:, i_time, :]
                print("time frame: ", time_frame.shape)
                sobol_samples = time_frame.transpose([1,0])
                sobol = sobol_fast.vec_sobol_total_only(sobol_samples, problem)
                print("sobol: ", sobol.shape)
                out_sobol_t[i_time, :, :] = sobol
                out_mean[i_time, :] = np.mean(time_frame, axis=0)
                out_std[i_time, :] = np.std(time_frame, axis=0)
        return out_file





def main(workdir):
    in_file = "sampled_fixed.h5"
    mesh_sobol_st(workdir, in_file)


if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    main(workdir)

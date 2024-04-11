"""
Using collected samples in a merged HDF5 file, compute Sobol indices for all elements in all times.

Usage: 
python3 sobol_data.py <workdir> [<force>]

workdir     - directory with "Bukov2_mesh.yaml" main config file
force       - 0 = (default) append intermediate result files, 1 = delete intermediate results:
   workdir / "characterize_samples.pkl"
   workdir / "mesh_sobol_st.h5"
   workdir / "sensitivity" directory with resulting VTU files
"""

import sys
import shutil

import h5py
import numpy as np
from pathlib import Path
import pyvista as pv

from endorse import common
from endorse.Bukov2.sample_storage import dataset_name
from endorse.Bukov2.bukov_common import memoize, file_result, load_cfg, soft_lim_pressure
from endorse.Bukov2 import sa_problem
from endorse.common import sobol_fast


#params_name = "parameters"



#####################################š

def get_shape(workdir, fname):
    with h5py.File(workdir / fname, mode='r') as in_f:
        return in_f[dataset_name].shape

def get_in_slice(workdir, in_file, in_slice):
    with h5py.File(workdir / in_file, mode='r') as in_f:
        in_dset = in_f[dataset_name]
        pressure = np.array(in_dset[in_slice])
    return soft_lim_pressure(pressure)

def read_time_frame(workdir, in_file, i_time):
    time_frame =  get_in_slice(workdir, in_file, (slice(None), i_time, slice(None)))
    print("time frame: ", time_frame.shape)
    return time_frame

@memoize
def characterize_samples(workdir, in_file):
    print("Characterize samples...")
    n_samples, n_times, n_els = get_shape(workdir, in_file)
    s_max = np.empty(n_samples)
    s_min = np.empty(n_samples)
    stride = 8
    for i_sample in range(0, n_samples, stride):
        print("  i_sample: ", i_sample)
        step = min(stride, n_samples - i_sample)
        frame = slice(i_sample, i_sample + step)
        sample = get_in_slice(workdir, in_file, (frame, slice(None), slice(None)))
        s_max[frame] = np.max(sample, axis=(1, 2))
        s_min[frame] = np.min(sample, axis=(1, 2))
    return s_max, s_min



"""
Call the function only if the given file does not exist.
"""
@file_result("mesh_sobol_st.h5")
def mesh_sobol_st(workdir, out_file, in_file):
    workdir, cfg = load_cfg(workdir / "Bukov2_mesh.yaml")
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    n_params = problem['num_vars']
    param_names = problem['names']


    n_samples, n_times, n_els = get_shape(workdir, in_file)

    group_size = 2 * (n_params + 1)
    n_groups = n_samples // group_size
    noise = cfg.chambers.noise
    a_noise = noise * np.random.randn(n_groups)
    b_noise = noise * np.random.randn(n_groups)

    def sobol_with_noise(sobol_samples):
        """
        Add a noise to given sample and compute total sobol indices for remaining parameters.
        """
        ch_data = sobol_samples.reshape(-1, n_groups, group_size)
        group_size_new = 2 * (n_params + 1) + 2
        ch_data_with_noise = np.empty((ch_data.shape[0], n_groups, group_size_new))
        # A matrix eval
        ch_data_with_noise[:, :, 0] = ch_data[:, :, 0] + a_noise
        # AB matrix eval
        ch_data_with_noise[:, :, 1:n_params + 1] = ch_data[:, :, 1:n_params + 1] + a_noise[None, :, None]
        ch_data_with_noise[:, :, n_params + 1] = ch_data[:, :, 0] + b_noise
        # BA matrix eval
        ch_data_with_noise[:, :, n_params + 2:2 * n_params + 2] = ch_data[:, :,
                                                                  n_params + 1:2 * n_params + 1] + b_noise[None,
                                                                                                   :, None]
        ch_data_with_noise[:, :, 2 * n_params + 2] = ch_data[:, :, 2 * n_params + 1] + a_noise
        # B matrix eval
        ch_data_with_noise[:, :, 2 * n_params + 3] = ch_data[:, :, 2 * n_params + 1] + b_noise

        problem_loc = dict(problem)
        problem_loc['num_vars'] += 1

        sobol = sobol_fast.vec_sobol_total_only(ch_data_with_noise.reshape(n_els, -1), problem_loc)
        sobol = sobol[:, :-1, 0]
        return sobol

    # with h5py.File(workdir / out_file, mode='w') as out_f:
    #     out_mean = out_f.create_dataset('mean', (n_times, n_els), dtype='float64')
    #     out_std = out_f.create_dataset('std', (n_times, n_els), dtype='float64')
    #     out_max_sample = out_f.create_dataset('max_sample', (n_times, n_els), dtype='float64')
    #     out_med_sample = out_f.create_dataset('med_sample', (n_times, n_els), dtype='float64')
    #     out_sobol_t = out_f.create_dataset('sobol_indices', (n_times, n_els, n_params), dtype='float64')

    def write_group(name, data, i_time):
        with h5py.File(workdir / out_file, mode='a') as out_f:
            dset = out_f.require_dataset(name, (n_times, n_els), dtype='float64')
            dset[i_time, :] = data

    s_max, s_min = characterize_samples(workdir, in_file)
    samples_sorted = np.argsort(s_max - s_min)

    print("Compute Pressure fields ...")
    def write_quantile(name, q):
        i_q = samples_sorted[int(q * (n_samples - 1))]
        sample = get_in_slice(workdir, in_file, (i_q, slice(None), slice(None)))
        all_times = slice(None)
        write_group(name, sample, all_times)

    write_quantile('max_sample', 1.0)
    write_quantile('q90_sample', 0.9)
    write_quantile('med_sample', 0.5)

    print("Compute Sobol indices ...")
    for i_time in range(n_times):
        print("  sobol time: ", i_time)


        time_frame = read_time_frame(workdir, in_file, i_time)
        sobol_samples = time_frame.transpose([1, 0])
        sobol = sobol_with_noise(sobol_samples)

        print("   sobol: ", sobol.shape)
        write_group('mean', np.mean(time_frame, axis=0), i_time)
        write_group('std', np.std(time_frame, axis=0), i_time)
        assert sobol.shape[1] == len(param_names)
        for i, param in enumerate(param_names):
            write_group(param, sobol[:, i], i_time)

    return out_file



def read_sobol(workdir, sobol_file):
    """
    Read all datasets of a HDF file into a dict of np arrays.
    """
    out_dict ={}
    with h5py.File(workdir / sobol_file, mode='r') as in_f:
        for name, item in in_f.items():
            out_dict[name] = np.array(item)
    return out_dict


def create_vtu_sequence(workdir, field_dict, mesh_vtu, out_dir):
    """
    Create a sequence of VTU files with cell data for each time step.

    Parameters:
    mesh (pyvista mesh): The 3D tetrahedral mesh.
    cell_data (numpy array): Array of shape (n_times, n_cells, n_quantities).
    quantity_names (list): List of names for each quantity.
    output_dir (str): Directory to save VTU files.
    """
    mesh = pv.read(workdir / mesh_vtu)
    mesh.clear_data()
    out_dir = workdir / out_dir
    (out_dir).mkdir(parents=True, exist_ok=True)
    n_times = field_dict['mean'].shape[0]

    for t in range(n_times):
        # Add each quantity to the mesh
        for q, field in field_dict.items():
            mesh.cell_data[q] = field[t, :]

        # Save the mesh as a VTU file
        mesh.save(out_dir / f"time_step_{t:02d}.vtu")
    return out_dir





def main(workdir):
    cfg_file = workdir / "Bukov2_mesh.yaml"
    cfg = common.load_config(cfg_file)
    sim_cfg = common.load_config(workdir / cfg.simulation.cfg)
    problem = sa_problem.sa_dict(sim_cfg)
    in_file = cfg.simulation.hdf
    sobol_file = mesh_sobol_st(workdir, in_file)

    field_dict = read_sobol(workdir, sobol_file)
    out_dir = 'sensitivity'
    create_vtu_sequence(workdir, field_dict, cfg.simulation.mesh, out_dir)



if __name__ == '__main__':
    arg_dict = dict(enumerate(sys.argv))
    workdir = Path(arg_dict[1]).absolute()
    force = arg_dict.get(2, False)
    if force:
        (workdir / "characterize_samples.pkl").unlink(missing_ok=True)
        (workdir / "mesh_sobol_st.h5").unlink(missing_ok=True)
        sens_dir = (workdir / "sensitivity")
        shutil.rmtree(sens_dir, ignore_errors=True)
    main(workdir)

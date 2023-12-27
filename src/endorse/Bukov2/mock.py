import endorse.Bukov2.bukov_common as bcommon
from endorse.Bukov2 import sa_problem, boreholes, optimize_packers, sample_storage
from endorse.sa import analyze, sample
from endorse import common
import numpy as np
import h5py
import json

def mock_hdf5(cfg_file):
    workdir, cfg = bcommon.load_cfg(cfg_file)
    hdf_path = workdir / cfg.simulation.hdf
    if not cfg.boreholes.force and hdf_path.exists():
        with h5py.File(hdf_path, 'r') as f:
            return hdf_path, f[sample_storage.dataset_name].shape

    pattern = workdir / 'flow_reduced' / 'flow_*.vtu'
    pressure_array, mesh = boreholes.get_time_field(str(pattern), 'pressure_p0')

    # Fictious model
    sa_dict = sa_problem.sa_dict(common.config.load_config(workdir / cfg.simulation.cfg))
    param_samples = sample.saltelli(sa_dict, 16, sa_dict['second_order'])
    params_norm = param_samples / param_samples.std(axis=0)
    sigma_x = params_norm[:, 1]
    sigma_y = params_norm[:, 2]
    n_params = param_samples.shape[1]
    scale = (np.arange(n_params) + 1) / n_params * 2
    sum_other = np.sum(params_norm * scale[None, :], axis=1)

    # field_samples = sigma_x[:, None, None] * pressure_array[None, :, :] + \
    #                 (pressure_array[None, :, :] ** 2) / sigma_y[:, None, None] + sum_other[:, None, None]
    field_samples = pressure_array[None, :, :] + sum_other[:, None, None]

    shape = field_samples.shape
    with h5py.File(hdf_path, 'w') as f:
        # chunks=True ... automatic chunk size
        # f.create_dataset(dataset_name, data=np.zeros(shape), chunks=True, dtype='float64')
        dset = f.create_dataset(sample_storage.dataset_name, shape=field_samples.shape, chunks=True, dtype='float64')
        f.create_dataset(sample_storage.failed_ids_name, shape=(0, 1), maxshape=(shape[0], 1), dtype='int')
        dset[...] = field_samples[...]

    times = [0] + list(range(100, 109))
    with open(workdir / "output_times.json", "w") as f:
        json.dump(times, f)
    return hdf_path, shape
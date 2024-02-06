import os
import sys
import shutil
# import csv
import pandas as pd
import time
import ruamel.yaml as yaml
import numpy as np

from endorse import common
from endorse.Bukov2 import sample_storage, flow_wrapper


def setup(output_dir, can_overwrite, clean):
    # create and cd workdir
    rep_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = output_dir

    # Files in the directory are used by each simulation at that level
    common_files_dir = os.path.join(work_dir, "common_files")
    # Create working directory if necessary
    common.force_mkdir(common_files_dir, force=clean)
    os.chdir(work_dir)

    # test if config exists, copy from rep_dir if necessary
    config_file = os.path.join(work_dir, "config.yaml")
    if not os.path.exists(config_file):
        # to enable processing older results
        config_file = os.path.join(common_files_dir, "config.yaml")
        if not os.path.exists(config_file):
            raise Exception("Main configuration file 'config.yaml' not found in workdir.")
        else:
            import warnings
            warnings.warn("Main configuration file 'config.yaml' found in 'workdir/common_files'.",
                          category=DeprecationWarning)

    # read config file and setup paths
    with open(config_file, "r") as f:
        yaml_reader = yaml.YAML(typ='safe', pure=True)
        config_dict = yaml_reader.load(f)

    config_dict["work_dir"] = work_dir
    config_dict["script_dir"] = rep_dir

    config_dict["common_files_dir"] = common_files_dir
    # config_dict["bayes_config_file"] = os.path.join(common_files_dir,
    #                                                 config_dict["surrDAMH_parameters"]["config_file"])

    # copy common files
    for f in config_dict["copy_files"]:
        filepath = os.path.join(common_files_dir, f)
        if not os.path.isfile(filepath) or can_overwrite:
            shutil.copyfile(os.path.join(rep_dir, f), filepath)

    return config_dict

def just_run_flow123d(config_dict, measured_data, params_in, output_dir_in, solver_id):

    wrap = flow_wrapper.Wrapper(solver_id=solver_id, output_dir=output_dir_in, config_dict=config_dict)

    for pars in params_in:
        idx = int(pars[0])
        wrap.set_parameters(data_par=pars[1:])
        t = time.time()
        res, sample_data = wrap.get_observations()

        print("Flow123d res: ", res)
        #if res >= 0:
            #print(obs_data)
            #boreholes = config_dict["surrDAMH_parameters"]["observe_points"]
            #measured_data.plot_comparison(obs_data, wrap.sim.sample_dir, boreholes)

        # print("LEN:", len(obs_data))
        print("TIME:", time.time() - t)

        # # write output
        # if config_dict["sample_subdir"] is not None:
        #     output_file = os.path.join(config_dict["sample_subdir"], 'output_' + str(solver_id) + '.csv')
        # else:
        #     output_file = os.path.join(output_dir_in, 'output_' + str(solver_id) + '.csv')
        # with open(output_file, 'a') as file:
        #     line = str(idx) + ',' + ','.join([str(s) for s in obs_data])
        #     file.write(line + "\n")

        output_file = os.path.join(output_dir_in, 'sampled_data_' + str(solver_id) + '.h5')
        sample_storage.set_sample_data(output_file, sample_data, idx)

        # if idx == 1:
        #     exit(0)


def add_output_keys(config_dict):
    fname = config_dict["hm_params"]["in_file"]
    fname_output = fname + '_vtk'
    ftemplate = os.path.join(config_dict["common_files_dir"], fname + '_tmpl.yaml')
    ftemplate_output = os.path.join(config_dict["common_files_dir"], fname_output + '_tmpl.yaml')

    yaml_handler = yaml.YAML()
    with open(ftemplate, "r") as f:
        file_content = f.read()
    template = yaml_handler.load(file_content)

    flow_fields = [
       {"field": "conductivity", "interpolation": "P1_average"},
       "piezo_head_p0",
       "pressure_p0",
       "velocity_p0",
       "region_id"
    ]
    template["problem"]["flow_equation"]["flow_equation"]["output"]["fields"] = flow_fields

    mech_fields = [
        {"field": "displacement", "interpolation": "P1_average"},
        "stress",
        "displacement_divergence",
        "mean_stress",
        "von_mises_stress",
        "initial_stress",
        "region_id"
    ]
    template["problem"]["flow_equation"]["mechanics_equation"]["output"]["fields"] = mech_fields

    config_dict["hm_params"]["in_file"] = fname_output
    with open(ftemplate_output, 'w') as f:
        yaml_handler.dump(template, f)


if __name__ == "__main__":

    # default parameters
    output_dir = None
    csv_data = None
    n_best_params = 0
    sample_subdir = None
    solver_id = 0

    len_argv = len(sys.argv)
    assert len_argv > 2, "Specify output dir and parameters in csv file!"
    if len_argv > 1:
        output_dir = os.path.abspath(sys.argv[1])
    if len_argv > 2:
        file = sys.argv[2]
        if os.path.exists(file):
            csv_data = os.path.abspath(sys.argv[2])
        else:
            n_best_params = int(sys.argv[2])
    if len_argv > 3:
        if os.path.isabs(sys.argv[3]):
            sample_subdir = sys.argv[3]
        else:
            sample_subdir = os.path.join(output_dir, sys.argv[3])
    if len_argv > 4:
        solver_id = sys.argv[4]

    # setup paths and directories
    config_dict = setup(output_dir, can_overwrite=False, clean=False)
    if "vtk_output" in config_dict and config_dict["vtk_output"]:
        add_output_keys(config_dict)
    if sample_subdir is not None:
        config_dict["sample_subdir"] = sample_subdir

    # preprocess(config_dict)

    # prepare measured data as observations
    md = None
    # md = MeasuredData(config_dict)
    # md.initialize()

    # boreholes = config_dict["surrDAMH_parameters"]["observe_points"]
    # times, values = md.generate_measured_samples(boreholes, [])

    if csv_data:
        print("Reading parameters from CSV: ", csv_data)
        pd_samples = pd.read_csv(csv_data, header=0)
        parameters = np.array(pd_samples.iloc[:,:])
        #with open(csv_data, newline='') as csvfile:
            #csvreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            #next(csvreader)
            #parameters = list(csvreader)
    else:
        # print("Getting " + str(n_best_params) + " best parameters.")
        # parameters = get_best_accepted_params(config_dict, output_dir, n_best_params)
        exit(1)

    # print(parameters)

    # print(boreholes)
    # JUST RUN FLOW123D FOR TESTING
    just_run_flow123d(config_dict, md, parameters, output_dir, solver_id)

"""
Usage:

python borehole_data.py workdir [from_sample]

TODO: this use lot of memory and must be executed on a single allocated node, make this process part of the script similar to the process_boreholes.py

Prepare sampled pressure data on individual boreholes.
Write to separate files in order to paralelize the chamber optimization.
Slice paralelization of this process is possible but it is bound by the IO of
the disk storage.

We need dask for better performance when the large computations would be split into smaller pieces
allowing better overleap of calculation and communication and also keeping files distributed.
"""
import sys
import logging
import subprocess

#import h5py
#import numpy as np
from pathlib import Path
from endorse.Bukov2.sample_storage import dataset_name, failed_ids_name, done_ids_name
from endorse.common import load_config
from endorse.Bukov2 import sa_problem, boreholes
from endorse.Bukov2.bukov_common import memoize, file_result, load_cfg
params_name="parameters"
from pathlib import Path
script_path = Path(__file__).absolute()


def main(workdir, from_sample):
    cfg_file = workdir / "Bukov2_mesh.yaml"
    workdir, cfg = load_cfg(cfg_file)
    bh_set = boreholes.make_borehole_set(workdir, cfg)
    bh_field = boreholes.project_field(workdir, cfg, bh_set, from_sample=from_sample)
    print("Updated: ", bh_field.data_files)


pbs_script_template = """
#!/bin/bash
#PBS -j oe
#PBS -m e
set -x 
env | grep PBS_
umask 0007
echo "===="
{python} {script_path} {workdir} {args}
"""
def submit_pbs(workdir, args):
    workdir, cfg = load_cfg(workdir / "Bukov2_mesh.yaml")    
    queue = cfg.pbs.queue
    pbs_filename = workdir / "borehole_data.pbs"
    parameters = dict(
        python=sys.executable,
        script_path=script_path,
        workdir=workdir,
        args=" ".join(args)
    )
    print(parameters['python'])
    pbs_script = pbs_script_template.format(**parameters)
    with open(pbs_filename, "w") as f:
        f.write(pbs_script)
    
    
    cmd = ['qsub', '-q', queue, '-l', f'select=1:ncpus=1:mem=90gb', pbs_filename]
    subprocess.run(cmd, check=True)


# TODO: use hyperque:
# - can interact from python
# - can wait for completed tasks
# - simple task workflow with explicit task dependency is clearly possible
# Could be a topic for a BP




if __name__ == '__main__':
    workdir = Path(sys.argv[1]).absolute()
    print(sys.argv)
    if len(sys.argv) > 2 and (sys.argv[2] == 'submit'):
        # submit the job
        print('submit')
        submit_pbs(workdir, sys.argv[3:])
    else:
        if len(sys.argv) > 2:
            from_sample = int(sys.argv[2])
        else:
            from_sample = 0

        # run the borhehole data preparation.
        main(workdir, from_sample)

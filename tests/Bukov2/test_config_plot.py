from endorse.Bukov2 import optimize_configs as oc
from pathlib import Path
script_dir = Path(__file__).absolute().parent


# borehole ID must be defined in subset in the Bukov2_mesh.yaml file
# indices * 0.5 m = offset of chamber centers from borehole head

# boreholes around ZK5-1J
"""
configs = \
[[
    ('P +5-25+15', [4,11,16,24]),
    ('P-13+66+12', [6, 13,19,26]),  # OK
    ('P-13+59-12', [16,21,28,34]),  # OK
    ('P-13+70-15', [4,11,17,26]),   # OK
    ('P -6+13+00', [4,10,20,26]),   # OK
    ('P +5-09+00', [10,18, 25, 31])  # OK

],
]
"""

configs = \
[[
    #('P +5-25+15', [3,9,14,24]),
    #('P-13+66+12', [6, 13,19,26]),  # OK
    #('P-13+59-12', [16,21,28,34]),
    # ('P-13+70-15', [0,6,12,18,24,30,36,42,48,54,60,66]), #('P-13+70-15', [4,10,16,26]),
#    ('P-13+70-15', range(2, 70, 2)),
    ('L-11+67-17', [4,9,14]),
    ('L -9+40+14', [5,12,18]),

    #('P -6+13-07', [11,18,24,30]),
#    ('P +5-09+00', [18, 26, 32, 38])
],
]
cfg_file = script_dir / "PE_01_02_zk40/Bukov2_mesh.yaml"


for i, cfg in enumerate(configs):
    oc.export_vtk_bh_chamber_set(cfg_file, cfg, f"boreholes_opt_cfg_{i}.vtk", plot=False)

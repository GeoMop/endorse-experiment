from endorse.Bukov2 import optimize_configs as oc
from pathlib import Path
script_dir = Path(__file__).absolute().parent

# borholes around ZK5-1J
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

# borholes around ZK5-1S
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


cfg_file = script_dir / "3d_model/PE_01_02/Bukov2_mesh.yaml"
for i, cfg in enumerate(configs):
    oc.export_vtk_bh_chamber_set(cfg_file, cfg, f"boreholes_opt_cfg_{i}.vtk", plot=False)

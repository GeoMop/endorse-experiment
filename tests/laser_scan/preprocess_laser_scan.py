import os, sys
import yaml
import numpy as np
from typing import Tuple, List
from endorse import common

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh
import gmsh as gmsh_api

script_dir = os.path.dirname(os.path.realpath(__file__))

def make_gmsh(cfg:'dotdict'):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """



    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    tunnel_laser_scan = factory.merge(boundary_mesh_healed_filename)

    print("gmsh_api.model.getEntities():\n", gmsh_api.model.getEntities())
    print(tunnel_laser_scan.dim_tags)
    print(tunnel_laser_scan.regions)
    print("factory.model.getEntities():\n", factory.model.getEntities())

    tunnel_boundary = tunnel_laser_scan.split_by_dimension()[2]
    # gmsh_api.model.mesh.createGeometry(tunnel_boundary.dim_tags)

    # factory.synchronize()
    # factory.write_brep()

    factory.synchronize()

    # exit(0)

    print("factory.model.getEntities():\n", factory.model.getEntities())
    geometry_set = tunnel_boundary
    # geometry_set = make_geometry(factory, cfg.geometry, cfg.mesh, tunnel_boundary)

    # print("gmsh_api.model.getEntities():\n", gmsh_api.model.getEntities())
    # print(tunnel_laser_scan.dim_tags)
    # print(tunnel_laser_scan.regions)

    # tunnel_boundary = tunnel_laser_scan.split_by_dimension()[2]
    # print("factory.model.getEntities():\n", factory.model.getEntities())
    # for i in range(2,25):
    #     tunnel_boundary.drop((2, i))
    #     gmsh_api.model.remove_entities(dimTags=[(2, i)], recursive=True)
    #     # gmsh_api.model.occ.remove(dimTags=[(2, i)], recursive=True)   # not working, mesh entities not in OCC
    # for i in range(33,66):
    #     gmsh_api.model.remove_entities(dimTags=[(1, i)], recursive=True)
    # for i in range(1, 25):
    #     gmsh_api.model.remove_entities(dimTags=[(0, i)], recursive=True)
    # print("gmsh_api.model.getEntities():\n", gmsh_api.model.getEntities())
    # gmsh_api.model.mesh.createGeometry()

    # print("factory.model.getEntities():\n", factory.model.getEntities())
    # tunnel_boundary.drop((2,12))
    # tunnel_boundary.drop((2,16))
    # factory.keep_only(tunnel_boundary)
    # gmsh_api.model.mesh.createGeometry(tunnel_boundary.dim_tags)
    # print("factory.model.getEntities():\n", factory.model.getEntities())
    # print("gmsh_api.model.getEntities():\n", gmsh_api.model.getEntities())
    # geometry_set = make_geometry(factory, cfg.geometry, cfg.mesh, tunnel_boundary)

    # geometry_set = make_geometry(factory, cfg.geometry, cfg.mesh, tunnel_laser_scan)

    # geometry_set = make_geometry(factory, cfg.geometry, cfg.mesh, None)

    factory.show()
    meshing(factory, [geometry_set], final_mesh_filename)
    # meshing(factory, [tunnel_boundary], final_mesh_filename)
    # factory.show()
    del factory
    return common.File(final_mesh_filename)


def make_mesh(workdir, output_dir, cfg_file):
    conf_file = os.path.join(workdir, cfg_file)
    cfg = common.config.load_config(conf_file)
    cfg.output_dir = output_dir

    boundary_mesh_filename = os.path.join(cfg.output_dir, cfg.boundary_meshfile)

    # heal mesh
    boundary_mesh_healed_filename = os.path.join(cfg.output_dir, cfg.mesh_name + "_healed.msh2")
    if not os.path.exists(boundary_mesh_healed_filename):
        print("HEAL MESH")
        hm = heal_mesh.HealMesh.read_mesh(boundary_mesh_filename, node_tol=1e-4)
        hm.heal_mesh(gamma_tol=0.02)
        # hm.stats_to_yaml(os.path.join(output_dir, cfg.mesh_name + "_heal_stats.yaml"))
        hm.write(file_name=boundary_mesh_healed_filename)

    # the number of elements written by factory logger does not correspond to actual count
    # reader = gmsh_io.GmshIO(mesh_file.path)
    # print("N Elements: ", len(reader.elements))

    # print("Mesh file: ", mesh_file)


if __name__ == '__main__':
    # output_dir = None
    # len_argv = len(sys.argv)
    # assert len_argv > 1, "Specify input yaml file and output dir!"
    # if len_argv == 2:
    #     output_dir = os.path.abspath(sys.argv[1])
    output_dir = script_dir

    make_mesh(script_dir, output_dir, "./l5_mesh_config.yaml")


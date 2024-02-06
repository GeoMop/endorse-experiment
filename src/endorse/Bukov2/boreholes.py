from typing import  *
import glob
import attrs
from functools import cached_property
import itertools
import pathlib

import h5py
import numpy as np
import pyvista as pv
import json
from endorse.Bukov2 import sample_storage
from endorse.Bukov2 import bukov_common as bcommon, plot_boreholes
from vtk import vtkCellLocator

"""
TODO:
1. Test problem genetic algorithm on cluster.
2. Single individual : N borehole ids, 4*N plug position ids
   ZK30 - N=6, 6 + 24 = 30 ints
3. validation of individual - check borehole interactions - for every pair compute transwersal, it must be > 1m 
4. Mutation : change with some probablility in every borehole group , change the plug position by certain step (-3 up to +3) 
5. Mating : one of borehole setups in every independent group; theta convex interpolation between plug positions, should preserve ordering
6. Evalutaion : 
"""

Point3d = Tuple[float, float, float]
Box = Tuple[Point3d, Point3d]

def direction_vector(y_phi, z_phi):
    y_phi = y_phi / 180 * np.pi
    z_phi = z_phi / 180 * np.pi
    sy, cy = np.sin(y_phi), np.cos(y_phi)
    sz, cz = np.sin(z_phi), np.cos(z_phi)
    return np.array([cy * cz, sy * cz, sz])


def direction_angles(unit_direction):
    z_angle = np.arcsin(unit_direction[2])
    y_angle = np.arcsin(unit_direction[1] / np.cos(z_angle))
    return 180 * y_angle / np.pi, 180 * z_angle / np.pi


@bcommon.memoize
def _make_borehole_set(workdir, cfg) -> 'BoreholeSet':
    # Prepare BoreholeSet with simulation data loaded
    workdir, cfg = bcommon.load_cfg(workdir / "Bukov2_mesh.yaml")
    cfg_zk = cfg.boreholes.active_zk
    lateral = Lateral.from_cfg(cfg_zk)
    def promote_to_list(l):
        if not isinstance(l, list):
            l = [l]
        return l

    def lines_group(start_x, start_y, start_z, end_x, end_y, end_z, group):
        ptl = promote_to_list
        parm_lists = map(ptl, [start_x,start_y, start_z, end_x, end_y, end_z])
        return [([sx, sy, sz], [ex,ey,ez], 10, group) for sx,sy,sz,ex,ey,ez in itertools.product(
            *parm_lists)
        ]

    def union(*lists):
        return [x for l in lists for x in l]

    lines = [lines_group(**line_space, group=group) for group, line_space in cfg_zk.lines_dict.items()]
    l_set = union(*lines)

    bh_set = lateral.set_from_points(l_set)
    print("N all boreholes:", bh_set.n_boreholes)
    bh_set.boreholes_print_sorted(f_name= workdir / "all_boreholes_list.txt")

    if cfg_zk.get("subset", []):
        bh_set = bh_set.subset_by_ids(cfg_zk.subset)
        print("N subset boreholes:", bh_set.n_boreholes)
        bh_set.boreholes_print_sorted(f_name= workdir / "subset_boreholes_list.txt")

    plot_boreholes.export_vtk_bh_set(workdir, bh_set)

    return bh_set

def make_borehole_set(workdir, cfg) -> 'BoreholeSet':
    return _make_borehole_set(workdir, cfg, force=cfg.boreholes.force)



@attrs.define
class Lateral:
    side: str
    origin_stationing: float
    galery_width: float
    l5_azimuth: float
    avoid_cylinder: Tuple[float, float, float]     # Boreholes can not insterset this cylinder (r, length_min, length_max) from origin in X direction
    active_cylinder: Tuple[float, float, float]    # Boreholes must intersect this cylinder (r, length_min, length_max) from origin in X direction
    transform_matrix: np.ndarray    # Mapping from lateral system to main system
    transform_shift: np.array       # position of the lateral system origin
    foliation_latitude: float       # latitude of foliation plane
    foliation_longitude: float      # longitude of foliation plane
    foliation_angle_tolerance: float # tolerance for borehole deviation from foliation angle

    @classmethod
    def from_cfg(cls, cfg):
        if cfg.invert_xy:
            side = 'L'
            transform_matrix = np.diag([-1, -1, 1])
        else:
            side = 'P'
            transform_matrix = np.eye(3)
        return cls(
            side,
            cfg.common.origin_stationing,
            cfg.common.galery_width,
            cfg.common.l5_azimuth,
            cfg.avoid_cylinder,
            cfg.active_cylinder,
            transform_matrix,
            np.array(cfg.transform_shift),
            cfg.common.foliation_latitude,
            cfg.common.foliation_longitude,
            cfg.common.foliation_angle_tolerance
        )

    @property
    def cylinder_line(self):
        cyl_max_l = self.avoid_cylinder[2]
        return np.array([0,0,0,cyl_max_l,0,0])

    # transform points from lateral coordinate system to model coordinate system
    def transform(self, points):
        points = np.array(points)[..., None]
        new_points = self.transform_matrix @ points + self.transform_shift[:,None]
        return new_points[..., 0]

    def stationing(self, y_pos):
        return y_pos + self.origin_stationing

    @staticmethod
    def cyl_line_intersect(cyl, line):
        r, l0, l1 = cyl
        pos, dir = line
        a = np.sum(dir[1:] ** 2)
        b = 2 * np.dot(pos[1:], dir[1:])
        c = np.sum(pos[1:] ** 2) - r ** 2

        # Calculate the roots of the quadratic equation
        roots = np.roots([a, b, c])

        # Check for real roots
        real_roots = roots[np.isreal(roots)].real

        # Check if the x-coordinate at any real root is within the bounds of the cylinder
        if len(real_roots) == 2:
            t0, t1 = real_roots
            t = (min(t0,t1), max(t0,t1))
            if max(t[0], l0) <= min(t[1], l1):
                x = np.maximum(l0, np.minimum(l1, pos[0] + np.array(t) * dir[0]))
                t_projected = (x - pos[0]) / dir[0]
                return t_projected

        return []

    def is_active(self, points):
        """
        Returns true for points that are inside the active cylinder.
        :param points:
        :return:
        """

        r, l0, l1 = self.active_cylinder
        points = np.atleast_2d(points)
        in_tubus = np.linalg.norm(points[:,:2], axis=1) < r
        return (points[:, 0] > l0) & (points[:, 0] < l1) & in_tubus

    def azimuth(self, bh_dir):
        global_vec = self.transform([[0,0,0], bh_dir.tolist()])
        global_dir_xy = (global_vec[1] - global_vec[0])[:2]
        return (np.degrees(np.arctan2(*global_dir_xy)) + self.l5_azimuth) % 360

    @staticmethod
    def transversal_params(line1, line2):
        a1, d1 = line1[:3], line1[3:]
        a2, d2 = np.array(line2[:3]), np.array(line2[3:])

        # Build the orthogonal coordinate system
        ey = d1 / np.linalg.norm(d1)  # cylinder direction
        ex = np.cross(d1, d2)  # transverzal direction, perpendicular to both lines
        norm_ex = np.linalg.norm(ex)
        assert not np.isnan(norm_ex)
        if norm_ex < 1e-12:
            # parallel borehole, skew slightly in Z direction, prescribe transversal point in the cylinder middle
            adiff_ey = np.dot((a2 - a1), ey)
            # adiff perpendicular to cylinder line1
            tmp_ex = a2 - a1 - adiff_ey * ey
            tmp_ez = np.cross(tmp_ex, ey)
            tmp_norm_ez = tmp_ez / np.linalg.norm(tmp_ez)
            mid_point = a2 + 0.5 * d1
            a2 -= 0.1 * tmp_norm_ez
            # t param according to mid_point X coordinate
            d2 = mid_point - a2

            ex = np.cross(d1, d2)  # transverzal direction, perpendicular to both lines
            norm_ex = np.linalg.norm(ex)
            assert norm_ex > 1e-12

        ex_normalized = ex / norm_ex
        ez = np.cross(d1, ex)  # tangent to cylinder
        ez_normalized = ez / np.linalg.norm(ez)

        diff = a2 - a1
        # Project (a2 - a1) onto ez to find t
        t = -np.dot(diff, ez_normalized) / np.dot(d2, ez_normalized)
        # ty = np.dot(diff, ey)
        point_2 = a2 + t * d2

        # Calculate s
        s = np.dot(point_2 - a1, ey) / np.dot(d1, ey)
        point_1 = a1 + s * d1

        length = np.abs(np.dot(diff, ex_normalized))
        point_dist = np.linalg.norm(point_1 - point_2)
        assert np.abs(length - point_dist) < 1e-5
        return length, s, t, point_1, point_2, ez_normalized

    def _filter_line(self, start, end_point, add_length=0.0):
        direction = end_point - start
        unit_direction = direction / np.linalg.norm(direction)
        length, cyl_t, bh_t, cyl_point, bh_point, yz_tangent = self.transversal_params(self.cylinder_line, np.array(
            [*start, *unit_direction]))

        # transversal on oposite direction
        # Some error in transversal calculation, TODO: simplify to YZ projection
        #if cyl_t < 0:
        #    return None

        # well head too close to lateral
        t_head = (self.galery_width / 2.0 - start[0]) / unit_direction[0]
        head = start + t_head * unit_direction
        if (-5 < head[1] < 4.7) or (5.1 < head[1] < 15):
            return None

        intersection =  self.cyl_line_intersect(self.avoid_cylinder, (start, unit_direction))
        if len(intersection) > 0:
            return None

        intersection =  self.cyl_line_intersect(self.active_cylinder, (start, unit_direction))
        if len(intersection) != 2:
            return None
        t_bounds = intersection
        bh_dir = unit_direction

        # deviation from foliation is below tolerance
        if self.foliation_angle_tolerance < 90:
            dir_model = self.transform(end_point) - self.transform(start)
            unit_dir_model = dir_model / np.linalg.norm(dir_model)
            foliation_dir_model = direction_vector(-self.foliation_longitude+self.l5_azimuth+90, self.foliation_latitude)
            if abs(np.dot(unit_dir_model, foliation_dir_model)) < np.cos(np.radians(self.foliation_angle_tolerance)):
                return None

        # Fix length
        bh_length = np.linalg.norm(direction) + add_length
        # print(f"({y_phi}, {z_phi}) dir: {dir} {bh_t} bh: {bh_dir} dot: {np.dot(dir, bh_dir)}")
        return start, bh_dir, bh_point, bh_length, t_bounds


    def bh_from_angle(self, start, angles, length=30, group=""):
        start = np.array(start)
        end = (start + length * direction_vector(*angles))
        line_points = self._filter_line(start, end)
        return self._make_bh(line_points, group)

    def bh_from_points(self, start, end, add_length=0.0, group=""):
        start = np.array(start)
        end = np.array(end)
        line_points = self._filter_line(start, end, add_length)
        return self._make_bh(line_points, group)

    def _make_bh(self, points, group):
        if points is None:
            return None
        else:
            return Borehole(self, *points, group)

    def set_from_points(self, lines):
        bh_list = [self.bh_from_points(*line) for line in lines]
        return self._make_bh_set(bh_list)

    def set_from_cfg(self, bh_set_cfg):
        cfg = bh_set_cfg
        y_angles = self._angle_array(cfg.y_angles)
        z_angles = self._angle_array(cfg.z_angles)

        ny, nz = len(y_angles), len(z_angles)
        bh_dir_table = [[[] for j in range(nz)] for i in range(ny)]
        bh_list = []
        yz_phi_range = list(itertools.product(y_angles, z_angles))
        for zone in np.array(cfg.wh_zones):
            ranges = [self.linspace(*zone[:, idim], cfg.wh_pos_step[idim]) for idim in [0, 1, 2]]
            xyz_range = itertools.product(*ranges)
            for pos in xyz_range:
                for yz_angle in yz_phi_range:
                    length = cfg.common.max_bh_length
                    bh_list.append(self.bh_from_angle(pos, yz_angle, length))

        return self._make_bh_set(bh_list)

    def _make_bh_set(self, bh_list):
        boreholes = [bh for bh in bh_list if bh is not None]
        indices = list(range(len(boreholes)))
        return BoreholeSet(boreholes, self, orig_indices=indices)

    @staticmethod
    def _angle_array(lst):
        if lst[0] == 0.0:
            # Make symmetric.
            lst = [-x for x in lst[1:]] + lst
        return np.array(lst)

    @staticmethod
    def linspace(a, b, step):
        a, b =  (a, b) if a < b else (b, a)
        n = int((b - a) / step)
        if n < 1:
            return np.array([(a+b) / 2])
        else:
            return np.linspace(a, b, n + 1)


@attrs.define(slots=False)
class Borehole:
    lateral: Lateral                    # Local coordinate system of lateral
    start: np.ndarray                   # start position of the well at drilling machine
    unit_direction: np.ndarray          # line direction
    transversal: np.ndarray             # endpoint of transversal to the lateral tunnel avoid cylinder axis
    length: float                       # length of the borehole from start point
    bounds: Tuple[float, float]                       # Intersection parameters with active cylinder.
    group : str                         # label of setup group

    # Moved to bcommon.direction_*
    # @staticmethod
    # def _direction(y_phi, z_phi):
    #     y_phi = y_phi / 180 * np.pi
    #     z_phi = z_phi / 180 * np.pi
    #     sy, cy = np.sin(y_phi), np.cos(y_phi)
    #     sz, cz = np.sin(z_phi), np.cos(z_phi)
    #     return np.array([cy * cz, sy * cz, sz])
    #
    # @staticmethod
    # def _angles(unit_direction):
    #     z_angle = np.arcsin(unit_direction[2])
    #     y_angle = np.arcsin(unit_direction[1] / np.cos(z_angle))
    #     return 180 * y_angle / np.pi, 180 * z_angle / np.pi

    @property
    def stationing(self):
        return self.lateral.stationing(self.well_head[1])

    @property
    def length_from_wall(self):
        return self.length - np.linalg.norm(self.well_head - self.start)

    @property
    def id(self):
        pos = f"{int(self.start[1]):+3d}"
        ya, za  = [f"{int(a):+03d}" for a in self.yz_angles]
        return f"{self.lateral.side}{pos}{ya}{za}"

    @property
    def t_well_head(self):
        t = (self.lateral.galery_width / 2.0 -self.start[0]) / self.unit_direction[0]
        return t

    @property
    def well_head(self):
        return self.line_point(self.t_well_head)

    @property
    def end_point(self):
        return self.start + self.length * self.unit_direction

    @property
    def yz_angles(self) -> Tuple[float, float]:
        # relative azimuth (y angle) and elevation (Z angle)
        return direction_angles(self.unit_direction)


    def __getstate__(self):
        state = self.__dict__.copy()
        # Optionally remove the cached_property data if not needed
        #state.pop('expensive_computation', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def line_point(self, t):
        return self.start + t * self.unit_direction

    def line_points(self, t):
        t = np.atleast_1d(t)
        return self.start + t[:, None] * self.unit_direction


    @property
    def t_transversal(self):
        # Assume boreholes can not be perpendicular to the lateral cylinder.
        return self.transversal[0] / self.unit_direction[0]


    @property
    def bh_description(self):
        pos = self.start
        dir = self.unit_direction
        azimuth = self.lateral.azimuth(dir)
        y_angle, z_angle = self.yz_angles
        x, y, z = self.lateral.transform(self.well_head)
        assert np.isclose(abs(x), 2.0)
        sy = self.lateral.stationing(y)
        pos_str =f"[{pos[0]:4.2f}, {pos[1]:4.2f}, {pos[2]:4.2f}],\n    point on wall: [ stationing: {sy:6.2f} m, height: {z:6.2f} m]"
        angle_str = f"(azimuth: {azimuth:4.2f}\N{DEGREE SIGN}, inclination: {z_angle:4.2f}\N{DEGREE SIGN})"
        length_str = f"length: {self.length_from_wall:5.2f} m"
        #range_str = f"range: {tuple(self.line_bounds[i_bh])}"
        return f"#{self.id} {pos_str} \n    -> {angle_str}, {length_str}"

    def place_t_points(self, n_points, point_step):
        #half_points = int((n_points - 1) / 2)
        i_points = np.arange(n_points)
        #dir_length = np.linalg.norm(self.unit_direction)
        #pt_step = (self.unit_direction / dir_length) * point_step
        t_points = self.t_well_head + point_step * i_points
        points = self.line_points(t_points)[:, 0]

        # Bounds given by active cylinder
        r, l0, l1 = self.lateral.active_cylinder
        mask = (points > l0)
        min_bound = np.argmax(mask)
        mask = (points < l1)
        min_bound_flipped = np.argmax(np.flip(mask))
        max_bound = n_points - min_bound_flipped
        assert len(points) == n_points

        min_bound = max(0, min_bound)
        max_bound = min(n_points, max_bound)
        #points = self.lateral.transform(points)
        return t_points, (min_bound, max_bound)


    def place_points(self, n_points, point_step):
        t_points, pt_range = self.place_t_points(n_points, point_step)
        points = self.lateral.transform(self.line_points(t_points)), pt_range
        return points

@attrs.define(slots=False)
class BoreholeSet:
    """
    Class for creating a set of boreholes around single lateral using its coordinate system:
    - meaning of axis is same, but X and Y have opposed sign
    - origin is at the center of avoidance cylinder at intersection of L5 and lateral central lines.
    """
    # y_angles: List[int]            # y boreholes angles to consider
    # z_angles: List[int]            # z borehole angles to consider
    # wh_pos_step: Tuple[float, float, float] # xyz wellhead position step
    #
    # wh_zones: List[Box]
    #point_step: float
    #n_points: int
    # _y_angle_range = (-80, 80)      # achive 2m from 10m distance
    # _z_angle_range = (-60, 60)
    boreholes: List[Borehole]
    lateral: 'Lateral'
    orig_indices: List[int]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Optionally remove the cached_property data if not needed
        #state.pop('expensive_computation', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    @property
    def n_boreholes(self):
        return len(self.boreholes)


    def subset(self, indices):
        new_set = [self.boreholes[i] for i in indices]

        return BoreholeSet(new_set, self.lateral, orig_indices=indices)

    def subset_by_ids(self, ids):
        ids = set(ids)
        new_indices = [idx for idx, bh in enumerate(self.boreholes) if bh.id in ids]
        return self.subset(new_indices)

    def _distance(self, cyl, line):
        """
        Line - Cylinder approximate distance, viw PyVista.
        Not reliable.
        :param cyl:
        :param line:
        :return:
        """
        points = np.linspace(line[0], line[1], 10)
        line_polydata = pv.PolyData(points)
        distances = line_polydata.compute_implicit_distance(cyl, inplace=False)['implicit_distance']
        return np.min(distances)

    def transform(self, points):
        return self.lateral.transform(points)

    def boreholes_print_sorted(self, f_name=None):
        """
        Sort boreholes by estimated distance from cylinder.
        Print distance - borehole ID pairs.
        :return:
        """
        r, l0, l1 = self.lateral.avoid_cylinder
        cylinder = pv.Cylinder(center=self.transform([0.5 * l0 + 0.5 * l1, 0, 0]), direction=(1, 0, 0), radius=r, height=l1-l0)
        distances = np.abs([self._distance(cylinder, (bh.start, bh.transversal)) for bh in self.boreholes])
        indices = np.argsort(distances)
        report = "\n".join([
            f"{dist} | {bh.bh_description}"
            for dist, bh in zip(distances, self.boreholes)
        ])
        if f_name:
            with open(f_name, 'w') as f:
                f.write(report)
        else:
            print(report)

    def points(self, cfg):
        """

        :return:
        """
        zk_cfg = cfg.boreholes.active_zk
        n_points = zk_cfg.common.n_points_per_bh
        point_step = zk_cfg.common.point_step
        placed = [ bh.place_points(n_points, point_step) for bh in self.boreholes]
        points, bounds = zip(*placed)
        return np.array((points)), bounds

    @property
    def line_bounds(self):
        """
        Range of valid point indices on the borehole.
        :return: (n_boreholes, 2)
        """
        return self.point_lines[1]


@bcommon.memoize
def project_field(workdir, cfg, bh_set, from_sample=0) -> 'BoreholeField':
    """
    Return array (n_boreholes, n_points, n_times, n_samples)
    """
    dset_name = "pressure"
    samples_chunk_size = 1024

    # get nonexisting borehole files within the range.
    (workdir / "borehole_data").mkdir(parents=True, exist_ok=True)
    bh_files = [workdir / "borehole_data" / f"bh_{i_bh:03d}.h5" for i_bh in range(bh_set.n_boreholes) ]
    # full dict
    bh_dict = {i_bh: bh_files[i] for i, i_bh in enumerate(range(bh_set.n_boreholes))}

    # borehole extraction matrix
    mesh = get_clear_mesh(workdir / cfg.simulation.mesh)
    n_el_mesh = mesh.n_cells
    points, bounds = bh_set.points(cfg)
    n_boreholes, n_points, n_dim = points.shape
    id_matrix = interpolation_slow(mesh, points)


    # Open the input HDF file
    field_file = workdir / cfg.simulation.hdf
    with h5py.File(field_file, 'r') as input_file:
        input_dataset = input_file[dset_name]
        n_samples, n_times, n_el = input_dataset.shape
        assert n_el == n_el_mesh

        output_shape = (n_points, n_times, input_dataset.shape[0])
        out_chunk_size = min(output_shape[2], samples_chunk_size)
        chunk_shape = (n_points, n_times, out_chunk_size)

        def write_group(out_file, out_slice, data):
            with h5py.File(workdir / out_file, mode='a') as out_f:
                dset = out_f.require_dataset(dset_name, shape=output_shape, chunks=chunk_shape, dtype='float64')
                dset[:, :, out_slice] = data


        # Iterate through chunks of the input dataset
        for i_sample in range(from_sample, input_dataset.shape[0], samples_chunk_size):
            print(f"Chunk: {i_sample} : {i_sample+samples_chunk_size}")
            sample_slice = slice(i_sample,i_sample+samples_chunk_size)
            input_chunk = np.array(input_dataset[sample_slice, :, :])
            transformed_chunk = input_chunk[:, :, id_matrix].transpose(2, 3, 1, 0)
            #cumul_chunk = np.cumsum(transformed_chunk, axis=1)  # cummulative sum along points
            for i, out_f in enumerate(bh_dict.values()):
                write_group(out_f, sample_slice, transformed_chunk[i])

    return BoreholeField(
        bh_set,
        points,
        bounds,
        bh_files)




@attrs.define(slots=False)
class BoreholeField:
    """
    Representation of data projected to the points on the boreholes
    """
    bh_set: BoreholeSet
    points : np.ndarray                     # (n_lines, n_points)
    point_bounds: List[Tuple[int, int]]     # for each borehole range of indices of active points
    data_files: List[pathlib.Path]             # assign a data file to each borehole

    def borohole_data(self, i_bh):
        with h5py.File(self.data_files[i_bh], mode='r') as f:
            bh_samples = np.array(f['pressure'])
        return bh_samples

def interpolation_slow(mesh, points):
    """
    Construct element_id matrix of shape (n_boreholes, n_points).

    Put n_points on every line
    :param mesh:
    :param points, shape (n_lines, n_points)
    :return:
    """
    cell_locator = vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    n_lines, n_points, dim = points.shape
    assert dim == 3
    interpolation_ids = np.empty([*points.shape[:2]], dtype=np.int32)
    for i_line in range(n_lines):
        interp_line = interpolation_ids[i_line, :]
        for i_pt in range(n_points):
            interp_line[i_pt] = max(0, cell_locator.FindCell(points[i_line, i_pt]))
        # good_ids = interp_line[interp_line != -1]
        # n_good = len(good_ids)
        # if n_good == n_points:
        #     continue
        #
        # # set first good id to first -1 ids, set the last good id to the rest
        # if len(good_ids) == 0:
        #     good_ids = [0]
        # i_subst = 0
        # for i_pt in range(n_points):
        #     if interp_line[i_pt] == -1:
        #         interp_line[i_pt] = good_ids[i_subst]
        #     else:
        #         i_subst = -1
    return interpolation_ids


"""
Single borehole given by:
position on L5 vertical axis plane, small vertical range, large horizontal range, target point

- wellheads generated uniformly over given ranges
- for each well head we generate range of direction vectors
- eliminate boreholes interacting with cylinder
Jedna konfigurace vrtů:
restrictions:
- 2 boreholes for ZK40, parallel under, 6 boreholes from ZK30, 2 parallel under, 4 generic 2 upper
=> directions to upper and lower, group boreholes according to direction, able to get wellheads for given direction
? how to prescribe restrictions to genetic algorithm

Separate optimization of:
 
 - ZK40 2 lower
 - ZK30 2 lower parallel, 2 upper generic, 2 generic 
 - ZK40 2 upper 
 In principle I'm takeing given number of boreholes from defined borehole subsets.
 Muting and mating within these subsets provides resonable results. 
 Posterior check of borehole interactions necessary, we iterate until we get viable ofspring.
  
ZK40: upper/lower, index of direction, 2 indeces from available 
"""



def get_clear_mesh(mesh_file):
    mesh = pv.read(mesh_file)
    mesh.clear_data()
    return mesh




def get_time_field(file_pattern, field_name):
    """
    Read files given by pattern, separate into numpy array and mesh.

    Return: field array, mesh
    - field array of shape (n_times, n_elements)
    - mesh without data fields
    """
    time_slices = []
    files = glob.glob(file_pattern)
    if not files:
        raise IOError(file_pattern)
    for filepath in files:
        mesh = pv.read(filepath)
        time_slices.append(mesh.cell_data.get_array(field_name))
        mesh.clear_data()
    return np.array(time_slices), mesh

def line_points(lines, n_points):
    """
    :param lines: (n_lines, 2, 3), single line: [point, direction]
    :param n_points:
    :return: (n_lines, n_points, 3)
    """
    p1 = lines[:, 0, :]
    dir = lines[:, 1, :]
    params = np.linspace(0, 1, n_points)
    return dir[:, None, :] * params[:,None] + p1[:, None, :]





"""
Úloha jak z daného bodu generovat vrty, které se vyhnou tunelu
- tunel/rozrážka zjednodušen na válec, kterému se potřebujeme vyhnout
- vrt daný příčkou mimoběžek (vrt vs. osa tunelu), 
  ta daná parametrem (st) bodu  (P) na ose tunelu
  a úhlem (phi) (směr vzhůru je phi=0) + pravidlo pravé ruky s palcem ve směru tunelu
- příčka nakonec dána jednotkovou normálou N a bodem na ose tunelu P
- rovnice přímky vrtu zhlaví PV, směrnice T:

  PV .. wellhead (given)
  Tv .. direction of well
  Pt .. center of (lateral) tunnel on the border of the L5 gallery (given), like (+/- 2, -/+ 5, 2) 
  Tt .. unit direction vector of tunnel (given) (+/- 1, 0, 0)

  We consider transversal of skew lines: the tunnel center line and the well
  It intersects and is perpendicular to both lines. 
  We consider the transversl given by its angle phi in YZ plane and by
  its intersection with the tunnel center line:
  N = (0, sin(phi), cos(phi))
  P = st * Tt + Pt # transwersal equation
  phi=0 .... upwards, e_z  
  phi ... jeden z parametrů vrtu


  rovnice vrtu:
  X = v * Tv  + PV

  1. rovina obsahující příšku a PV
  n1 = (P - PV) - (P-PV).N N  
  n1 = n1 / |n1|

  n2 = cross(N, n1)
  normálová rovnice roviny:
  X . n1 = PV . n1 

  (T . t + PV) . n1 = PV . n1
  T. t . n1 = 0

  2. z pruseciku roviny kolme na N a prochazejici PV:
  rovina: X . N = PV . N
  pricka: X = N . s + P
  s + P . N = PV . N
  s = (PV - P) . N

  přímka vrtu:
  prochází skrze PV a (PV-P)@N * N * s + P
  X = PV + v * ((PV-P)@N * N * s - (PV-P))

  We want s > r(st). 
  So how P and N exactly looks like:

  Lp = abs(PVy - yt)    # vzdálenost zhlaví od pozice rozrážky 
  Lt .. délka tunelu

  přímka osy tunelu ve směru X:
  Tt = (+/-1, 0, 0)
  Pt = (xt, zy, zt)  ; fixed, xt = 2 (half width of the main tunnel), 
  P = Tt . st + Pt, 0 < st < Lt 

  st .. parameter of the tunnel center line
  r(st) = min(Rt, Lp + st * (Rt - Lp) / Lt)    


  So the plusability criteria is:
  (PV - Tt.st - Pt) . (0, sin(phi), cos(phi))  > Rt
  or 

  (PV - Tt.st - Pt) . (0, sin(phi), cos(phi))  > Lp + st * (Rt - Lp) / Lt
  (PV - Pt).N - Lp > [(Tt . N) + (Rt-Lp) / Lt ] * st 

  Tt . N = 0 =>
  (PV - Pt).N - Lp > (Rt-Lp) / Lt  * st
  st <  [(PV - Pt).N - Lp] * Lt / (Rt - Lp) 

"""

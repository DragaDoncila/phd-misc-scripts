from functools import partial
import math
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import igraph
from itertools import combinations, product
import dask.dataframe as df
import dask.array as da

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from tqdm import tqdm
import pandas as pd


def euclidean_cost_func(source_node, dest_node):
    return np.linalg.norm(
        np.asarray(dest_node["coords"]) - np.asarray(source_node["coords"])
    )


def min_pairwise_distance_cost(child_distances):
    """Get the smallest sum of distance from parent to two potential children.

    Parameters
    ----------
    child_distances : List[Tuple[float]]
        list of distances to k closest children
    """
    min_dist = math.inf
    for i, j in combinations(range(len(child_distances)), 2):
        # TODO: no concept of angles
        distance_first_child = child_distances[i]
        distance_second_child = child_distances[j]
        distance_sum = distance_first_child + distance_second_child
        if distance_sum < min_dist:
            min_dist = distance_sum
    return min_dist

def closest_neighbour_child_cost(parent_coords, child_coords):
    min_dist = math.inf
    for i, j in combinations(range(len(child_coords)), 2):
        coords_i = child_coords[i]
        coords_j = child_coords[j]
        inter_child_dist = np.linalg.norm(coords_j - coords_i)
        dist_i = np.linalg.norm(parent_coords - coords_i)
        dist_j = np.linalg.norm(parent_coords - coords_j)
        total_dist = inter_child_dist + min(dist_i, dist_j)
        if total_dist < min_dist:
            min_dist = total_dist
    return min_dist


def dist_to_edge_cost_func(bounding_box_dimensions, node_coords):
    min_to_edge = math.inf
    box_mins = [dim[0] for dim in bounding_box_dimensions]
    # skip time coords as not relevant
    for i in range(len(node_coords)):
        box_dim_max = bounding_box_dimensions[1][i]
        box_dim_min = bounding_box_dimensions[0][i]

        node_val = node_coords[i]
        distance_to_min = node_val - box_dim_min
        distance_to_max = box_dim_max - node_val
        smallest = (
            distance_to_min if distance_to_min < distance_to_max else distance_to_max
        )
        min_to_edge = min_to_edge if min_to_edge < smallest else smallest
    return min_to_edge


class FlowGraph:
    def __init__(
        self,
        im_dim: Tuple[int],
        coords: 'dask.dataframe.DataFrame',
        min_t=0,
        max_t=None,
        pixel_vals: List[int] = None,
        migration_only: bool = False
    ) -> None:
        """Generate a FlowGraph from the coordinates with given pixel values

        Coords should be an list or numpy ndarray of nD point coordinates
        corresponding to identified objects within an nD image. Pixel vals
        is optional and should be of same length as coords. Time
        is assumed to be the first dimension of each coordinate.

        Parameters
        ----------
        im_dim : List[Tuple[int], Tuple[int]]
            top left and bottom right of frame bounding box (across all frames)
        coords : DataFrame
            DataFrame with columns 't', 'y', 'x' and optionally 'z' of
            blob center coordinates for which to solve
        min_t: int, optional
            smallest frame number in the image. If missing, will be determined
            from min value of first coordinate of each object in coords
        max_t: int, optional
            largest frame number in the image. If missing, will be determined
            from max value of first coordinate of each object in coords.
        pixel_vals : List[int], optional
            pixel value of each object at coordinate, by default None
        migration_only: bool, optional
            Whether the model ignores divisions or not, by default False
        """
        self.min_t = min_t or coords['t'].min()
        self.max_t = max_t or coords['t'].max()
        self.t = self.max_t - self.min_t + 1
        self.im_dim = im_dim
        self.migration_only = migration_only
        # TODO: we basically immediately instantiate the dask df - let's just take pandas from the getgo
        self.spatial_cols = ['y', 'x']
        if 'z' in coords.columns:
            self.spatial_cols.insert(0, 'z')
        if not isinstance(coords, df.DataFrame):
            coords = df.from_pandas(coords, chunksize=10000)
        self._g = self._init_nodes(coords, pixel_vals)
        self._kdt_dict = self._build_trees()
        self._init_edges()

    
    def _build_trees(self):
        """Build dictionary of t -> kd tree of all coordinates in frame t.
        """
        kd_dict = {}

        for t in tqdm(range(self.min_t, self.max_t + 1), total=self.t, desc='Building kD trees'):
            frame_vertices = self._g.vs(t=t)
            frame_indices = np.asarray([v.index for v in frame_vertices])
            frame_coords = frame_vertices['coords']
            new_tree = KDTree(frame_coords)
            kd_dict[t] = {'tree': new_tree, 'indices': frame_indices}
        return kd_dict

    def _init_nodes(self, coords, pixel_vals=None):
        """Create igraph from coords and pixel_vals with len(coords) vertices.

        Parameters
        ----------
        coords : DataFrame
            DataFrame with columns 't', 'y', 'x' and optionally 'z' of
            blob center coordinates for which to solve
        pixel_vals : List[int], optional
            List of integer values for each node, by default None
        """
        n = len(coords)
        if not pixel_vals:
            pixel_vals = np.arange(n, dtype=np.uint16)
        pixel_vals = pd.Series(pixel_vals)

        coords_numpy = coords[self.spatial_cols].compute().to_numpy()
        false_arr = np.broadcast_to(False, n)
        times = coords['t'].compute()
        all_attrs_dict = {
            'label': times.astype(str).str.cat(pixel_vals.astype(str), sep='_'),
            'coords': coords_numpy,
            'pixel_value': pixel_vals,
            't': times.to_numpy(dtype=np.int32),
            'is_source': false_arr,
            'is_target': false_arr,
            'is_appearance': false_arr,
            'is_division': false_arr,
        }
        g = igraph.Graph(directed=True)
        g.add_vertices(len(coords), all_attrs_dict)

        self.source = g.add_vertex(
            name="source",
            label="source",
            coords=np.asarray((-5, -2)),
            pixel_value=None,
            t=-1,
            is_source=True,
            is_target=False,
            is_appearance=False,
            is_division=False,
        )

        self.appearance = g.add_vertex(
            name="appearance",
            label="appearance",
            coords=np.asarray((-1, -1)),
            pixel_value=None,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=True,
            is_division=False,
        )

        self.target = g.add_vertex(
            name="target",
            label="target",
            coords=np.asarray((-1, -1)),
            pixel_value=None,
            t=self.max_t + 1,  # max frame index is max_t
            is_source=False,
            is_target=True,
            is_appearance=False,
            is_division=False,
        )

        if not self.migration_only:
            self.division = g.add_vertex(
                name="division",
                label="division",
                # TODO: will break for 4d maybe
                coords=np.asarray((-1, -5)),
                pixel_value=None,
                t=-1,
                is_source=False,
                is_target=False,
                is_appearance=False,
                is_division=True,
            )

        return g

    def _init_edges(self):
        self._init_appear_exit_edges()
        self._init_migration_division_edges()

    def _init_appear_exit_edges(self):
        """Connect appearance to all vertices, and all vertices to target.

        Cost for appearance is 0 for nodes in the first frame,
        and proportional to the distance of the node from a box
        edge for remaining frames.
        Cost for exit is 0 for nodes in the final frame,
        and proportional to distance of the node from closest
        box edge for remaining frames.
        """
        self._g.add_edge(self.source, self.appearance, cost=0, var_name="e_sa", label=0)

        print("Computing appearance/exit costs")
        real_nodes = self._g.vs(lambda v: not self._is_virtual_node(v))
        real_node_coords = np.asarray([v["coords"] for v in real_nodes])
        cost_func = partial(dist_to_edge_cost_func, self.im_dim)
        real_node_costs = np.apply_along_axis(cost_func, 1, real_node_coords)

        var_names_app = []
        costs_app = []
        edges_app = []
        labels_app = []

        var_names_target = []
        costs_target = []
        edges_target = []
        labels_target = []

        for i, v in tqdm(
            enumerate(real_nodes),
            desc="Making appearance/exit edges",
            total=len(real_nodes),
        ):
            # first frame should be able to appear at no extra cost
            if v["t"] == self.min_t:
                cost_app = 0
                cost_target = real_node_costs[i]
            # final frame should flow into exit at no extra cost
            elif v["t"] == self.max_t:
                cost_app = real_node_costs[i]
                cost_target = 0
            else:
                cost_app = cost_target = real_node_costs[i]
            
            var_names_app.append(f"e_a_{v['t']}.{v['pixel_value']}")
            costs_app.append(cost_app)
            edges_app.append((self.appearance.index, v.index))
            labels_app.append(str(cost_app)[:4])

            var_names_target.append(f"e_{v['t']}.{v['pixel_value']}_t")
            costs_target.append(cost_target)
            edges_target.append((v.index, self.target.index))
            labels_target.append(str(cost_target)[:4])
        
        app_attrs = {
            'label': labels_app,
            'var_name': var_names_app,
            'cost': costs_app
        }
        target_attrs = {
            'label': labels_target,
            'var_name': var_names_target,
            'cost': costs_target
        }

        self._g.add_edges(edges_app, attributes=app_attrs)
        self._g.add_edges(edges_target, attributes=target_attrs)

    def _init_migration_division_edges(self):
        """Connect all pairs vertices in frames 0..n-1 to 1..n.

        Cost is computed using the migration cost function.
        """
        edges = []
        var_names = []
        all_costs = []
        labels = []

        if not self.migration_only:
            self._g.add_edge(self.source, self.division, cost=0, var_name="e_sd", label=0)
        edges_div = []
        all_costs_div = []
        var_names_div = []
        labels_div = []

        for source_t in tqdm(
                range(self.min_t, self.max_t),
                desc=f"Making migration & division edges",
                total=self.t-1,
            ):
            dest_t = source_t + 1
            source_nodes = self._g.vs(t=source_t)

            source_coords = da.asarray(source_nodes['coords'])
            dest_tree = self._kdt_dict[dest_t]['tree']
            # TODO: parameterize the closest neighbours
            dest_distances, dest_indices = dest_tree.query(source_coords, k=10 if dest_tree.n > 10 else dest_tree.n - 1)

            for i, src in enumerate(source_nodes):
                dest_vertex_indices = self._kdt_dict[dest_t]['indices'][dest_indices[i]]
                dest_vertices = self._g.vs[list(dest_vertex_indices)]
                # We're relying on these indices not changing partway through construction.
                np.testing.assert_allclose(dest_tree.data[dest_indices[i]], [v['coords'] for v in dest_vertices])
                
                current_edges = [(src.index, dest_index) for dest_index in dest_vertex_indices]
                current_costs = dest_distances[i]
                current_var_names = [f"e_{src['t']}.{src['pixel_value']}_{dest['t']}.{dest['pixel_value']}" for dest in dest_vertices]
                current_labels = [str(cost)[:5] for cost in current_costs]

                edges.extend(current_edges)
                var_names.extend(current_var_names)
                labels.extend(current_labels)
                all_costs.extend(current_costs)

                if self.migration_only:
                    continue

                division_edge = (self.division.index, src.index)
                cost_div = closest_neighbour_child_cost(src['coords'], dest_tree.data[dest_indices[i]])
                var_name_div = f"e_d_{src['t']}.{src['pixel_value']}"
                label_div = str(cost_div)[:5]

                edges_div.append(division_edge)
                all_costs_div.append(cost_div)
                var_names_div.append(var_name_div)
                labels_div.append(label_div)

        all_attrs = {
            'var_name': var_names + var_names_div,
            'cost': all_costs + all_costs_div,
            'label': labels + labels_div
        }
        self._g.add_edges(edges+edges_div, attributes=all_attrs)

    def _is_virtual_node(self, v):
        return (
            v["is_source"] or v["is_appearance"] or v["is_division"] or v["is_target"]
        )

    def _may_divide(self, v):
        return not self._is_virtual_node(v) and self.min_t <= v["t"] < self.max_t

    def _bounded_edge(self, e):
        var_name = e["var_name"]
        return "e_sd" not in var_name and "e_sa" not in var_name

    def _get_var_sum_str(self, var_names, neg=""):
        var_sum = ""
        for edge_name in var_names:
            var_sum += f"{neg}{edge_name} + "
        var_sum = var_sum.rstrip(" +")
        return var_sum

    def _get_objective_string(self):
        var_names = self._g.es['var_name']
        edge_costs = self._g.es['cost']
        obj_str = "Minimize\n\t"
        for i in range(len(var_names)):
            var_cost_str = f"{edge_costs[i]} {var_names[i]} + "
            obj_str += var_cost_str
        obj_str = obj_str.rstrip(" +")
        obj_str += "\n"
        return obj_str

    def _get_incident_edges(self, node):
        incoming_indices = self._g.incident(node, "in")
        incoming = self._g.es.select(incoming_indices)
        outgoing_indices = self._g.incident(node, "out")
        outgoing = self._g.es.select(outgoing_indices)
        return incoming, outgoing

    def _get_flow_constraints(self):
        # out of source and into target
        source_outgoing_names = self._get_incident_edges(self.source)[1]['var_name']
        target_incoming_names = self._get_incident_edges(self.target)[0]["var_name"]
        source_outgoing_sum = self._get_var_sum_str(source_outgoing_names)
        target_incoming_sum = self._get_var_sum_str(target_incoming_names, neg="-")
        network_capacity_str = f"\tflow_all: {source_outgoing_sum} + {target_incoming_sum} = 0\n"

        # division & appearance
        appearance_incoming, appearance_outgoing = self._get_incident_edges(
            self.appearance
        )
        appearance_incoming_sum = self._get_var_sum_str(appearance_incoming['var_name'])
        appearance_outgoing_sum = self._get_var_sum_str(appearance_outgoing['var_name'], neg="-")
        virtual_capacity_str = (
            f"\tflow_app: {appearance_incoming_sum} + {appearance_outgoing_sum} = 0\n"
        )

        if not self.migration_only:
            division_incoming, division_outgoing = self._get_incident_edges(self.division)
            division_incoming_sum = self._get_var_sum_str(division_incoming['var_name'])
            division_outgoing_sum = self._get_var_sum_str(division_outgoing["var_name"], neg="-")
            virtual_capacity_str += (
                f"\tflow_div: {division_incoming_sum} + {division_outgoing_sum} = 0\n"
            )

        # inner nodes
        inner_node_str = ""
        for t in range(self.min_t, self.max_t + 1):
            t_nodes = self._g.vs.select(t=t)
            for i, node in enumerate(t_nodes):
                incoming_edges, outgoing_edges = self._get_incident_edges(node)
                incoming_names = incoming_edges['var_name']
                outgoing_names = outgoing_edges['var_name']

                incoming_sum = self._get_var_sum_str(incoming_names)
                outgoing_sum = self._get_var_sum_str(outgoing_names, neg="-")
                inner_node_str += f"\tflow_{t}.{i}: {incoming_sum} + {outgoing_sum} = 0\n"
            inner_node_str += "\n"
        flow_const = f"\\Total network\n{network_capacity_str}\n\\Virtual nodes\n{virtual_capacity_str}\n\\Inner nodes\n{inner_node_str}"
        return flow_const

    def _get_flow(self):
        """Generate minimal flow through model - all nodes in first frame must appear.

        Returns
        -------
        flow_str: str
            string enforcing appearance flow into first frame
        """
        # get first frame vertices
        first_frame_appearance = self._g.es(cost=0)
        first_frame_appearance = list(filter(lambda e: f'a_{self.min_t}' in e['var_name'], first_frame_appearance))
        appearance_names = [e['var_name'] for e in first_frame_appearance]
        flow_str = "\\Flow from source\n"
        for i, edge in enumerate(appearance_names):
            flow_str += f"\tforced_{i}: {edge} = 1\n"
        return flow_str

    def _get_division_constraints(self):
        """Constrain conditions required for division to occur.

        1. We must have flow from appearance or migration before we have flow
            from division
        """
        div_str = "\\Division constraints\n"
        potential_parents = self._g.vs(self._may_divide)
        for i, v in enumerate(potential_parents):
            incoming, outgoing = self._get_incident_edges(v)
            div_edge = incoming(lambda e: "e_d" in e["var_name"])[0]["var_name"]
            other_incoming_edges = incoming(lambda e: "e_d" not in e["var_name"])
            incoming_sum = self._get_var_sum_str(
                [e["var_name"] for e in other_incoming_edges]
            )

            # must have appearance or immigration before we divide
            div_str += f"\tdiv_{i}: {incoming_sum} - {div_edge} >= 0\n"

        return div_str

    def _get_constraints_string(self):
        cons_str = "Subject To\n"
        cons_str += self._get_flow_constraints()
        if not self.migration_only:
            cons_str += self._get_division_constraints()
        cons_str += self._get_flow()
        return cons_str

    def _get_bounds_string(self):
        bounds_str = "Bounds\n"

        for edge in self._g.es(self._bounded_edge):
            bounds_str += f'\t0 <= {edge["var_name"]} <= 1\n'
        return bounds_str

    def _to_lp(self, path):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string()
        bounds_str = self._get_bounds_string()

        total_str = f"{obj_str}\n{constraints_str}\n{bounds_str}"
        with open(path, "w") as f:
            f.write(total_str)


if __name__ == "__main__":
    import pandas as pd
    # coords = [(0, 50.0, 50.0), (0, 40, 50), (0, 30, 57), (1, 50, 52), (1, 38, 51), (1, 29, 60)]
    # pixel_vals = [1, 2, 3, 1, 2, 3]
    model_pth = '/home/draga/PhD/code/experiments/preconfig/models/misc'
    coords = [
        (0, 50.0, 50.0),
        (0, 40, 50),
        (0, 30, 57),
        (1, 50, 52),
        (1, 38, 51),
        (1, 29, 60),
        (2, 52, 53),
        (2, 37, 53),
        (2, 28, 64),
    ]
    coords = pd.DataFrame(coords, columns=['t', 'y', 'x'])

    pixel_vals = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    graph = FlowGraph([(0, 0), (100, 100)], coords, min_t=0, max_t=2, pixel_vals=pixel_vals, migration_only=True)
    igraph.plot(graph._g, layout=graph._g.layout('rt'))
    graph._to_lp(os.path.join(model_pth, 'labelled_constraints.lp'))
    # print(cdist(coords[0:3], coords[3:6]))

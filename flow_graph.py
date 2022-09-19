from functools import partial
import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import igraph
from itertools import combinations, product

from scipy.spatial.distance import cdist
from tqdm import tqdm


def euclidean_cost_func(source_node, dest_node):
    # smaller distance should be bigger cost, so we take reciprocal
    return 1 / np.linalg.norm(
        np.asarray(dest_node["coords"]) - np.asarray(source_node["coords"])
    )


def min_pairwise_distance_cost(potential_parent, potential_children):
    """Get the smallest sum of distance from parent to two potential children.

    Parameters
    ----------
    potential_parent : Tuple[float]
        coordinates of potential parent
    potential_children : List[Tuple[float]]
        list of coordinates of potential children
    """
    min_dist = math.inf
    for child_pair in combinations(potential_children, 2):
        # TODO: no concept of angles
        distance_first_child = np.linalg.norm(
            np.asarray(potential_parent) - np.asarray(child_pair[0])
        )
        distance_second_child = np.linalg.norm(
            np.asarray(potential_parent) - np.asarray(child_pair[1])
        )
        distance_sum = distance_first_child + distance_second_child
        if distance_sum < min_dist:
            min_dist = distance_sum
    # the smaller the distance, the bigger the prize
    return 1 / min_dist


# TODO: fix so box doesn't have to start at 0
def dist_to_edge_cost_func(bounding_box_dimensions, node_coords):
    min_to_edge = math.inf
    # skip time coords as not relevant
    for i in range(1, len(bounding_box_dimensions)):
        box_max = bounding_box_dimensions[i]
        node_val = node_coords[i]
        distance_to_min = node_val - 0
        distance_to_max = box_max - node_val
        smallest = (
            distance_to_min if distance_to_min < distance_to_max else distance_to_max
        )
        min_to_edge = min_to_edge if min_to_edge < smallest else smallest
    # smaller distance should be bigger cost (to encourage taking it) so take reciprocal
    return 1 / min_to_edge


class FlowGraph:
    def __init__(
        self,
        im_dim: Tuple[int],
        coords: List[Union[Tuple, np.ndarray]],
        cost_func: Optional[Callable] = None,
        min_t=0,
        max_t=None,
        pixel_vals: List[int] = None,
    ) -> None:
        """Generate a FlowGraph from the coordinates with given pixel values

        Coords should be an list or numpy ndarray of nD point coordinates
        corresponding to identified objects within an nD image. Pixel vals
        is optional and should be of same length as coords. Time
        is assumed to be the first dimension of each coordinate.

        Parameters
        ----------
        im_dim : Tuple[int]
            maxes of image bounding box
        coords : List[Union[Tuple|np.ndarray]]
            coordinates of each identified object, with first coordinate
            being time
        cost_func: Optional[Callable]
            function that takes two vertices and determines cost of edge
            between them. By default, euclidean distance between node centers
        min_t: int, optional
            smallest frame number in the image. If missing, will be determined
            from min value of first coordinate of each object in coords
        max_t: int, optional
            largest frame number in the image. If missing, will be determined
            from max value of first coordinate of each object in coords.
        pixel_vals : List[int], optional
            pixel value of each object at coordinate, by default None
        """
        self.min_t = min_t or min([coord[0] for coord in coords])
        self.max_t = max_t or max([coord[0] for coord in coords])
        self.t = self.max_t - self.min_t + 1
        self.cost_func = cost_func or euclidean_cost_func
        self.im_dim = im_dim
        self._g = self._init_nodes(coords, pixel_vals)
        self._binary_indicators = []
        self._init_edges()

    def _init_nodes(self, coords, pixel_vals=None):
        """Create igraph from coords and pixel_vals with len(coords) vertices.

        Parameters
        ----------
        coords : _type_
            _description_
        pixel_vals : _type_, optional
            _description_, by default None
        """
        if not pixel_vals:
            pixel_vals = [None for _ in range(len(coords))]
        g = igraph.Graph(directed=True)
        for i in range(len(coords)):
            g.add_vertex(
                name=f"{i}",
                label=f"{coords[i][0]}_{pixel_vals[i]}",
                # TODO: time shouldn't factor here right?
                coords=coords[i],
                pixel_value=pixel_vals[i],
                t=coords[i][0],
                is_source=False,
                is_target=False,
                is_appearance=False,
                is_division=False,
            )

        self.source = g.add_vertex(
            name="source",
            label="source",
            coords=None,
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
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=True,
            is_division=False,
        )

        self.division = g.add_vertex(
            name="division",
            label="division",
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=False,
            is_division=True,
        )

        self.target = g.add_vertex(
            name="target",
            label="target",
            coords=None,
            pixel_value=None,
            t=self.max_t + 1,  # max frame index is max_t
            is_source=False,
            is_target=True,
            is_appearance=False,
            is_division=False,
        )

        return g

    def _init_edges(self):
        # self._init_appear_exit_edges()

        # make edge for all inner nodes
        self._init_migration_edges()

        self._init_division_edges()

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
        for i, v in tqdm(
            enumerate(real_nodes),
            desc="Making appearance/exit edges",
            total=len(real_nodes),
        ):
            var_name_app = f"e_a_{v['t']}{v['pixel_value'] or ''}"
            var_name_target = f"e_{v['t']}{v['pixel_value' or '']}_t"
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
            self._g.add_edge(
                self.appearance,
                v,
                cost=cost_app,
                var_name=var_name_app,
                label=str(cost_app)[:4],
            )
            self._g.add_edge(
                v,
                self.target,
                cost=cost_target,
                var_name=var_name_target,
                label=str(cost_target)[:4],
            )

    def _init_migration_edges(self):
        """Connect all pairs vertices in frames 0..n-1 to 1..n.

        Cost is computed using the migration cost function.
        """
        for source_t in range(self.min_t, self.max_t):
            dest_t = source_t + 1
            source_nodes = self._g.vs(t=source_t)
            source_coords = np.asarray([node["coords"] for node in source_nodes])
            dest_nodes = self._g.vs(t=dest_t)
            dest_coords = np.asarray([node["coords"] for node in dest_nodes])
            print(f"Computing costs {source_t}-{dest_t}")
            costs = cdist(source_coords, dest_coords)
            for i, src in tqdm(
                enumerate(source_nodes),
                desc=f"Making migration edges frame {source_t}",
                total=len(source_nodes),
            ):
                for j, dest in enumerate(dest_nodes):
                    cost = costs[i, j]
                    var_name = f"e_{src['t']}{src['pixel_value']}_{dest['t']}{dest['pixel_value']}"
                    self._g.add_edge(
                        src, dest, cost=cost, var_name=var_name, label=str(cost)[:5]
                    )

    def _init_division_edges(self):
        """Connect source to division to all vertices up to 2nd last frame."""
        self._g.add_edge(self.source, self.division, cost=0, var_name="e_sd", label=0)
        potential_parents = self._g.vs(self._may_divide)
        for v in potential_parents:
            var_name = f"e_d_{v['t']}{v['pixel_value' or '']}"
            # TODO: should just grab them all in a func outside loop
            potential_children = self._g.vs(t=v["t"] + 1)
            cost = min_pairwise_distance_cost(
                v["coords"], [child["coords"] for child in potential_children]
            )
            self._g.add_edge(
                self.division, v, cost=cost, var_name=var_name, label=str(cost)[:5]
            )

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
        var_names = [edge["var_name"] for edge in self._g.es]
        edge_costs = [edge["cost"] for edge in self._g.es]
        obj_str = "Maximize\n\t"
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
        source_outgoing_names = [
            edge["var_name"] for edge in self._get_incident_edges(self.source)[1]
        ]
        target_incoming_names = [
            edge["var_name"] for edge in self._get_incident_edges(self.target)[0]
        ]
        source_outgoing_sum = self._get_var_sum_str(source_outgoing_names)
        target_incoming_sum = self._get_var_sum_str(target_incoming_names, neg="-")
        network_capacity_str = f"{source_outgoing_sum} + {target_incoming_sum} = 0\n"

        # division & appearance
        division_incoming, division_outgoing = self._get_incident_edges(self.division)
        appearance_incoming, appearance_outgoing = self._get_incident_edges(
            self.appearance
        )
        division_incoming_sum = self._get_var_sum_str(
            [edge["var_name"] for edge in division_incoming]
        )
        division_outgoing_sum = self._get_var_sum_str(
            [edge["var_name"] for edge in division_outgoing], neg="-"
        )
        appearance_incoming_sum = self._get_var_sum_str(
            [edge["var_name"] for edge in appearance_incoming]
        )
        appearance_outgoing_sum = self._get_var_sum_str(
            [edge["var_name"] for edge in appearance_outgoing], neg="-"
        )
        virtual_capacity_str = (
            f"\t{division_incoming_sum} + {division_outgoing_sum} = 0\n"
        )
        virtual_capacity_str += (
            f"\t{appearance_incoming_sum} + {appearance_outgoing_sum} = 0\n"
        )

        # inner nodes
        inner_node_str = ""
        for t in range(self.min_t, self.max_t + 1):
            t_nodes = self._g.vs.select(t=t)
            for node in t_nodes:
                incoming_edges, outgoing_edges = self._get_incident_edges(node)
                incoming_names = [edge["var_name"] for edge in incoming_edges]
                outgoing_names = [edge["var_name"] for edge in outgoing_edges]

                incoming_sum = self._get_var_sum_str(incoming_names)
                outgoing_sum = self._get_var_sum_str(outgoing_names, neg="-")
                inner_node_str += f"\t{incoming_sum} + {outgoing_sum} = 0\n"
            inner_node_str += "\n"
        flow_const = f"\\Total network\n\t{network_capacity_str}\n\\Virtual nodes\n{virtual_capacity_str}\n\\Inner nodes\n{inner_node_str}"
        return flow_const

    def _get_flow(self):
        source_outgoing_names = [
            edge["var_name"] for edge in self._get_incident_edges(self.source)
        ]
        flow_str = "\\Flow from source\n"
        for edge in source_outgoing_names:
            flow_str += f"\t{edge} = 1\n"
        return flow_str

    def _get_division_constraints(self):
        """Constrain conditions required for division to occur.

        1. We must have flow from appearance or migration before we have flow
            from division
        2. If we have flow from division, we cannot have flow to target.
        """
        div_str = "\\Division constraints\n"
        potential_parents = self._g.vs(self._may_divide)
        for v in potential_parents:
            incoming, outgoing = self._get_incident_edges(v)
            div_edge = incoming(lambda e: "e_d" in e["var_name"])[0]["var_name"]
            other_incoming_edges = incoming(lambda e: "e_d" not in e["var_name"])
            incoming_sum = self._get_var_sum_str(
                [e["var_name"] for e in other_incoming_edges]
            )

            # must have appearance or immigration before we divide
            div_str += f"\t{incoming_sum} - {div_edge} >= 0\n"

            # if we have dv, we cannot flow into target
            target_edge = outgoing(lambda e: "_t" in e["var_name"])[0]["var_name"]
            delta_var = f"delta_{div_edge}"
            self._binary_indicators.append(delta_var)
            # TODO: make coefficient parameter?
            div_str += f"\t{div_edge} - {delta_var} <= 0\n"
            div_str += f"\t{div_edge} - 0.00001 {delta_var} >= 0\n"
            div_str += f"\t{delta_var} + {target_edge} <= 1\n\n"

        return div_str

    def _get_constraints_string(self):
        cons_str = "Subject To\n"
        cons_str += self._get_flow_constraints()
        cons_str += self._get_division_constraints()
        # cons_str += self._get_flow()
        return cons_str

    def _get_bounds_string(self):
        bounds_str = "Bounds\n"

        for edge in self._g.es(self._bounded_edge):
            bounds_str += f'\t0 <= {edge["var_name"]} <= 1\n'
        return bounds_str

    def _get_binary_var_str(self):
        return "Binary\n\t" + " ".join(self._binary_indicators)

    def _to_lp(self, path):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string()
        bounds_str = self._get_bounds_string()
        binary_vars = self._get_binary_var_str()

        total_str = f"{obj_str}\n{constraints_str}\n{bounds_str}\n{binary_vars}"
        with open(path, "w") as f:
            f.write(total_str)


if __name__ == "__main__":
    # coords = [(0, 50.0, 50.0), (0, 40, 50), (0, 30, 57), (1, 50, 52), (1, 38, 51), (1, 29, 60)]
    # pixel_vals = [1, 2, 3, 1, 2, 3]

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
    pixel_vals = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    graph = FlowGraph((3, 100, 100), coords, max_t=2, pixel_vals=pixel_vals)
    # igraph.plot(graph._g, layout=graph._g.layout('rt'))
    # graph._to_lp('new_division_cost.lp')
    # print(cdist(coords[0:3], coords[3:6]))

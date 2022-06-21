from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import igraph
from itertools import product


def euclidean_cost_func(source_node, dest_node):
    # smaller distance should be bigger cost, so we take reciprocal
    return 1 / np.linalg.norm(np.asarray(dest_node['coords']) - np.asarray(source_node['coords']))


class FlowGraph:

    def __init__(self, coords: List[Union[Tuple, np.ndarray]], cost_func: Optional[Callable] = None, t:int = None, pixel_vals: List[int]=None) -> None:
        """Generate a FlowGraph from the coordinates with given pixel values

        Coords should be an list or numpy ndarray of nD point coordinates
        corresponding to identified objects within an nD image. Pixel vals
        is optional and should be of same length as coords. Time
        is assumed to be the first dimension of each coordinate.

        Parameters
        ----------
        coords : List[Union[Tuple|np.ndarray]]
            coordinates of each identified object, with first coordinate
            being time
        cost_func: Optional[Callable]
            function that takes two vertices and determines cost of edge 
            between them. By default, euclidean distance between node centers
        t : int, optional
            number of frames in the image. If missing, will be determined
            from max value of first coordinate of each object in coords
        pixel_vals : List[int], optional
            pixel value of each object at coordinate, by default None
        """
        self.t = t or max([coord[0] for coord in coords])
        self.cost_func = cost_func or euclidean_cost_func
        self._g = self._init_graph(coords, pixel_vals)
        self._init_edges()

    def _init_graph(self, coords, pixel_vals=None):
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
        g = igraph.Graph(n=len(coords), directed=True)
        for i in range(len(coords)):
            g.add_vertex(
                name=f'{i}',
                coords=coords[i],
                pixel_value=pixel_vals[i],
                t=coords[i][0],
                is_source=False,
                is_target=False,
            )
        
        g.add_vertex(
            name='source',
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=True,
            is_target=False,
        )

        g.add_vertex(
            name='target',
            coords=None,
            pixel_value=None,
            t=self.t, # max frame index is t-1
            is_source=False,
            is_target=True,
        )

        return g

    def _init_edges(self):
        # make edge from source to all vertices in first frame
        src = self._g.vs(name='source')[0]
        first_frame = self._g.vs(t=0)
        for v in first_frame:
            var_name = f"e_s_{v['t']}{v['pixel_value'] or ''}"
            self._g.add_edge(src, v, cost=0, var_name=var_name)

        # make edge from vertices in last frame to target
        target = self._g.vs(name='target')[0]
        last_frame = self._g.vs(t=self.t-1)
        for v in last_frame:
            var_name = f"e_{v['t']}{v['pixel_value' or '']}_t"
            self._g.add_edge(v, target, cost=0, var_name=var_name)

        # make edge for all inner nodes
        for source_t in range(0, self.t-1):
            dest_t = source_t + 1
            source_nodes = self._g.vs(t=source_t)
            dest_nodes = self._g.vs(t=dest_t)
            all_edges = product(source_nodes, dest_nodes)
            for source, dest in all_edges:
                var_name = f"e_{source['t']}{source['pixel_value']}_{dest['t']}{dest['pixel_value']}"
                cost = self.cost_func(source, dest)
                self._g.add_edge(source, dest, cost=cost, var_name=var_name)

    def _get_var_sum_str(self, var_names, neg=''):
        var_sum = ''
        for edge_name in var_names:
            var_sum += f'{neg}{edge_name} + '
        var_sum = var_sum.rstrip(' +')
        return var_sum

    def _get_objective_string(self):
        var_names = [edge['var_name'] for edge in self._g.es]
        edge_costs = [edge['cost'] for edge in self._g.es]
        obj_str = 'Maximize\n\t'
        for i in range(len(var_names)):
            var_cost_str = f'{edge_costs[i]} {var_names[i]} + '
            obj_str += var_cost_str
        obj_str = obj_str.rstrip(' +')
        obj_str += '\n'
        return obj_str

    def _get_incident_edges(self, node):
        incoming_indices = self._g.incident(node, 'in')
        incoming = self._g.es.select(incoming_indices)
        outgoing_indices = self._g.incident(node, 'out')
        outgoing = self._g.es.select(outgoing_indices)
        return incoming, outgoing

    def _get_flow_constraints(self):
        # out of source and into target
        source_outgoing_names = [edge['var_name'] for edge in self._get_incident_edges(self._g.vs.find(is_source=True))[1]]
        target_incoming_names = [edge['var_name'] for edge in self._get_incident_edges(self._g.vs.find(is_target=True))[0]]
        source_outgoing_sum = self._get_var_sum_str(source_outgoing_names)
        target_incoming_sum = self._get_var_sum_str(target_incoming_names, neg='-')
        network_capacity_str = f'{source_outgoing_sum} + {target_incoming_sum} = 0\n'

        inner_node_str = ''
        for t in range(self.t):
            t_nodes = self._g.vs.select(t=t)
            for node in t_nodes:
                incoming_edges, outgoing_edges = self._get_incident_edges(node)
                incoming_names = [edge['var_name'] for edge in incoming_edges]
                outgoing_names = [edge['var_name'] for edge in outgoing_edges]

                incoming_sum = self._get_var_sum_str(incoming_names)
                outgoing_sum = self._get_var_sum_str(outgoing_names, neg='-')
                inner_node_str += f'\t{incoming_sum} + {outgoing_sum} = 0\n'
            inner_node_str += '\n'
        flow_const = f'\\Total network\n\t{network_capacity_str}\n\\Inner nodes\n{inner_node_str}'
        return flow_const

    def _get_flow(self):
        source_outgoing_names = [edge['var_name'] for edge in self._get_incident_edges(self._g.vs.find(is_source=True))[1]]
        flow_str = '\\Flow from source\n'
        for edge in source_outgoing_names:
            flow_str += f'\t{edge} = 1\n'
        return flow_str

    def _get_constraints_string(self):
        cons_str = 'Subject To\n'
        cons_str += self._get_flow_constraints()
        # cons_str += self._get_flow()
        return cons_str

    def _get_bounds_string(self):
        bounds_str = 'Bounds\n'

        for edge in self._g.es:
            bounds_str += f'\t0 <= {edge["var_name"]} <= 1\n'
        return bounds_str

    def _to_lp(self, path):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string()
        bounds_str = self._get_bounds_string()

        total_str = f'{obj_str}\n{constraints_str}\n{bounds_str}'
        with open(path, 'w') as f:
            f.write(total_str)

if __name__ == '__main__':
    coords = [(0, 50.0, 50.0), (0, 40, 50), (0, 30, 57), (1, 50, 52), (1, 38, 51), (1, 29, 60)]
    pixel_vals = [1, 2, 3, 1, 2, 3]
    graph = FlowGraph(coords, t=2, pixel_vals=pixel_vals)
    graph._to_lp('simple_auto_gen2.lp')

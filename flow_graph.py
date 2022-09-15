import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import igraph
from itertools import combinations, product


def euclidean_cost_func(source_node, dest_node):
    # smaller distance should be bigger cost, so we take reciprocal
    return 1 / np.linalg.norm(np.asarray(dest_node['coords']) - np.asarray(source_node['coords']))

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
        #TODO: no concept of angles
        distance_first_child = np.linalg.norm(np.asarray(potential_parent)-np.asarray(child_pair[0]))
        distance_second_child = np.linalg.norm(np.asarray(potential_parent) - np.asarray(child_pair[1]))
        distance_sum = distance_first_child + distance_second_child
        if distance_sum < min_dist:
            min_dist = distance_sum
    # the smaller the distance, the bigger the prize
    return 1 / min_dist

# TODO: fix so box doesn't have to start at 0
def dist_to_edge_cost_func(node, bounding_box_dimensions):
    min_to_edge = math.inf
    # skip time coords as not relevant
    for i in range(1, len(bounding_box_dimensions)):
        box_max = bounding_box_dimensions[i]
        node_val = node['coords'][i]
        distance_to_min = node_val - 0
        distance_to_max = box_max - node_val
        smallest = distance_to_min if distance_to_min < distance_to_max else distance_to_max
        min_to_edge = min_to_edge if min_to_edge < smallest else smallest
    # smaller distance should be bigger cost (to encourage taking it) so take reciprocal
    return 1 / min_to_edge

class FlowGraph:

    def __init__(self, im_dim: Tuple[int], coords: List[Union[Tuple, np.ndarray]], cost_func: Optional[Callable] = None, t:int = None, pixel_vals: List[int]=None) -> None:
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
        t : int, optional
            number of frames in the image. If missing, will be determined
            from max value of first coordinate of each object in coords
        pixel_vals : List[int], optional
            pixel value of each object at coordinate, by default None
        """
        self.t = t or max([coord[0] for coord in coords])
        self.cost_func = cost_func or euclidean_cost_func
        self.im_dim = im_dim
        self._g = self._init_graph(coords, pixel_vals)
        self._binary_indicators = []
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
        g = igraph.Graph(directed=True)
        for i in range(len(coords)):
            g.add_vertex(
                name=f'{i}',
                label=f'{coords[i][0]}_{pixel_vals[i]}',
                # TODO: time shouldn't factor here right?
                coords=coords[i],
                pixel_value=pixel_vals[i],
                t=coords[i][0],
                is_source=False,
                is_target=False,
                is_appearance=False,
                is_division=False
            )
        
        g.add_vertex(
            name='source',
            label='source',
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=True,
            is_target=False,
            is_appearance=False,
            is_division=False
        )

        g.add_vertex(
            name='appearance',
            label='appearance',
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=True,
            is_division=False
        )

        g.add_vertex(
            name='division',
            label='division',
            coords=None,
            pixel_value=None,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=False,
            is_division=True
        )

        g.add_vertex(
            name='target',
            label='target',
            coords=None,
            pixel_value=None,
            t=self.t, # max frame index is t-1
            is_source=False,
            is_target=True,
            is_appearance=False,
            is_division=False
        )

        return g

    def _init_edges(self):
        # make edge from source to all vertices
        self._init_appearance_edges()

        # make edge from vertices in last frame to target
        self._init_exit_edges()

        # make edge for all inner nodes
        self._init_migration_edges()
        
        self._init_division_edges()

    def _init_appearance_edges(self):
        """Connect source to appearance to all other vertices.

        Cost for appearance is 0 for nodes in the first frame,
        and proportional to the distance of the node from a box
        edge for remaining frames.
        """
        src = self._g.vs(name='source')[0]
        app = self._g.vs(name='appearance')[0]
        self._g.add_edge(src, app, cost=0, var_name='e_sa', label=0)
        first_frame = self._g.vs(t=0)
        remaining_frames = self._g.vs(t_gt=0)
        # first frame should be able to appear at no extra cost
        for v in first_frame:
            var_name = f"e_a_{v['t']}{v['pixel_value'] or ''}"
            self._g.add_edge(app, v, cost=0, var_name=var_name, label=0)
        for v in remaining_frames:
            if not v['is_target']:
                var_name = f"e_a_{v['t']}{v['pixel_value'] or ''}"
                self._g.add_edge(app, v, cost=dist_to_edge_cost_func(v, self.im_dim), var_name=var_name, label=str(dist_to_edge_cost_func(v, self.im_dim))[:5])

    def _init_exit_edges(self):
        """Connect all vertices to target.

        Cost for exit is 0 for nodes in the final frame,
        and proportional to distance of the node from closest
        box edge for remaining frames.
        """
        target = self._g.vs(name='target')[0]
        real_nodes = self._g.vs(lambda v: not self._is_virtual_node(v))
        last_frame = real_nodes(t=self.t-1)
        remaining_frames = real_nodes(t_lt=self.t-1)
        for v in remaining_frames:
                var_name = f"e_{v['t']}{v['pixel_value' or '']}_t"
                #TODO: incorporate chance of death as well?
                cost = dist_to_edge_cost_func(v, self.im_dim)
                self._g.add_edge(v, target, cost=cost, var_name=var_name, label=str(cost)[:5])

        # final frame should flow into exit at no extra cost
        for v in last_frame:
            var_name = f"e_{v['t']}{v['pixel_value' or '']}_t"
            self._g.add_edge(v, target, cost=0, var_name=var_name, label=0)

    def _init_migration_edges(self):
        """Connect all pairs vertices in frames 0..n-1 to 1..n.

        Cost is computed using the migration cost function.
        """
        for source_t in range(0, self.t-1):
            dest_t = source_t + 1
            source_nodes = self._g.vs(t=source_t)
            dest_nodes = self._g.vs(t=dest_t)
            all_edges = product(source_nodes, dest_nodes)
            for source, dest in all_edges:
                var_name = f"e_{source['t']}{source['pixel_value']}_{dest['t']}{dest['pixel_value']}"
                cost = self.cost_func(source, dest)
                self._g.add_edge(source, dest, cost=cost, var_name=var_name, label=str(cost)[:5])

    def _init_division_edges(self):
        """Connect source to division to all vertices up to 2nd last frame.

        Cost?
        """
        src = self._g.vs(name='source')[0]
        div = self._g.vs(name='division')[0]
        self._g.add_edge(src, div, cost=0, var_name='e_sd', label=0)
        potential_parents = self._g.vs(self._may_divide)
        for v in potential_parents:
            var_name = f"e_d_{v['t']}{v['pixel_value' or '']}"
            # TODO: should just grab them all in a func outside loop
            potential_children = self._g.vs(t=v['t']+1)
            cost = min_pairwise_distance_cost(v['coords'], [child['coords'] for child in potential_children])
            self._g.add_edge(div, v, cost=cost, var_name=var_name, label=str(cost)[:5])

    def _is_virtual_node(self, v):
        return v['is_source'] or v['is_appearance'] or v['is_division'] or v['is_target']

    def _may_divide(self, v):
        return not self._is_virtual_node(v) and v['t'] < self.t-1

    def _bounded_edge(self, e):
        var_name = e['var_name']
        return 'e_sd' not in var_name and 'e_sa' not in var_name

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

        # division & appearance
        division_incoming, division_outgoing = self._get_incident_edges(self._g.vs.find(is_division=True))
        appearance_incoming, appearance_outgoing = self._get_incident_edges(self._g.vs.find(is_appearance=True))
        division_incoming_sum = self._get_var_sum_str([edge['var_name'] for edge in division_incoming])
        division_outgoing_sum = self._get_var_sum_str([edge['var_name'] for edge in division_outgoing], neg='-')
        appearance_incoming_sum = self._get_var_sum_str([edge['var_name'] for edge in appearance_incoming])
        appearance_outgoing_sum = self._get_var_sum_str([edge['var_name'] for edge in appearance_outgoing], neg='-')
        virtual_capacity_str = f'\t{division_incoming_sum} + {division_outgoing_sum} = 0\n'
        virtual_capacity_str += f'\t{appearance_incoming_sum} + {appearance_outgoing_sum} = 0\n'

        # inner nodes
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
        flow_const = f'\\Total network\n\t{network_capacity_str}\n\\Virtual nodes\n{virtual_capacity_str}\n\\Inner nodes\n{inner_node_str}'
        return flow_const

    def _get_flow(self):
        source_outgoing_names = [edge['var_name'] for edge in self._get_incident_edges(self._g.vs.find(is_source=True))[1]]
        flow_str = '\\Flow from source\n'
        for edge in source_outgoing_names:
            flow_str += f'\t{edge} = 1\n'
        return flow_str

    def _get_division_constraints(self):
        """Constrain conditions required for division to occur.

        1. We must have flow from appearance or migration before we have flow
            from division
        2. If we have flow from division, we cannot have flow to target.
        """
        div_str = '\\Division constraints\n'
        potential_parents = self._g.vs(self._may_divide)
        for v in potential_parents:
            incoming, outgoing = self._get_incident_edges(v)
            div_edge = incoming(lambda e: 'e_d' in e['var_name'])[0]['var_name']
            other_incoming_edges = incoming(lambda e: 'e_d' not in e['var_name'])
            incoming_sum = self._get_var_sum_str([e['var_name'] for e in other_incoming_edges])
            
            # must have appearance or immigration before we divide
            div_str += f'\t{incoming_sum} - {div_edge} >= 0\n'

            # if we have dv, we cannot flow into target
            target_edge = outgoing(lambda e: '_t' in e['var_name'])[0]['var_name']
            delta_var = f'delta_{div_edge}'
            self._binary_indicators.append(delta_var)
            #TODO: make coefficient parameter?
            div_str += f'\t{div_edge} - {delta_var} <= 0\n'
            div_str += f'\t{div_edge} - 0.00001 {delta_var} >= 0\n'
            div_str += f'\t{delta_var} + {target_edge} <= 1\n\n'

        return div_str
    
    def _get_constraints_string(self):
        cons_str = 'Subject To\n'
        cons_str += self._get_flow_constraints()
        cons_str += self._get_division_constraints()
        # cons_str += self._get_flow()
        return cons_str

    def _get_bounds_string(self):
        bounds_str = 'Bounds\n'

        for edge in self._g.es(self._bounded_edge):
            bounds_str += f'\t0 <= {edge["var_name"]} <= 1\n'
        return bounds_str

    def _get_binary_var_str(self):
        return 'Binary\n\t' + ' '.join(self._binary_indicators)

    def _to_lp(self, path):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string()
        bounds_str = self._get_bounds_string()
        binary_vars = self._get_binary_var_str()

        total_str = f'{obj_str}\n{constraints_str}\n{bounds_str}\n{binary_vars}'
        with open(path, 'w') as f:
            f.write(total_str)

if __name__ == '__main__':
    coords = [(0, 50.0, 50.0), (0, 40, 50), (0, 30, 57), (1, 50, 52), (1, 38, 51), (1, 29, 60)]
    pixel_vals = [1, 2, 3, 1, 2, 3]
    graph = FlowGraph((2, 100, 100), coords, t=2, pixel_vals=pixel_vals)
    # igraph.plot(graph._g, layout=graph._g.layout('kk'))
    graph._to_lp('new_division_cost.lp')

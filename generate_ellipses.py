from dis import dis
from importlib.resources import is_resource
from typing import Callable, Optional
import napari
from skimage.draw import disk
import numpy as np

def euclidean_cost_func(edge: 'Edge'):
    return np.linalg.norm(np.asarray(edge.source.coords) - np.asarray(edge.dest.coords))


class Node:

    def __init__(self, coords=None, pixel_val=None, t=None, is_source=False, is_target=False) -> None:
        self.coords = coords
        self.pixel_value = pixel_val
        self.t = t
        self.is_source = is_source
        self.is_target = is_target

    def __repr__(self):
        return f"[{self.pixel_value}, {self.coords}]"

class Edge:

    def __init__(self, source: Node, dest: Node, cost: Optional[float] = 0) -> None:
        self.source = source
        self.dest = dest
        self.cost = cost
        #TODO: assuming frame+pixel_val is unique identifier which is big if true...
        self.var_name = f"e_{self.source.t if not self.source.is_source else 's'}{self.source.pixel_value or ''}_{self.dest.t if not self.dest.is_target else 't'}{self.dest.pixel_value or ''}"

    def __repr__(self):
        return f"{self.var_name}: {self.cost}"

class FlowGraph:

    def __init__(self, coords, pixel_vals=None, r=None) -> None:
        self.t = len(coords)
        if pixel_vals is None:
            pixel_vals = [list(range(1, len(frame_coords)+1)) for frame_coords in coords]
        self.nodes = self._init_nodes(coords, pixel_vals)
        self.edges = self._init_edges()
        self.r = r

    def _init_nodes(self, coords, pixel_vals):
        nodes = {}

        # add Source node (t = -1 implies source)
        nodes[-1] = [Node(t=-1, is_source=True)]

        for t, cells in enumerate(coords):
            nodes[t] = []
            for i, cell in enumerate(cells):
                node = Node(cell, pixel_vals[t][i], t)
                nodes[t].append(node)

        # add Target node
        nodes[len(coords)] = [Node(t=len(coords), is_target=True)]
        return nodes

    def _init_edges(self):
        edges = []
        # make source - frame 1 edges
        src = self.nodes[-1][0]
        for cell in self.nodes[0]:
            new_source_edge = Edge(src, cell)
            edges.append(new_source_edge)

        # make frame by frame edges
        for t_i in range(0, self.t-1):
            t_j = t_i+1
            current_frame_cells = self.nodes[t_i]
            next_frame_cells = self.nodes[t_j]

            for src in current_frame_cells:
                for dest in next_frame_cells:
                    new_edge = Edge(src, dest)
                    new_edge.cost = euclidean_cost_func(new_edge)
                    edges.append(new_edge)
            
        # make last frame - target edges
        dest = self.nodes[self.t][0]
        for cell in self.nodes[self.t-1]:
            new_target_edge = Edge(cell, dest)
            edges.append(new_target_edge)

        return edges

    def _get_var_sum_str(self, var_names):
        var_sum = ''
        for edge_name in var_names:
            var_sum += f'{edge_name} + '
        var_sum = var_sum.rstrip(' +')
        return var_sum

    def _get_objective_string(self):
        var_names = [edge.var_name for edge in self.edges]
        edge_costs = [edge.cost for edge in self.edges]
        obj_str = 'Minimize\n\t'
        for i in range(len(var_names)):
            var_cost_str = f'{edge_costs[i]} {var_names[i]} + '
            obj_str += var_cost_str
        obj_str = obj_str.rstrip(' +')
        obj_str += '\n'
        return obj_str

    def _get_incident_edges(self, node):
        incoming = []
        outgoing = []
        for edge in self.edges:
            if edge.dest is node:
                incoming.append(edge)
            if edge.source is node:
                outgoing.append(edge)
        return incoming, outgoing

    def _get_flow_constraints(self, known_cell_count):
        # out of source and into target
        source_outgoing_names = [edge.var_name for edge in self.edges if edge.source.is_source]
        target_incoming_names = [edge.var_name for edge in self.edges if edge.dest.is_target]
        source_outgoing_sum = self._get_var_sum_str(source_outgoing_names)
        target_incoming_sum = self._get_var_sum_str(target_incoming_names)
        network_capacity_str = f'{source_outgoing_sum} = {target_incoming_sum}\n'

        inner_node_str = ''
        for t in range(self.t):
            t_nodes = self.nodes[t]
            for node in t_nodes:
                incoming_edges, outgoing_edges = self._get_incident_edges(node)
                incoming_names = [edge.var_name for edge in incoming_edges]
                outgoing_names = [edge.var_name for edge in outgoing_edges]

                incoming_sum = self._get_var_sum_str(incoming_names)
                outgoing_sum = self._get_var_sum_str(outgoing_names)
                inner_node_str += f'{incoming_sum} = {outgoing_sum}\n'
            inner_node_str += '\n'
        flow_const = f'\\Total network\n{network_capacity_str}\n\\Inner nodes\n{inner_node_str}'
        print(flow_const)

        return flow_const
            
    def _get_constraints_string(self, known_cell_count):
        cons_str = 'Subject To\n\t'
        cons_str += self._get_flow_constraints(known_cell_count)

    def _to_lp(self, path, known_cell_count=None):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string(known_cell_count)

        total_str = obj_str + constraints_str
        print(total_str)

def draw_first_frame(image, n, r, max_intercell_dist):
    """Draw n non-overlapping circles of radius r on image, and return center coords

    Parameters
    ----------
    image : np.ndarray
        3D numpy array to draw on
    n : int
        number of circles to draw
    r : int
        ellipse radius
    max_intercell_dist: float
        maximum distance between different cells in single frame
    """
    # initialize to center to give us wiggle room
    first_center = (image.shape[1]/2, image.shape[2]/2)
    center_coords = [first_center]
    rr, cc = disk(first_center, r)
    image[0][rr, cc] = 1

    while len(center_coords) < n:
        potential_center = (np.random.randint(2*r, image.shape[1]-2*r), np.random.randint(2*r, image.shape[2])-2*r)
        new_canvas = np.zeros(image.shape[1:], dtype=np.uint8)
        rr, cc = disk(potential_center, r)
        new_canvas[rr, cc] = len(center_coords) + 1
        if not np.any(np.logical_and(image[0], new_canvas)) and close_enough_to_previous(potential_center, center_coords[-1], max_intercell_dist, r):
            image[0] += new_canvas
            center_coords.append(potential_center)

    print("Generated first frame:", center_coords)
    return center_coords

def valid_prev_frame_partner(new_cell, old_cell, min_dist, max_dist):
    return min_dist <= np.linalg.norm(np.asarray(new_cell)-np.asarray(old_cell)) <= max_dist
        
        
def close_enough_to_previous(new_cell, last_cell, max_intercell_dist, r):
    return 2*r <= np.linalg.norm(np.asarray(new_cell)-np.asarray(last_cell)) <= 2*r+max_intercell_dist

def draw_remaining_frames(image, circle_coords, n, r, min_dist, max_dist):
    """Generate n circles for remaining t-1 frames.

    Circles are coloured by their "matching" object in the previous frame,
    and will be generated so as not to overlap with other ellipses in the
    same frame. Circle centers will be between min_dist and max_dist away
    from their matching object in the previous frame. 

    Parameters
    ----------
    image : np.ndarray
        2D+T image of circles
    circle_coords : List
        n*t nested list of center coords for all ellipses
    n : int
        number of circles to generate
    r : int
        radius of circles
    min_dist : int
        minimum euclidean distance between circle centers in neighbour frames
    max_dist : int
        maximum euclidean distance between circle centers in neighbour frames
    """
    current_frame_coords = []
    t, height, width = image.shape
    for i in range(1, t):
        while len(current_frame_coords) < n:
            current_num_ellipses = len(current_frame_coords)
            new_canvas = np.zeros((height, width), dtype=np.uint8)
            previous_center = circle_coords[i-1][current_num_ellipses]
            potential_center = (np.random.randint(2*r, height-2*r), np.random.randint(2*r, width-2*r))
            # close enough to itself in previous cell but also close enough to other cells in same frame
            while not valid_prev_frame_partner(potential_center, previous_center, min_dist, max_dist):
                potential_center = (np.random.randint(2*r, height-2*r), np.random.randint(2*r, width-2*r))
            rr, cc = disk(potential_center, r)
            new_canvas[rr, cc] = current_num_ellipses + 1
            # no overlap between new frame and old frame
            if not np.any(np.logical_and(image[i], new_canvas)):
                image[i] += new_canvas
                current_frame_coords.append(potential_center)
        circle_coords[i] = current_frame_coords
        current_frame_coords = []


def generate_circles(shape, r, n, t, min_dist, max_dist, max_intercell_dist):
    """Generate n circles across t frames.

    Each circle in one frame "moves" in the next frame
    by a randomly generated euclidean distance between min_dist
    and max_dist away. Overlapping circles are rejected.

    Parameters
    ----------
    shape : tuple
        size of canvas - (x, y) for each frame
    r : int
        radius of each circle
    n : int
        number of circles for each frame
    t : int
        number of frames
    min_dist : float
        minimum distance between "partner" circles in neighbouring frames
    max_dist : float
        maximum distance between "partner" circles in neighbouring frames
    max_intercell_dist: float
        maximum distance between different cells in single frame
    """
    image = np.zeros((t, *shape), dtype=np.uint8)
    circle_coords = [[None for _ in range(n)] for _ in range(t)]

    # generate circles for first frame
    circle_coords[0] = draw_first_frame(image, n, r, max_intercell_dist)

    # generate remaining frames
    draw_remaining_frames(image, circle_coords, n, r, min_dist, max_dist)

    return image, circle_coords

def get_image_from_coords(shape, t, coords, radius):
    image = np.zeros((t, *shape), dtype=np.uint8)
    for i in range(t):
        for j, cell_center in enumerate(coords[i]):
            rr, cc = disk(cell_center, radius)
            image[i, rr, cc] += j+1
    return image

if __name__ == "__main__":
    canvas_size = (100, 100)
    n_cells = 3
    time_frames = 2

    # generating new ones
    # radius = 5
    # min_displacement = 0.5
    # max_displacement = 4
    # max_intercell_dist = 5
    # image, coords = generate_circles(canvas_size, radius, n_cells, time_frames, min_displacement, max_displacement, max_intercell_dist)
    # print(coords)

    # close together coords
    # radius = 5
    # min_displacement = 0.5
    # max_displacement = 1
    # max_intercell_dist = 3    
    # coords = [[(50.0, 50.0), (49, 62), (56, 70)], [(50, 49), (50, 62), (56, 69)]]
    # image = get_image_from_coords(canvas_size, time_frames, coords, radius)

    # less close together coords
    radius = 5
    min_displacement = 0.5
    max_displacement = 4
    max_intercell_dist = 5
    coords = [[(50.0, 50.0), (40, 50), (30, 57)], [(50, 52), (38, 51), (29, 60)]]
    image = get_image_from_coords(canvas_size, time_frames, coords, radius)

    graph = FlowGraph(coords, r=radius)
    # for edge in graph.edges:
    #     print(edge)
    graph._to_lp('blah')

    # viewer = napari.Viewer()
    # viewer.add_labels(image)

    # napari.run()                    

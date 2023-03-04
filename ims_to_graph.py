import glob
import numpy as np
from tifffile import TiffFile
import napari
import h5py
from scipy.ndimage import center_of_mass, label
from skimage.measure import regionprops
from skimage.graph import pixel_graph, central_pixel
# from generate_ellipses import FlowGraph
from flow_graph import FlowGraph

IMS_DIR = '/home/draga/PhD/data/Fluo-C2DL-Huh7/01/'
GT_DIR = '/home/draga/PhD/data/Fluo-C2DL-Huh7/01_GT/SEG/'

H5_PTH = '/home/draga/PhD/data/mitocheck_2d+t/mitocheck_94570_2D+t_01-53.h5'

def peek(im_file):
    with TiffFile(im_file) as im:
        im_shape = im.pages[0].shape
        im_dtype = im.pages[0].dtype
    return im_shape, im_dtype

def load_tiff_frames(im_dir=IMS_DIR):
    all_tiffs = list(sorted(glob.glob(f'{im_dir}*.tif')))
    n_frames = len(all_tiffs)
    frame_shape, im_dtype = peek(all_tiffs[0])
    im_array = np.zeros((n_frames, *frame_shape), dtype=im_dtype)
    for i, tiff_pth in enumerate(all_tiffs):
        with TiffFile(tiff_pth) as im:
            im_array[i] = im.pages[0].asarray()
    return im_array

def load_tiff(im_pth):
    with TiffFile(im_pth) as im:
        return im.pages[0].asarray()


def load_h5(im_pth=H5_PTH, group='volume', key='data'):
    with h5py.File(im_pth, "r") as f:
        print(f)
        dataset = np.squeeze(f[group][key][...])
    return dataset    

def get_real_center(prop):
    if prop.solidity > 0.9:
        return prop.centroid
    
    # shape is too convex to use centroid, get center from pixelgraph
    region = prop.image
    g, nodes = pixel_graph(region, connectivity=2)
    medoid_offset, _ = central_pixel(
            g, nodes=nodes, shape=region.shape, partition_size=100
            )
    medoid_offset = np.asarray(medoid_offset)
    top_left = np.asarray(prop.bbox[:region.ndim])
    medoid = tuple(top_left + medoid_offset)
    return medoid

def get_centers(segmentation):
    n_frames = segmentation.shape[0]
    cell_vals = [np.unique(frame)[1:] for frame in segmentation]
    centers_of_mass = []
    for i in range(n_frames):
        current_frame = segmentation[i]
        # regionprops the frame
        props = regionprops(current_frame)
        current_centers = [get_real_center(prop) for prop in props]
        centers_of_mass.append(current_centers)
    return centers_of_mass

def get_point_coords(centers_of_mass):
    points_list = []
    for i, frame_centers in enumerate(centers_of_mass):
        points = [(i, *center) for center in frame_centers]
        points_list.extend(points)
    return points_list

def get_pixel_value_at_centers(seg, seg_centers):
    pixel_vals = []
    for t in range(seg.shape[0]):
        frame_centers = seg_centers[t]
        frame = seg[t]
        int_centers = list(map(lambda tpl: tuple(int(el) for el in tpl), frame_centers))
        int_coords = list(map(list, zip(*int_centers)))
        frame_vals = list(frame[tuple(int_coords)])
        pixel_vals.extend(frame_vals)
    return pixel_vals

if __name__ == "__main__":
    # raw_im = load_tiff_frames(im_dir='/home/draga/PhD/data/Fluo-N2DL-HeLa/01/')
    # ds = load_h5()
    # raw_im = load_tiff('/home/draga/PhD/data/toy_data/easy-no-divide-swaps/Fluo-N2DL-HeLa.tif')
    # seg = load_tiff('/home/draga/PhD/data/toy_data/easy-no-divide-swaps/seg_Fluo-N2DL-HeLa.tif')

    viewer = napari.Viewer()

    im = viewer.open('/home/draga/PhD/data/toy_data/easy-no-divide-swaps/Fluo-N2DL-HeLa.tif', layer_type='image')[0].data
    # seg = viewer.open('/home/draga/PhD/data/toy_data/easy-no-divide-swaps/seg_Fluo-N2DL-HeLa.tif', layer_type='labels')[0].data
    # seg_centers = get_centers(seg)
    # point_coords = get_point_coords(seg_centers)
    # pixel_vals = get_pixel_value_at_centers(seg, seg_centers)
    # viewer.add_points(point_coords, size=5)     

    # graph = FlowGraph(point_coords, pixel_vals=pixel_vals)
    # for edge in graph._g.es:
    #     print(edge)
    # graph._to_lp('cell_swaps_autogen2.lp')
    napari.run()

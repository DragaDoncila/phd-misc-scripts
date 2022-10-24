import glob
import os
import napari
import numpy as np

from csbdeep.utils import normalize
import pandas as pd
from stardist.models import StarDist2D
import tifffile
from tifffile import TiffFile
from tqdm import tqdm

def peek(im_file):
    with TiffFile(im_file) as im:
        im_shape = im.pages[0].shape
        im_dtype = im.pages[0].dtype
    return im_shape, im_dtype

def load_tiff_frames(im_dir):
    all_tiffs = list(sorted(glob.glob(f'{im_dir}*.tif')))
    n_frames = len(all_tiffs)
    frame_shape, im_dtype = peek(all_tiffs[0])
    im_array = np.zeros((n_frames, *frame_shape), dtype=im_dtype)
    for i, tiff_pth in enumerate(all_tiffs):
        with TiffFile(tiff_pth) as im:
            im_array[i] = im.pages[0].asarray()
    return im_array
    

def segment_all_frames(im, pretrained=True, gt_path=None):
    """Segment all frames in image independently and return
    stacked label image, label centers and other details.

    Parameters
    ----------
    im : np.ndarray
        2D/3D+t (t, y, x) array of image
    pretrained : bool
        whether to load versatile pretrained model or train from GT
    gt_path: str | None
        path to ground truth frames - must be provided if pretrained is False
    """
    if pretrained:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    else:
        if not gt_path:
            raise RuntimeError("Path to ground truth annotations was not provided, but pretrained is False.")
        else:
            raise NotImplementedError("Coming soon.")

    label_stack = []
    center_list = []
    prob_list = []
    n_frames = im.shape[0]
    for t in tqdm(range(n_frames), total=n_frames, desc='Segmenting frames'):
        frame = im[t]
        labels, details = model.predict_instances(normalize(frame))
        label_stack.append(labels)
        centers = np.insert(details['points'], 0, t, axis=1)
        center_list.append(centers)
        prob_list.append(details['prob'])
    return (np.stack(label_stack), np.concatenate(center_list, axis=0), np.concatenate(prob_list, axis=0))

if __name__ == '__main__':
    DS_ROOT = os.path.expanduser('~/PhD/data/Cell Tracking Challenge/')
    DS_NAME = 'Fluo-N2DL-HeLa/01/'

    OUT_ROOT = os.path.expanduser('~/PhD/code/repos/misc-scripts/ctc')
    
    # load images from dataset into stacked np array
    im = load_tiff_frames(os.path.join(DS_ROOT, DS_NAME))
    # segment with StarDist frame by frame
    labels, centers, probs = segment_all_frames(im)


    viewer = napari.Viewer()
    viewer.add_image(im, name=DS_NAME)
    viewer.add_labels(labels, name='stardist_seg')
    viewer.add_points(centers, name='seg_centers')
    napari.run()
    
    # # make dataframe of t, y, x
    # centers_df = pd.DataFrame(centers, columns = ['t', 'y', 'x'])
    # centers_df['prob'] = probs

    # out_dir_pth = os.path.join(OUT_ROOT, DS_NAME)
    # out_csv_pth = os.path.join(out_dir_pth, 'centers.csv')
    # out_tif_pth = os.path.join(out_dir_pth, 'labels.tif')
    # os.makedirs(out_dir_pth, exist_ok=True)

    # centers_df.to_csv(out_csv_pth)
    # tifffile.imwrite(out_tif_pth, labels, compression=('zlib', 1))

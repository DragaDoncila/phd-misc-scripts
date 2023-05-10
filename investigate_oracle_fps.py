

import os
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatched
from traccuracy.metrics import CTCMetrics
from bmvc_metrics import load_sol

ROOT_DATA_DIR = '/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/'

ds_name = 'Fluo-N2DL-HeLa' # row.ds_name
seq = 2 # row.seq

sol_dir = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_RES/'.format(seq))
og_sol_pth = os.path.join(sol_dir, 'full_solution.graphml')
final_sol_pth = os.path.join(sol_dir, 'final_solution.graphml')

seg_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_ST/SEG/'.format(seq))
gt_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_GT/TRA/'.format(seq))

gt_data = load_ctc_data(gt_pth)
og_sol_data = load_sol(og_sol_pth, seg_pth, gt_data.segmentation)
final_sol_data = load_sol(final_sol_pth, seg_pth, gt_data.segmentation)

og_match = CTCMatched(gt_data, og_sol_data)
final_match = CTCMatched(gt_data, final_sol_data)

og_ctc = CTCMetrics(og_match)
final_ctc = CTCMetrics(final_match)

print(og_ctc.results)
print(final_ctc.results)

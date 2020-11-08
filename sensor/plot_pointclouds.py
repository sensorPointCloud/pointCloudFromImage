import numpy as np
import pptk

#val  = np.load('results/initial dot laser/dist_3_single_path_threshold_50_erosion_dilution_False_210_blk_dot.npy')
val  = np.load('dist_3_single_path_threshold_50_erosion_dilution_False_210_blk_dot_mid_floor.npy')
v = pptk.viewer(val, val[:, 2])
#v.attributes(val[['r', 'g', 'b']] / 255., val['i'])
v.set(point_size=5)
v.wait()

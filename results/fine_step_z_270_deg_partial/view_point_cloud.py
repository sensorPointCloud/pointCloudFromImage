import numpy as np
import pptk

val = np.load('results/dist_3_single_path_threshold_50_erosion_dilution_False_270_blk_dot_mid_floor.npy')
v = pptk.viewer(val, val[:, 2])
v.set(point_size=2)
v.wait()
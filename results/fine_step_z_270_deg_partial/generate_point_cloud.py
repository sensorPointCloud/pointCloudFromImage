import os
import sys

file_dir_path = os.path.dirname(os.path.realpath(__file__))

# add code directory to path
sys.path.append(file_dir_path + '/../../')

from sensor.optimization_angles_general import brute_optimize, plot_print_length, save_distances, \
    save_3d_distances

description = "270_blk_dot_mid_floor"

path_img_conf = file_dir_path + "/../../experiments/270deg_partial"

# Brute force optimize -------------------------------------------------------------------------------------------------
images_brute_optimize = [6, 13]

flip_xy_on_ols = True
use_erosion_and_dilation = True
threshold = 50
ranges_cam_laser_FOV = rclf = ((-20, 20), (-20, 20), (43, 55))

file_name_optimize = f'image_{images_brute_optimize}_' \
                     f'cam_{rclf[0][0]}_{rclf[0][1]}_' \
                     f'laser_{rclf[1][0]}_{rclf[1][1]}_' \
                     f'FOV_{rclf[2][0]}_{rclf[2][1]}_' \
                     f'flip_xy_on_ols_{flip_xy_on_ols}_' \
                     f'use_erosion_and_dilation_{use_erosion_and_dilation}'

file_name_optimize = file_dir_path + \
                     '/results/' + \
                     file_name_optimize \
                         .replace('[', '_') \
                         .replace(']', '_') \
                         .replace(',', '_') \
                         .replace(' ', '_')

# show_images(path_img_conf, images_brute_optimize, threshold)


if not os.path.exists(file_name_optimize + '.npy'):
    print('Starting optimization')
    brute_optimize(path_img_conf,
                   images_brute_optimize,
                   flip_xy_on_ols=True,
                   use_erosion_and_dilation=use_erosion_and_dilation,
                   threshold=threshold,
                   output_file_name=file_name_optimize,
                   ranges_cam_laser_FOV=ranges_cam_laser_FOV)
# Eo brute force optimize


# Find best initial parameters -----------------------------------------------------------------------------------------

fov, cam_angle_offset, laser_angle_offset = plot_print_length(file_name_optimize + '.npy',
                                                              path_img_conf, length_to_vertical_wall=2380,
                                                              do_plot=False)

# EO find best initial parameters


# Calculate and save distances -----------------------------------------------------------------------------------------

filename_2d_ang = 'results/' + \
                  f'dist_2d_single_path_threshold_{threshold}' \
                  f'_erosion_dilution_{use_erosion_and_dilation}' \
                  f'_{description}.npy'

filename_3d_ang = 'results/' + \
                  f'dist_3_single_path_threshold_{threshold}' \
                  f'_erosion_dilution_{use_erosion_and_dilation}' \
                  f'_{description}.npy'

# Calculate and save distances, x, z, rot z (2d + z angle)
if not os.path.exists(filename_2d_ang):
    save_distances(path_img_conf, filename_2d_ang, fov=fov, cam_angle_offset=cam_angle_offset,
                   laser_angle_offset=laser_angle_offset,
                   threshold=threshold, use_erosion_and_dilation=use_erosion_and_dilation,
                   limit_estimated_angle_to_field_of_view=37)

# Calculate and save distances, x, y, z
if not os.path.exists(filename_3d_ang):
    # Only distances in the range of the filter will be included in the point cloud
    xyz_filter = [[-10000, 3000], [-3500, 3500], [-1000, 3500]]
    save_3d_distances(filename_2d_ang, filename_3d_ang, xyz_filter)

# EO calculate and save distances


# # View point cloud -----------------------------------------------------------------------------------------------------
# val = np.load(filename_3d_ang)
# v = pptk.viewer(val, val[:, 2])
# v.set(point_size=2)
# v.wait()
# # EO view point cloud

import cv2

from sensor.cam_estimate_laser_angle import load_img_as_grey_with_threshold, show_image, \
    load_img_as_grey_with_threshold_erosion_dilation
from sensor.distance_from_files import load_dir, cam_angles_and_laser_angles_from_file, distances_from_image
from sensor.distance_from_files import cam_angles_from_image, distances_from_directory
from sensor.distance_estimation import get_distance, rot_z_3d
import tests.test_distance_estimation

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

config = tests.test_distance_estimation.config


def brute_optimize(path, img_nrs, use_erosion_and_dilation, threshold,
                   flip_xy_on_ols, output_file_name, ranges_cam_laser_FOV):  # Flip model if points are vertical
    """
    :string path: ex. 'C:\my\image\folder', contains folder of images and MCU data file
    :list img_nrs: ex. [1,5,7], the image number the optimization will run on
    :bool use_erosion_and_dilation:
    :int threshold: 0-255
    :bool flip_xy_on_ols:
    :string output_file_name: ex. 'C:\my\results', save path of results
    :tuple ranges_cam_laser_FOV: ex. ((-20, 20), (-20, 20), (43, 55)), optimize within this range of angles
    :return: None
    """

    rclf = ranges_cam_laser_FOV
    cam_range = [x / 10 for x in range(rclf[0][0], rclf[0][1])]
    laser_range = [x / 10 for x in range(rclf[1][0], rclf[1][1])]
    FOV_range = [x for x in range(rclf[2][0], rclf[2][1])]

    path_images, _ = load_dir(path)
    image_paths = []
    for img_nr in img_nrs:
        image_paths.append(path + '/' + path_images[img_nr])
    mcu_data_path = path + '/MCU_data.txt'
    dim_y = 300
    mcu_datas = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)

    n_elements = len(cam_range) * len(laser_range) * len(FOV_range)
    res = np.zeros((n_elements, 8))
    n = 0
    for fov in FOV_range:
        print('FOV:', fov)
        for cam_angle in cam_range:
            print('cam angle:', cam_angle)
            print(f'{n} of {n_elements}. Total: {round((n / n_elements) * 100, 3)}% ')
            for laser_angle in laser_range:
                const_cam_laser_offset = (cam_angle, laser_angle)
                distance_vecs = np.zeros((dim_y * len(img_nrs), 3))
                for idx, (image_path, img_nr) in enumerate(zip(image_paths, img_nrs)):
                    st = 0 + dim_y * idx
                    en = dim_y + dim_y * idx
                    distance_vecs[st:en, :] = distances_from_image(config, image_path, mcu_datas[img_nr], fov, dim_y,
                                                                   use_erosion_and_dilation, threshold,
                                                                   const_cam_laser_offset,
                                                                   limit_estimated_angle_to_fov=None
                                                                   )
                if flip_xy_on_ols:
                    model = LinearRegression().fit(distance_vecs[:, 1].reshape(-1, 1),
                                                   distance_vecs[:, 0].reshape(-1, 1))
                    err = model.score(distance_vecs[:, 1].reshape(-1, 1), distance_vecs[:, 0].reshape(-1, 1))
                else:
                    model = LinearRegression().fit(distance_vecs[:, 0].reshape(-1, 1),
                                                   distance_vecs[:, 1].reshape(-1, 1))
                    err = model.score(distance_vecs[:, 0].reshape(-1, 1), distance_vecs[:, 1].reshape(-1, 1))

                b = model.intercept_
                a = model.coef_

                res[n][0] = fov
                res[n][1] = cam_angle
                res[n][2] = laser_angle
                res[n][3] = a
                res[n][4] = b
                res[n][5] = err
                res[n][6] = np.mean(distance_vecs[:, 0])
                res[n][7] = np.mean(distance_vecs[:, 1])

                n = n + 1
    np.save(output_file_name, res)
    print('done')


def show_img_as_grey_with_threshold_erosion_dilation(path, threshold):
    img = load_img_as_grey_with_threshold(path, threshold)
    di_kernel = np.ones((5, 5), np.uint8)
    er_kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, er_kernel, iterations=1)
    img_dilation = cv2.dilate(img, di_kernel, iterations=1)
    img_erosion_dilation = cv2.dilate(img_erosion, di_kernel, iterations=1)
    show_image(img, 'org')
    erode55_blur55 = cv2.blur(cv2.erode(img, (5, 5)), (5, 5))

    _, thresh_img = cv2.threshold(erode55_blur55, 1, 255, cv2.THRESH_BINARY)
    show_image(thresh_img, 'er_bl_thresh_img')
    img_rgb = cv2.merge((img, img, thresh_img))
    show_image(img_rgb, 'org and er_bl')


def show_images(path_dir, images_numbers:list, threshold):
    path_images, _ = load_dir(path_dir)
    for i in images_numbers:
        img_path = path_dir + '/' + path_images[i]
        img = load_img_as_grey_with_threshold(img_path,threshold)
        show_image(img, f'image_{i}')


def plot_image(cam_angle_offset, laser_angle_offset, fov, img_nr, path, threshold, use_erosion_and_dilation=False,
               plot_linear_reg=False):
    path_images, _ = load_dir(path)
    image_path = path + '/' + path_images[img_nr]
    mcu_data_path = path + '/MCU_data.txt'
    dim_y = 300
    mcu_data = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)

    const_cam_laser_offset = (cam_angle_offset, laser_angle_offset)
    distance_vecs = distances_from_image(config, image_path, mcu_data[img_nr], fov, dim_y, use_erosion_and_dilation,
                                         threshold, const_cam_laser_offset, limit_estimated_angle_to_fov=None)

    d_min_max = round(max(distance_vecs[:, 0]) - min(distance_vecs[:, 0]), 3)

    if plot_linear_reg:
        model = LinearRegression().fit(distance_vecs[:, 0].reshape(-1, 1), distance_vecs[:, 1].reshape(-1, 1))
        y = model.predict(distance_vecs[:, 0].reshape(-1, 1))
        a = round(model.coef_[0][0], 3)
        b = round(model.intercept_[0], 3)
        plt.plot(distance_vecs[:, 0], y)
        title = f'diff_min_max {d_min_max}, a: {a}, b: {b}, fov: {fov}, cam_off: {const_cam_laser_offset[0]}, laser_off: {const_cam_laser_offset[1]} '
    else:
        title = f'diff_min_max {d_min_max}, fov: {fov}, cam_off: {const_cam_laser_offset[0]}, laser_off: {const_cam_laser_offset[1]} '

    plt.title(title)
    plt.scatter(distance_vecs[:, 0], distance_vecs[:, 1])
    plt.ion()
    plt.show()

    plt.pause(0.3)


def plot_row(row, img_nr, path, threshold):
    plot_image(row[1], row[2], row[0], img_nr, path, threshold)


# Find best initial parameters (FOV, Camera offset and laser offset)
def plot_print_length(optim_file_path, image_path, length_to_vertical_wall, do_plot=False):
    # define array "keys"
    fov = 0
    cam_offset = 1
    laser_offset = 2
    a = 3
    b = 4
    err = 5
    x_mean = 6
    y_mean = 7
    # eo "keys"

    l = length_to_vertical_wall
    res = np.load(optim_file_path)

    val = res[res[:, b] < (l + 20)]
    val = val[val[:, b] > (l - 20)]
    min_a = np.min(np.abs(val[:, a]))
    idx = np.abs(val[:, a]) == min_a
    row = val[idx, :]

    optimized_values = f'FOV: {row[0, fov]}, cam_offset: {row[0, cam_offset]}, laser_offset: {row[0, laser_offset]},\
        a: {row[0, a]}, b: {row[0, b]}, err: {row[0, err]}, x mean: {row[0, x_mean]}, y mean: {row[0, y_mean]}'

    # return if no plotting
    if not do_plot:
        return row[0, fov], row[0, cam_offset], row[0, laser_offset]

    print(optimized_values)

    plot_row(row[0, :], 6, image_path, threshold=50)
    plot_row(row[0, :], 13, image_path, threshold=50)

    for i in range(30):
        plot_row(row[0, :], i, image_path, threshold=50)
    input("Press [enter] to continue.")
    return row[0, fov], row[0, cam_offset], row[0, laser_offset]


# Calculate and save distances, x, z, rot z (2d + z angle)
def save_distances(image_dir, save_to_file_name, fov, cam_angle_offset, laser_angle_offset,
                   threshold, use_erosion_and_dilation,limit_estimated_angle_to_field_of_view):
    dim_y = 300
    const_cam_laser_offset = (cam_angle_offset, laser_angle_offset)
    distances = distances_from_directory(config, image_dir, fov, dim_y,
                                         use_erosion_and_dilation=use_erosion_and_dilation,
                                         threshold=threshold,
                                         const_cam_laser_offset=const_cam_laser_offset,
                                         limit_estimated_angle_to_fov=limit_estimated_angle_to_field_of_view,
                                         debug_info=True)

    np.save(save_to_file_name, distances)


# Calculate and save distances, x, y, z
def save_3d_distances(path_npy_distances_angle, output_path_3d_points, xyz_max_min_filter):
    distances_all = np.load(path_npy_distances_angle)

    x_min = xyz_max_min_filter[0][0]
    x_max = xyz_max_min_filter[0][1]
    y_min = xyz_max_min_filter[1][0]
    y_max = xyz_max_min_filter[1][1]
    z_min = xyz_max_min_filter[2][0]
    z_max = xyz_max_min_filter[2][1]

    # remove negative distances caused by wrong calculation og gamma. see readme
    idx = distances_all[:, 0] > 0
    distances = distances_all[idx]

    n_rows = len(distances)

    distances_3d = np.zeros((n_rows, 3))
    for i, dist_ang in enumerate(distances):
        v = np.array([[dist_ang[0]], [0], [dist_ang[1]]])
        R = rot_z_3d(dist_ang[2])
        d = R @ v
        distances_3d[i, :] = d[:, 0]
        if i % 10000 == 0:
            print(f'{i} of {n_rows}')

    x = distances_3d[~np.isnan(distances_3d)]
    x2 = x.reshape(-1, 3)
    x2 = x2[x2[:, 0] < x_max]
    x2 = x2[x2[:, 0] > x_min]
    x2 = x2[x2[:, 1] < y_max]
    x2 = x2[x2[:, 1] > y_min]
    x2 = x2[x2[:, 2] < z_max]
    x2 = x2[x2[:, 2] > z_min]

    distances_3d_no_nan = x2
    np.save(output_path_3d_points, distances_3d_no_nan)
    np.savetxt(output_path_3d_points + '.txt', distances_3d_no_nan, delimiter=',')


# Used for manually tuning image parameters
def tune_img(path_img, threshold):
    img = load_img_as_grey_with_threshold(path_img, threshold)
    show_image(img, 'original')
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    show_image(img_erosion, 'erosion')

    img_dilation = cv2.dilate(img, kernel, iterations=1)
    show_image(img_dilation, 'dilation')

    img_erosion_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    show_image(img_erosion_dilation, 'erosion_and_dialation')

    img_rgb = cv2.merge((img, img, img_erosion_dilation))
    show_image(img_rgb, 'original_erosion_and_dialation')

    img_rgb = cv2.merge((img, img, img_erosion))
    show_image(img_rgb, 'original_erosion')



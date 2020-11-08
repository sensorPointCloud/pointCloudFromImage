import os

from sensor.distance_from_files import load_dir, cam_angles_from_image, cam_angles_and_laser_angles_from_file, \
    distances_from_image, distances_from_directory, clean_mcu_data

import tests.test_distance_estimation

file_dir_path = os.path.dirname(os.path.realpath(__file__))

config = tests.test_distance_estimation.config
wall_dir = file_dir_path + '/../experiments/wall'
wall_dir_blk = file_dir_path + '/../experiments/wall_blk'


def test_load_dir():
    expected_image_files = 22
    expected_file_name_0 = 'Frame-1--Aux time-13847.088225--Host time-428644.5235078.bmp'
    expected_file_name_1 = 'Frame-2--Aux time-13847.690225--Host time-428645.1255832.bmp'
    expected_file_name_9 = 'Frame-10--Aux time-13852.724225--Host time-428650.1591944.bmp'
    expected_file_name_21 = 'Frame-22--Aux time-13860.222225--Host time-428657.6569954.bmp'
    path_images, path_mcu_data = load_dir(wall_dir)

    assert len(path_images) == expected_image_files
    assert expected_file_name_0 == path_images[0]
    assert expected_file_name_1 == path_images[1]
    assert expected_file_name_9 == path_images[9]
    assert expected_file_name_21 == path_images[21]

    assert path_mcu_data != None


def test_cam_angles_from_image():
    FOV = 15
    expected_number_of_angles = 300
    path_images, _ = load_dir(wall_dir)
    angles = cam_angles_from_image(wall_dir + '/' + path_images[0], FOV, use_erosion_and_dilation=False,  threshold=100)

    # all angles should be positive

    assert expected_number_of_angles == len(angles)
    assert 0 < min(angles)


def test_cam_angles_and_laser_angles_from_file():
    expected_n_frames = 22
    expected_m_per_frame = 300
    dim_y = 300

    data = cam_angles_and_laser_angles_from_file(wall_dir + '/MCU_data.txt', dim_y)

    assert len(data) == expected_n_frames
    assert len(data[0].time_s) == expected_m_per_frame
    assert len(data[10].time_s) == expected_m_per_frame

    assert data[0].time_s[0] == 7180826 / 1000
    assert data[20].time_s[2] == 7193362 / 1000


def test_distances_from_image():
    FOV = 44.73
    dim_y = 300

    min_x_distance = 1000

    path_images, _ = load_dir(wall_dir)
    image_path = wall_dir + '/' + path_images[0]
    mcu_data_path = wall_dir + '/MCU_data.txt'

    mcu_datas = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)

    distance_vecs = distances_from_image(config, image_path, mcu_datas[0], FOV, dim_y, use_erosion_and_dilation=False,
                                         threshold=100, const_cam_laser_offset=(0, 0),
                                         limit_estimated_angle_to_fov=None)
    x_min = min(distance_vecs[:, 0])

    assert x_min > min_x_distance
    assert len(distance_vecs[:, 0]) == dim_y


def test_distances_from_directory():
    FOV = 44.73
    dim_y = 300
    n_frames = 22
    n_distances = dim_y * n_frames
    min_x_distance = 1000
    max_x_distance = 2000
    min_y_distance = -500
    max_y_distance = 1000

    distance_vecs = distances_from_directory(config, wall_dir, FOV, dim_y, use_erosion_and_dilation=False,
                                             threshold=100, limit_estimated_angle_to_fov=None,
                                             const_cam_laser_offset=(0, 0))
    x_min = min(distance_vecs[:, 0])
    x_max = max(distance_vecs[:, 0])
    y_min = min(distance_vecs[:, 1])
    y_max = max(distance_vecs[:, 1])

    assert len(distance_vecs) == n_distances
    assert x_min > min_x_distance
    assert x_max < max_x_distance
    assert y_min > min_y_distance
    assert y_max < max_y_distance


def test_clean_mcu_data():
    dir_path = file_dir_path + '/../experiments/exp_with_fails'
    mcu_data_path = dir_path + '/MCU_data.txt'
    dim_y = 300
    mcu_datas_original = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)
    mcu_data = clean_mcu_data(mcu_data_path, dim_y)
    expected_errors_in_file = 9
    expected_rows = len(mcu_datas_original) - 2 * expected_errors_in_file

    assert mcu_data[0].time_s[0] == 248.664
    assert mcu_data[640].time_s[0] == 693.511
    assert mcu_data[640].yaw_angles[0] == 55.139
    assert mcu_data[640].laser_angles[0] == -8.610

    assert len(mcu_data) == expected_rows


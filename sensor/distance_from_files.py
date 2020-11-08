import os
import numpy as np

from sensor.cam_estimate_laser_angle import load_img_as_grey_with_threshold, get_contours, get_largest_contour_1xn, \
    estimate_angle_from_contour_1xn, load_img_as_grey_with_threshold_erosion_dilation
from sensor.distance_estimation import get_distance


class McuData():
    def __init__(self):
        self.cam_angles = None
        self.laser_angles = None
        self.yaw_angles = None
        self.time_s = None
        self.recieved = None  # Will contain 'frame_error' if error


def cam_angles_from_image(path, FOV, use_erosion_and_dilation, threshold):
    if use_erosion_and_dilation:
        img_grey = load_img_as_grey_with_threshold_erosion_dilation(path, threshold)
    else:
        img_grey = load_img_as_grey_with_threshold(path, threshold)

    dim_x = img_grey.shape[1]
    dim_y = img_grey.shape[0]

    angles = np.zeros(dim_y)
    for y in range(dim_y):  # TODO: Parallel

        contours = get_contours(img_grey[y:y + 1, 0:dim_x])
        c_min, c_max = get_largest_contour_1xn(contours)

        # If not contour can be found in image, there will be no angle
        if c_min == None:
            angles[y] = np.nan
        else:
            angle, _ = estimate_angle_from_contour_1xn(c_min, c_max, FOV, dim_x)
            angles[y] = -angle

    return angles


def cam_angles_and_laser_angles_from_file(path, dim_y):
    mcu_datas = []
    with open(path, mode='r') as file:  # Use file to refer to the file object
        for line in file.readlines():
            line = line.replace('[', '').split('],')
            time = line[1]
            data = line[0].split(',')
            data = np.array([float(n) / 1000 for n in data])
            data = data.reshape(int(len(data) / 4), 4)

            # Mulitiple frames are stored on the same line sent from mcu
            # this could typically be values for two frames saved on one line
            dim = len(data[:, 0])
            if dim % dim_y != 0:
                raise ValueError('Log data is missing from MCU file')
            n_dims = int(dim / dim_y)

            dim_start = 0
            dim_end = dim_y
            for _ in range(n_dims):
                info = McuData()
                info.time_s = data[dim_start:dim_end, 0]
                info.yaw_angles = data[dim_start:dim_end, 1]
                info.cam_angles = data[dim_start:dim_end, 2]
                info.laser_angles = -data[dim_start:dim_end, 3]
                info.recieved = time

                mcu_datas.append(info)

                dim_start = dim_end
                dim_end += dim_y

    return mcu_datas


# cam_angles_and_laser_angles_from_file('./experiments/wall/MCU_data.txt', dim_y=300)

def distances_from_image(config, image_path, mcu_data, FOV, dim_y, use_erosion_and_dilation,
                         threshold, const_cam_laser_offset,
                         limit_estimated_angle_to_fov):  # const_cam_laser_offset=(0, 0)
    """


    :param config:
    :param image_path:
    :param mcu_data:
    :float FOV:
    :int dim_y:
    :bool use_erosion_and_dilation:
    :int threshold:
    :param const_cam_laser_offset:
    :float limit_estimated_angle_to_fov:
        limits the angles used to calculate distance. If estimated angle is found outside of
        limit_estimated_angle_to_fov, then the angle is discarded. This removes invalid distances caused by laser
        captured in the edge of the image.
    :return:
    """

    angles = cam_angles_from_image(image_path, FOV, use_erosion_and_dilation, threshold)

    # limit_estimated_angle_to_fov = 40
    if limit_estimated_angle_to_fov != None:
        # np.warnings.filterwarnings('ignore')
        idx_outside_bounds = np.abs(angles) > (limit_estimated_angle_to_fov / 2)
        angles[idx_outside_bounds] = np.nan

    if len(angles) != dim_y:
        raise ValueError('image dimension and dim_y does not match')

    distances = np.zeros((dim_y, 3))
    # TODO: Parallel
    for i in range(dim_y):
        internal_laser_angle = angles[i]
        cam_angle = mcu_data.cam_angles[i]
        laser_angle = mcu_data.laser_angles[i]
        yaw_angle = mcu_data.yaw_angles[i]

        # Only calculate distance if the laser can be found in the image
        if internal_laser_angle is None:
            distances[i][0] = None
            distances[i][1] = None
            distances[i][2] = None
        else:
            # Add offset
            cam_angle += const_cam_laser_offset[0]
            laser_angle += const_cam_laser_offset[1]
            distance = get_distance(config, cam_angle, laser_angle, internal_laser_angle)
            distances[i][0] = distance[0]
            distances[i][1] = distance[1]
            distances[i][2] = yaw_angle  # z angle

    return distances


def distances_from_directory(config, dir_path, FOV, dim_y, use_erosion_and_dilation, threshold, const_cam_laser_offset,
                             limit_estimated_angle_to_fov,
                             debug_info=False):  # const_cam_laser_offset=(0, 0)
    path_images, _ = load_dir(dir_path)
    n_images = len(path_images)
    mcu_data_path = dir_path + '/MCU_data.txt'

    mcu_datas = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)

    all_distance_vecs = np.zeros((dim_y * n_images, 3))
    dim_start = 0
    dim_end = dim_y
    for i in range(n_images):  # TODO PARFOR

        image_path = dir_path + '/' + path_images[i]
        distance_vecs = distances_from_image(config, image_path, mcu_datas[i], FOV, dim_y, use_erosion_and_dilation,
                                             threshold, const_cam_laser_offset, limit_estimated_angle_to_fov)
        all_distance_vecs[dim_start:dim_end, :] = distance_vecs
        dim_start = dim_end
        dim_end += dim_y
        if debug_info:
            print(f'{i} of {n_images}')

    return all_distance_vecs


def clean_mcu_data(path, dim_y=300):
    mcu_datas = cam_angles_and_laser_angles_from_file(path, dim_y)
    if mcu_datas[0].recieved.__contains__('frame_error'):
        print(
            'There might occur large syncing errors, because the initial frame is not synced. It contains data frame errors')
    mcu_datas_clean = list(filter(lambda x: not x.recieved.__contains__('frame_error'), mcu_datas))
    return mcu_datas_clean


#
# def match_images_with_data(dir_path, dim_y=300):
#     path_images, _ = load_dir(dir_path)
#     n_images = len(path_images)
#     mcu_data_path = dir_path + '/MCU_data.txt'
#
#     mcu_datas = cam_angles_and_laser_angles_from_file(mcu_data_path, dim_y)
#
#     matched_images, matched_data = 0, 0
#     return matched_images, matched_data


def load_dir(path, sort=True):
    (_, _, filenames) = next(os.walk(path))
    images_path = [i for i in filenames if os.path.splitext(i)[1] == '.bmp']

    if sort:
        # Sort images windows style s.t. Frame 1, Frame 2... (and not Frame 1, Frame 10....)
        images_path.sort(key=lambda f: int(f.split('Frame-')[1].split('--Aux')[0]))
    mcu_data = 'MCU_data.txt'
    return images_path, mcu_data

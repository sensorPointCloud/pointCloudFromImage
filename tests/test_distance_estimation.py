from sensor.distance_estimation import get_distance, rot_z_2d, get_camera_pos, get_laser_pos, \
    get_cam_laser_vector_and_angle, get_vector_angle_absolute
from sensor.distance_estimation import get_laser_cam_vector_and_angle, get_cam_laser_and_laser_cam_vector_and_angle, \
    get_camera_global_laser_angle
from sensor.distance_estimation import get_epsilon_beta_gamma_alpha_cam_laser, get_alpha_omega, rot_z_3d

from sensor.distance_estimation import get_camera_pos_mpmath

import numpy as np

config = {
    'camera_mount_x': 315,
    'camera_mount_y': 46,
    # 'laser_mount_x': 0,
    'laser_mount_y': 25,
    'center_center_offset_y': 259,
}


def test_rot_z_2d():
    rot90 = rot_z_2d(90)

    # x should be y only
    assert round(rot90[0][0], 4) == 0
    assert round(rot90[1][0], 4) == 1

    # y should be -x only
    assert round(rot90[0][1], 4) == -1
    assert round(rot90[1][1], 4) == 0

    # Rotate camera, check vector @ autocad test 2
    R = rot_z_2d(6)
    v = np.array([[315], [46]])
    xy = R @ v  # @ is matrix multiplication

    x_expected = 308.4661
    y_expected = 78.6745
    assert round(xy[0][0], 4) == x_expected
    assert round(xy[1][0], 4) == y_expected


def test_rot_z_3d():
    rot90 = rot_z_3d(90)

    # x should be y only
    assert round(rot90[0][0], 4) == 0
    assert round(rot90[1][0], 4) == 1

    # y should be -x only
    assert round(rot90[0][1], 4) == -1
    assert round(rot90[1][1], 4) == 0

    assert rot90[2][0] == 0
    assert rot90[2][1] == 0
    assert rot90[2][2] == 1

    # Rotate camera, check vector @ autocad test 2
    R = rot_z_3d(6)
    v = np.array([[315], [46], [0]])
    xy = R @ v  # @ is matrix multiplication

    x_expected = 308.4661
    y_expected = 78.6745
    assert round(xy[0][0], 4) == x_expected
    assert round(xy[1][0], 4) == y_expected
    assert xy[2][0] == v[2]


# autocad test 2
def test_get_get_camera_pos():
    angle = 6

    pos = get_camera_pos(config, angle)
    x_expected = 308.4661
    y_expected = 78.6745
    assert round(pos[0][0], 4) == x_expected
    assert round(pos[1][0], 4) == y_expected


# autocad test 6
def test_get_camera_pos_verify():
    angle = 2.294796
    x_expected = 312.9055
    y_expected = 58.5760
    pos = get_camera_pos_mpmath(config, angle)

    assert round(pos[0], 4) == x_expected
    assert round(pos[1], 4) == y_expected


# autocad test 2
def test_get_laser_pos():
    angle = -11
    x_expected = 4.7702
    y_expected = 283.5407

    pos = get_laser_pos(config, angle)
    assert round(pos[0][0], 4) == x_expected
    assert round(pos[1][0], 4) == y_expected


# autocad test 6
def test_get_laser_pos_verify():
    angle = -20.030
    x_expected = 8.5628
    y_expected = 282.4878

    pos = get_laser_pos(config, angle)
    assert round(pos[0][0], 4) == x_expected
    assert round(pos[1][0], 4) == y_expected


# autocad test 3
def test_get_vector_angle_absolute():
    x = -303.6959
    y = 204.8662
    angle_expected = 145.997

    vector = np.array([[x], [y]])
    angle = get_vector_angle_absolute(vector)

    assert round(angle, 3) == angle_expected


# autocad test 3
def test_get_cam_laser_vector_and_angle():
    laser_angle = -11
    cam_angle = 6
    x_expected = -303.6959
    y_expected = 204.8662
    angle_expected = 145.997

    v, angle = get_cam_laser_vector_and_angle(config, cam_angle, laser_angle)

    assert round(v[0][0], 4) == x_expected
    assert round(v[1][0], 4) == y_expected
    assert round(angle, 3) == angle_expected


# autocad test 3
def test_get_laser_cam_vector_and_angle():
    laser_angle = -11
    cam_angle = 6
    x_expected = 303.6959
    y_expected = -204.8662
    angle_expected = 325.997
    v, angle = get_laser_cam_vector_and_angle(config, cam_angle, laser_angle)

    assert round(v[0][0], 4) == x_expected
    assert round(v[1][0], 4) == y_expected
    assert round(angle, 3) == angle_expected


# autocad test 3
def test_get_cam_laser_and_laser_cam_vector_and_angle():
    laser_angle = -11
    cam_angle = 6

    x_cam_laser_expected = -303.6959
    y_cam_laser_expected = 204.8662
    angle_cam_laser_expected = 145.997

    x_laser_cam_expected = 303.6959
    y_laser_cam_expected = -204.8662
    angle_laser_cam_expected = 325.997

    cam_laser, cam_laser_angle, laser_cam, laser_cam_angle = get_cam_laser_and_laser_cam_vector_and_angle(config,
                                                                                                          cam_angle,
                                                                                                          laser_angle)

    assert round(cam_laser[0][0], 4) == x_cam_laser_expected
    assert round(cam_laser[1][0], 4) == y_cam_laser_expected
    assert round(cam_laser_angle, 3) == angle_cam_laser_expected

    assert round(laser_cam[0][0], 4) == x_laser_cam_expected
    assert round(laser_cam[1][0], 4) == y_laser_cam_expected
    assert round(laser_cam_angle, 3) == angle_laser_cam_expected


# autocad test 3
def test_get_camera_global_laser_angle():
    cam_angle = 6
    internal_laser_angle = -10.916
    angle_expected = 355.084

    angle = get_camera_global_laser_angle(cam_angle, internal_laser_angle)

    assert round(angle, 3) == angle_expected


# autocad test 4
def test_get_alpha_beta_gamma():
    laser_angle = -11
    cam_angle = 6
    internal_laser_angle = -10.916
    epsilon_expected = 23.003
    beta_expected = 150.913
    gamma_expected = 6.084
    alpha_expected = 355.084
    x_cam_laser_expected = -303.6959
    y_cam_laser_expected = 204.8662

    epsilon, beta, gamma, alpha, cam_laser = get_epsilon_beta_gamma_alpha_cam_laser(config, cam_angle, laser_angle,
                                                                                    internal_laser_angle)

    assert round(epsilon, 3) == epsilon_expected
    assert round(beta, 3) == beta_expected
    assert round(gamma, 3) == gamma_expected
    assert round(alpha, 3) == alpha_expected

    assert round(cam_laser[0][0], 4) == x_cam_laser_expected
    assert round(cam_laser[1][0], 4) == y_cam_laser_expected


# autocad test 4
def test_get_alpha_omega():
    alpha_expected = 355.084
    omega_expected = 1350.6895
    laser_angle = -11
    cam_angle = 6
    internal_laser_angle = -10.916016

    alpha, omega = get_alpha_omega(config, cam_angle, laser_angle, internal_laser_angle)

    assert round(alpha, 3) == alpha_expected
    assert round(omega, 4) == omega_expected


# autocad test 5
def test_get_alpha_omega_verify():
    alpha_expected = 344.339
    omega_expected = 1392.993  # 2
    laser_angle = -20.030053
    cam_angle = 2.294796
    internal_laser_angle = -17.955472

    alpha, omega = get_alpha_omega(config, cam_angle, laser_angle, internal_laser_angle)

    assert round(alpha, 3) == alpha_expected
    assert round(omega, 3) == omega_expected


# autocad test 4
def test_get_distance():
    laser_angle = -11
    cam_angle = 6
    internal_laser_angle = -10.916016

    expected_distance_x = 1654.187
    expected_distance_y = -37.073

    distance = get_distance(config, cam_angle, laser_angle, internal_laser_angle)

    assert round(distance[0][0], 3) == expected_distance_x
    assert round(distance[1][0], 3) == expected_distance_y


# autocad test 5
def test_get_distance_verify():
    laser_angle = -20.030053
    cam_angle = 2.294796
    internal_laser_angle = -17.955472

    expected_distance_x = 1654.187
    expected_distance_y = -317.448

    distance = get_distance(config, cam_angle, laser_angle, internal_laser_angle)

    assert round(distance[0][0], 3) == expected_distance_x
    assert round(distance[1][0], 3) == expected_distance_y

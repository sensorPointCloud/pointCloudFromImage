import numpy as np
import mpmath as mp


def rot_z_2d_mpmath(angle_degrees):
    theta = mp.radians(angle_degrees)
    c, s = mp.cos(theta), mp.sin(theta)
    return mp.matrix([[c, -s], [s, c]])


def get_camera_pos_mpmath(config, angle):
    x = config['camera_mount_x']
    y = config['camera_mount_y']

    R = rot_z_2d_mpmath(angle)
    v = mp.matrix([[x], [y]])
    xy = R * v
    return xy


def rot_z_2d(angle_degrees):
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def rot_z_3d(angle_degrees):
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)

    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def get_camera_pos(config, angle):
    x = config['camera_mount_x']
    y = config['camera_mount_y']

    R = rot_z_2d(angle)
    v = np.array([[x], [y]])
    xy = R @ v  # @ is matrix multiplication
    return xy


# Returns position with respect to origin ( rotational axis of camera )
def get_laser_pos(config, angle):
    x = 0
    y = config['laser_mount_y']
    oy = config['center_center_offset_y']

    R = rot_z_2d(angle)
    v = np.array([[x], [y]])
    xy = R @ v  # @ is matrix multiplication
    xy[1][0] += oy
    return xy


# Returns vector starting at camera center, ending at laser center and absolute vector angle
def get_cam_laser_vector_and_angle(config, cam_angle, laser_angle):
    cam_pos = get_camera_pos(config, cam_angle)
    laser_pos = get_laser_pos(config, laser_angle)

    cam_laser = laser_pos - cam_pos
    angle = get_vector_angle_absolute(cam_laser)

    return cam_laser, angle


# Returns vector starting at laser center, ending at camera center and absolute vector angle
def get_laser_cam_vector_and_angle(config, cam_angle, laser_angle):
    cam_pos = get_camera_pos(config, cam_angle)
    laser_pos = get_laser_pos(config, laser_angle)

    laser_cam = cam_pos - laser_pos
    angle = get_vector_angle_absolute(laser_cam)

    return laser_cam, angle


# This function is only implemented to increase speed of calculations
# Returns vector starting at camera center, ending at laser center and absolute vector angle
# and returns vector starting at laser center, ending at camera center and absolute vector angle
def get_cam_laser_and_laser_cam_vector_and_angle(config, cam_angle, laser_angle):
    cam_pos = get_camera_pos(config, cam_angle)
    laser_pos = get_laser_pos(config, laser_angle)

    cam_laser = laser_pos - cam_pos
    cam_laser_angle = get_vector_angle_absolute(cam_laser)

    laser_cam = cam_pos - laser_pos

    # Can improve the following line, laser_cam is 180deg offset from cam_laser
    laser_cam_angle = get_vector_angle_absolute(laser_cam)

    return cam_laser, cam_laser_angle, laser_cam, laser_cam_angle


# Gives the angle from 0 to 360 degrees
def get_vector_angle_absolute(vector):
    # np.arctan2(y,x)
    x = vector[0][0]
    y = vector[1][0]
    rads = np.arctan2(y, x)
    rads = rads if rads > 0 else np.pi * 2 + rads

    return np.degrees(rads)


# Global laser angle as seen from camera
def get_camera_global_laser_angle(cam_angle, internal_laser_angle):
    angle = cam_angle + internal_laser_angle
    angle = angle if angle > 0 else 360.0 + angle
    return angle


# maybe not needed
def get_epsilon_beta_gamma_alpha_cam_laser(config, cam_angle, laser_angle, internal_laser_angle):
    laser_angle_absolute = laser_angle if laser_angle > 0 else 360.0 + laser_angle
    cam_laser, cam_laser_angle, _, laser_cam_angle = get_cam_laser_and_laser_cam_vector_and_angle(config, cam_angle,
                                                                                                  laser_angle)

    epsilon = laser_angle_absolute - laser_cam_angle

    alpha = get_camera_global_laser_angle(cam_angle, internal_laser_angle)  # global_laser_seen_from_cam
    beta = 360.0 - (alpha - cam_laser_angle)
    #beta = 360.0-alpha+cam_laser_angle
    gamma = 180.0 - epsilon - beta

    return epsilon, beta, gamma, alpha, cam_laser


def get_alpha_omega(config, cam_angle, laser_angle, internal_laser_angle):
    epsilon, _, gamma, alpha, cam_laser = get_epsilon_beta_gamma_alpha_cam_laser(config, cam_angle, laser_angle,
                                                                                 internal_laser_angle)
    a = np.linalg.norm(cam_laser)
    # a = np.sqrt(cam_laser[0][0]**2 + cam_laser[1][0]**2)
    A = np.sin(np.deg2rad(gamma))
    B = np.sin(np.deg2rad(epsilon))
    omega = B * a / A

    return alpha, omega


# Returns vector from camera center of rotation to laser
def get_distance(config, cam_angle, laser_angle, internal_laser_angle):
    alpha_deg, omega = get_alpha_omega(config, cam_angle, laser_angle, internal_laser_angle)

    # This has already been calculated when calling get_alpha_omega, but is not returned. Small optimization possible, but might ruin sensor
    camera_pos = get_camera_pos(config, cam_angle)

    alpha_rad = np.radians(alpha_deg)
    ao_vec = np.array([[omega * np.cos(alpha_rad)], [omega * np.sin(alpha_rad)]])

    distance = camera_pos + ao_vec

    return distance

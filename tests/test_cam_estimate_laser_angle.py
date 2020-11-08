import os

from sensor.cam_estimate_laser_angle import load_img, load_img_as_grey_with_threshold, get_contours, \
    get_largest_contour_1xn, estimate_angle_from_contour_1xn

file_dir_path = os.path.dirname(os.path.realpath(__file__))
img_wall = file_dir_path + '/../images/wall.bmp'
img_100px_white_99_100_rest_black = file_dir_path + '/../images/100px_white_99_100_rest_black.bmp'
img_3px_white_2_3_rest_black = file_dir_path + '/../images/3px_white_2_3_rest_black.bmp'
img_3px_white_1_2_rest_black = file_dir_path + '/../images/3px_white_1_2_rest_black.bmp'
img_4px_white_2_3_rest_black = file_dir_path + '/../images/4px_white_2_3_rest_black.bmp'


def test_load_image():
    expected_height = 300
    expected_width = 4096
    img = load_img(img_wall)

    height = img.shape[0]
    width = img.shape[1]

    assert height == expected_height
    assert width == expected_width


def test_load_image_as_grey_with_treshold():
    thresh = 100
    img = load_img_as_grey_with_threshold(img_wall, thresh)
    assert len(img.shape) == 2


def test_get_contours():
    thresh = 100
    img_grey = load_img_as_grey_with_threshold(img_wall, thresh)

    img_first_row = img_grey[0:1, 0:img_grey.shape[1]]  # [y:y+h, x:x+w]
    contours = get_contours(img_first_row)

    assert len(contours) == 4  # img_wall has 4 contours in first row


def test_get_largest_contour_1xn():
    expected_min = 2260
    expected_max = 2299

    img_grey = load_img_as_grey_with_threshold(img_wall, threshold=100)
    img_first_row = img_grey[0:1, 0:img_grey.shape[1]]  # [y:y+h, x:x+w]
    contours = get_contours(img_first_row)

    c_min, c_max = get_largest_contour_1xn(contours)
    assert c_min == expected_min
    assert c_max == expected_max


# autocad test 7
def test_estimate_angle_from_contour():
    FOV = 30
    expected_angle = -5.103909
    expected_percent = 66.6667

    img_grey = load_img_as_grey_with_threshold(img_3px_white_2_3_rest_black, threshold=100)
    dim_x = img_grey.shape[1]
    contours = get_contours(img_grey)
    c_min, c_max = get_largest_contour_1xn(contours)

    angle, percent = estimate_angle_from_contour_1xn(c_min, c_max, FOV, dim_x)

    assert round(percent, 4) == expected_percent
    assert round(angle, 6) == expected_angle


# autocad test 7
def test_estimate_angle_from_contour_verify():
    FOV = 30
    expected_angle = 5.103909
    expected_percent = 33.3333

    img_grey = load_img_as_grey_with_threshold(img_3px_white_1_2_rest_black, threshold=100)
    dim_x = img_grey.shape[1]
    contours = get_contours(img_grey)
    c_min, c_max = get_largest_contour_1xn(contours)

    angle, percent = estimate_angle_from_contour_1xn(c_min, c_max, FOV, dim_x)

    assert round(percent, 4) == expected_percent
    assert round(angle, 6) == expected_angle


# autocad test 8
def test_estimate_angle_from_contour_verify2():
    FOV = 50
    expected_angle = 0
    expected_percent = 50.0

    img_grey = load_img_as_grey_with_threshold(img_4px_white_2_3_rest_black, threshold=100)
    dim_x = img_grey.shape[1]
    contours = get_contours(img_grey)
    c_min, c_max = get_largest_contour_1xn(contours)

    angle, percent = estimate_angle_from_contour_1xn(c_min, c_max, FOV, dim_x)

    assert round(percent, 4) == expected_percent
    assert round(angle, 6) == expected_angle


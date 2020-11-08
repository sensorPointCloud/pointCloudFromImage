import cv2
import numpy as np


def load_img(path):
    return cv2.imread(path)


def load_img_as_grey_with_threshold(path, threshold):
    img = load_img(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh_img

def load_img_as_grey_with_threshold_erosion_dilation(path, threshold):
    img = load_img_as_grey_with_threshold(path, threshold)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_erosion_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    return img_erosion_dilation



def show_image(img, name='wall'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # img = cv2.resize(img, (1024, 75))
    cv2.imshow(name, img)
    cv2.waitKey(0)


def get_contours(grey_img):
    contours, hierarchy = cv2.findContours(grey_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# both numbers are inclusive
def get_largest_contour_1xn(contours):
    c_min, c_max, c_diff = None, None, 0
    for c in contours:
        # exclude 1 pixel contours
        if len(c) == 1:
            continue

        # x1: c[0][0][0], x2: c[1][0][0]
        diff = c[1][0][0] - c[0][0][0]

        # check if larger contour is found
        if (diff > c_diff):
            c_diff = diff
            c_min = c[0][0][0]
            c_max = c[1][0][0]

    return c_min, c_max


# see autocad test 7
# returns angle from center range e.g. 7.5 degrees to -7.5 degrees
# left side of picture is positive
def estimate_angle_from_contour_1xn(c_min, c_max, FOV_deg, dim_x):
    px_center = (c_min + c_max + 1) / 2  # (c_min+0.5 + c_max+0.5)/2
    FOV_rad = np.radians(FOV_deg)

    frac = px_center / dim_x

    hk = (dim_x / 2) / np.tan(FOV_rad / 2)

    frac_center = (1 - frac) - 0.5

    mk = dim_x * frac_center

    angle_deg = np.degrees(np.arctan(mk / hk))

    return angle_deg, frac * 100


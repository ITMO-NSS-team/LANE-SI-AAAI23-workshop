from copy import deepcopy

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )
    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def return_interpolated_field(image):
    image[image == 0] = np.nan
    mask = np.isnan(image)
    image_smoothed = interpolate_missing_pixels(image, mask, 'nearest')
    image_smoothed = gaussian_filter(image_smoothed, sigma=2.5)
    image[np.isnan(image)] = image_smoothed[np.isnan(image)]
    return image


def rotate(image):
    image = np.fliplr(image)
    image = np.rot90(image, 2)
    return image


def binary(matrix, threshold=0.8):
    matrix_copy = deepcopy(matrix)
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    return matrix_copy


def gradate(raw_array):
    array = deepcopy(raw_array)
    array[np.where((array > 0.9))] = 9
    array[np.where((array > 0.8) & (array <= 0.9))] = 8
    array[np.where((array > 0.7) & (array <= 0.8))] = 7
    array[np.where((array > 0.6) & (array <= 0.7))] = 6
    array[np.where((array > 0.4) & (array <= 0.6))] = 4
    array[np.where((array > 0.3) & (array <= 0.4))] = 3
    array[np.where((array > 0.2) & (array <= 0.3))] = 2
    array[np.where((array > 0.1) & (array <= 0.2))] = 1
    array[array <= 0.1] = 0
    return array
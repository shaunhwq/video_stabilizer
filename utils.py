import os
from typing import Tuple, Optional, List

import cv2
import numpy as np


def is_same_extension(file_path: str, extension: str):
    name, ext = os.path.splitext(file_path)
    return ext == extension


def get_subwindow(image: np.array, window_name: str = "Select Rectangle") -> Optional[List[Tuple[float]]]:
    """
    Prompts user to select 2 points on an image

    :param image: Image to select the points on
    :param window_name: Name of the prompt window
    :returns: selected 2 points
    """
    selected_pts = []

    def on_mouse_clicked(event, x, y, flags, param):
        nonlocal selected_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_pts) < 2:
                selected_pts.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_pts = []

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse_clicked)

    while True:
        copied = image.copy()

        cv2.putText(copied, "Select 2 points", (25, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        for point in selected_pts:
            cv2.circle(copied, point, 5, (255, 255, 255), -1)
        if len(selected_pts) == 2:
            cv2.rectangle(copied, selected_pts[0], selected_pts[1], (255, 255, 255), 2)
        cv2.imshow(window_name, copied)
        key = cv2.waitKey(1)
        if key & 255 == 32:
            cv2.destroyWindow(window_name)
            return selected_pts
        elif key & 255 == 27:
            break

    cv2.destroyWindow(window_name)
    return None


def scale_homography_matrix(homography_matrix, scale_factor):
    homography_matrix[0, 2] *= scale_factor
    homography_matrix[1, 2] *= scale_factor
    homography_matrix[2, 0] /= scale_factor
    homography_matrix[2, 1] /= scale_factor
    return homography_matrix


def pyr_scale_image(image: np.array, scale_power: int) -> np.array:
    """
    Uses cv2.pyrDown or cv2.pyrUp depending on provided scale

    :param image: image you want to scale
    :param scale_power: number of times to apply scaling operator. +ve scales up, -ve scales down.
    :returns: scaled image
    """
    if scale_power == 0:
        return image

    operator = cv2.pyrUp if scale_power > 0 else cv2.pyrDown
    for i in range(abs(scale_power)):
        image = operator(image)

    return image

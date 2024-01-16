import os
import sys
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


class SiftHelper:
    def __init__(self):
        """
        Sift keypoint detection followed by FLANN matching, with hardcoded parameters.
        """
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, image: np.array):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return self.sift.detectAndCompute(gray_image, None)

    def match_points(self, kp1, desc1, kp2, desc2):
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(
            indexParams=dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            searchParams=dict(checks=50)
        )
        matches = flann.knnMatch(desc1, desc2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            return M

        return None

    def backwarp_image(self, img2, img1_shape, homography_matrix):
        h, w = img1_shape
        warped_image = cv2.warpPerspective(img2, homography_matrix, (w, h))
        return warped_image


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


def is_same_extension(file_path: str, extension: str):
    name, ext = os.path.splitext(file_path)
    return ext == extension


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to input video", required=True)
    parser.add_argument("-o", "--output_path", help="Desired output path", required=True)
    args = parser.parse_args()

    # File path checks
    assert os.path.exists(args.input_path)
    assert is_same_extension(args.input_path, ".mp4")
    assert is_same_extension(args.output_path, ".mp4")

    # File IO
    cap = cv2.VideoCapture(args.input_path)
    writer = None
    if not cap.isOpened():
        sys.exit("Unable to open the input video")

    # Progress tracking
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc="Extracting keypoints...")

    # Sift related
    sift_helper = SiftHelper()
    kp_1, desc_1 = None, None
    cropped_shape = None

    # For display
    cropped_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scaled_frame = cv2.pyrDown(frame)

        if kp_1 is None:
            rect_pts = get_subwindow(scaled_frame)
            if rect_pts is None:
                sys.exit("User terminated the program.")

            # Get even sized crop (easier time later since we can just x2 without problems)
            (x1, y1), (x2, y2) = rect_pts
            if (x2 - x1) % 2 != 0:
                x2 -= 1
            if (y2 - y1) % 2 != 0:
                y2 -= 1

            cropped_img = scaled_frame[y1: y2, x1: x2, ...]
            cropped_shape = cropped_img.shape

            kp_1, desc_1 = sift_helper.detect_and_compute(cropped_img)

            # Create instance of writer since now we know what the desired output size is
            writer = cv2.VideoWriter(
                args.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (cropped_shape[1] * 2, cropped_shape[0] * 2),  # Because we pyrDown earlier
            )

            # So we can concatenate with normal sized image later
            cropped_img = cv2.pyrUp(cropped_img)
            continue

        kp_n, desc_n = sift_helper.detect_and_compute(scaled_frame)

        h_matrix = sift_helper.match_points(kp_1, desc_1, kp_n, desc_n)

        # To account for usage of scaled image
        h_matrix[0, 2] *= 2
        h_matrix[1, 2] *= 2
        h_matrix[2, 0] /= 2
        h_matrix[2, 1] /= 2

        warped_region = sift_helper.backwarp_image(frame, (cropped_shape[0] * 2, cropped_shape[1] * 2), h_matrix)

        writer.write(warped_region)

        pbar.update(1)

        cv2.imshow("Warped", np.hstack([cropped_img, warped_region]))
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

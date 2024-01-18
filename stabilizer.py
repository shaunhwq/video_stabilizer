import os
import sys
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import utils


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to input video", required=True, type=str)
    parser.add_argument("-o", "--output_path", help="Desired output path", required=True, type=str)
    parser.add_argument("-s", "--scale_power", help="For improving computational speed via downsizing matching images.", default=0, type=int)
    args = parser.parse_args()

    # Parameter checks
    assert os.path.exists(args.input_path)
    assert utils.is_same_extension(args.input_path, ".mp4")
    assert utils.is_same_extension(args.output_path, ".mp4")
    assert args.scale_power >= 0 and isinstance(args.scale_power, int)

    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

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

        scaled_frame = frame
        scaled_frame = utils.pyr_scale_image(scaled_frame, -1 * args.scale_power)

        if kp_1 is None:
            rect_pts = utils.get_subwindow(scaled_frame)
            if rect_pts is None:
                sys.exit("User terminated the program.")

            # Crop selected region
            (x1, y1), (x2, y2) = rect_pts
            cropped_img = scaled_frame[y1: y2, x1: x2, ...]
            cropped_shape = cropped_img.shape

            # Get keypoints and descriptors
            kp_1, desc_1 = sift_helper.detect_and_compute(cropped_img)

            # Scale cropped back to original size frame's scale for getting the size of warped image later.
            cropped_img = utils.pyr_scale_image(cropped_img, args.scale_power)

            # Create instance of writer since now we know what the desired output size is
            temp_out_video_path = os.path.join(temp_dir, os.path.basename(args.output_path))
            writer = cv2.VideoWriter(
                temp_out_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                cropped_img.shape[:2][::-1],
            )
            continue

        kp_n, desc_n = sift_helper.detect_and_compute(scaled_frame)

        h_matrix = sift_helper.match_points(kp_1, desc_1, kp_n, desc_n)

        if h_matrix is None:
            writer.release()
            os.remove(temp_out_video_path)
            sys.exit("Unable to predict homography matrix, please try again with another crop with more features")

        # To account for usage of scaled image
        h_matrix = utils.scale_homography_matrix(h_matrix, 2 ** args.scale_power)

        warped_region = sift_helper.backwarp_image(frame, cropped_img.shape[:2], h_matrix)
        writer.write(warped_region)

        pbar.update(1)

        #scaled_cropped_frame = scaled_frame[y1: y2, x1: x2, ...]
        #print(scaled_frame.shape, scaled_cropped_frame.shape, warped_region.shape)
        cropped_frame = scaled_frame[y1: y2, x1: x2, ...]
        cropped_frame = utils.pyr_scale_image(cropped_frame, args.scale_power)
        cv2.imshow("Cropped", np.hstack([cropped_frame, cropped_img, warped_region]))

        key = cv2.waitKey(1)
        if key & 255 == 27:
            writer.release()
            os.remove(temp_out_video_path)
            sys.exit("User terminated the program.")

    cv2.destroyAllWindows()
    cap.release()
    writer.release()

    # Extract and copy audio over to the source image
    os.system(f"ffmpeg -y -loglevel error -i {args.input_path} -i {temp_out_video_path} -filter_complex \"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[audio]\" -map 1:v -map [audio] -c:v copy -c:a aac -strict experimental {args.output_path}")
    os.remove(temp_out_video_path)

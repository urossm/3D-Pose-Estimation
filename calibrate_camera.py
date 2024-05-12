import argparse
import numpy as np
import cv2
import os
from utils import rotate_image


def calibrate(images_path, file_name, widht, height, cell_size, rotate):
    all_files = list()
    for f in os.listdir(images_path):
        if os.path.splitext(f)[-1] == ".png" or os.path.splitext(f)[-1] == ".jpg":
            all_files.append(os.path.join(images_path, f))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    single_object_points = np.zeros((widht * height, 3), np.float32)

    # prepare object points, like (0, 0, 0), (30, 0, 0), (60, 0, 0), (90, 0, 0), (120, 0, 0), (0, 30, 0) ...
    # ... ((w-1)*30, (h-1)*30, 0)

    for i in range(height):
        for j in range(widht):
            single_object_points[i * widht + j, :] = np.array([j * cell_size, i * cell_size, 0], dtype=np.float32)

    object_points = []  # 3d point in real world space
    imgage_points = []

    for image in all_files:
        img = cv2.imread(image)
        img = rotate_image(img, rotate)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (widht, height), None)
        if ret:
            object_points.append(single_object_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgage_points.append(corners2)

            img = cv2.drawChessboardCorners(img, (widht, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, imgage_points, gray.shape[::-1], None, None)
    mean_error = 0

    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            single_object_points, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgage_points[i],
                         imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(object_points)))

    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist)
    file.write("intrinsic", mtx)
    file.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calibrate camera and save intrinsix matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p", "--path", help="Path for png/jpg images for camera calibration")
    parser.add_argument("-f", "--file_name",
                        help="Calibration file save path")
    parser.add_argument("-c", "--cell_size",
                        help="Cell size in millimeters", default=60)
    parser.add_argument("-g", "--grid_size",
                        help="Grid size", default="8x5")
    parser.add_argument("-r", "--rotate",
                        help="Image rotation", default=0, choices={0, 90, -90, 180}, type=int)

    options = parser.parse_args()
    (w, h) = options.grid_size.split('x')
    calibrate(options.path, options.file_name,
              int(w), int(h), options.cell_size,
              options.rotate)

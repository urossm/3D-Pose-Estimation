import argparse

import numpy as np
import cv2
import os
from utils import rotate_image


def draw_axis(img, corners, imgpts):

    corner = tuple(corners[0].ravel())

    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

    return img


def detect_camera_position(images_path, calib_file, file_name, widht, height, cell_size, rotate):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeffs = fs.getNode('distortion').mat()

    all_files = list()
    for f in os.listdir(images_path):
        if os.path.splitext(f)[-1] == ".png" or os.path.splitext(f)[-1] == ".jpg":
            all_files.append(os.path.join(images_path, f))

    object_points = np.zeros((widht*height, 3), np.float32)

    for i in range(height):
        for j in range(widht):
            object_points[i*widht + j, :] = np.array([j*cell_size, i*cell_size, 0], dtype=np.float32)

    all_rvec = np.zeros((3,0), dtype=np.float32)
    all_tvec = np.zeros((3,0), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in all_files:
        img = cv2.imread(image)
        img = rotate_image(img, rotate)

        undistorted = cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (widht, height), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            ret, rvec, tvec, _ = cv2.solvePnPRansac(object_points, corners2, cam_mat, None)
            if ret:
                all_rvec = np.hstack((all_rvec, rvec))
                all_tvec = np.hstack((all_tvec, tvec))

    rvec = np.mean(all_rvec, axis=1).reshape(3,1)
    tvec = np.mean(all_tvec, axis=1).reshape(3,1)
    cv2.namedWindow('Undistorted', cv2.WINDOW_FREERATIO)
    rot_mat, _ = cv2.Rodrigues(rvec)

    axis = np.float32([[5*cell_size, 0, 0], [0, 5*cell_size, 0], [0, 0, -5*cell_size]]).reshape(-1, 3)

    for image in all_files:
        img = cv2.imread(image)
        img = rotate_image(img, rotate)
        undistorted =  cv2.undistort(img, cam_mat, dist_coeffs, None, cam_mat)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cam_mat, dist_coeffs)

        undistorted = draw_axis(undistorted, corners2, imgpts)
        undistorted = rotate_image(undistorted, -90)
        cv2.imshow('Undistorted', undistorted)

        cv2.waitKey(200)

    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist_coeffs)
    file.write("intrinsic", cam_mat)
    file.write("Rot", rot_mat)
    file.write("Trans", tvec)
    file.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera position and extrinsic matrix detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p", "--path", help="Path for png/jpg images for camera position detection")
    parser.add_argument("-b", "--calib_file",
                        help="Calibration file path")
    parser.add_argument("-f", "--file_name",
                        help="Calibration file save path")
    parser.add_argument("-c", "--cell_size",
                        help="Cell size", default=60)
    parser.add_argument("-g", "--grid_size",
                        help="Grid size", default="8x5")
    parser.add_argument("-r", "--rotate",
                        help="Image rotation", default=0, choices={0, 90, -90, 180}, type=int)

    options = parser.parse_args()
    (w, h) = options.grid_size.split('x')

    detect_camera_position(options.path,
                           options.calib_file,
                           options.file_name,
                           int(w), int(h),
                           options.cell_size,
                           options.rotate)

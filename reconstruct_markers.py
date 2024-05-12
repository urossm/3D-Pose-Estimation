import argparse
import numpy as np
import cv2
import os
from itertools import zip_longest
from utils import load_json_file, load_calib_files, project_3d_markers, draw_pose


def draw_markers_on_video(input_video, json_file, rot, json_3d_file, calib_file, output_video):
    try:
        video_cap = cv2.VideoCapture(input_video)
    except:
        print("Not able to open: {}".format(input_video))
        return

    if not video_cap.isOpened():
        print("Not able to open: {}".format(input_video))
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(video_cap.get(cv2.CAP_PROP_FOURCC))

    video_out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    keypoints_2d = np.asarray(load_json_file(json_file)["2d_points"])
    keypoints_3d = np.asarray(load_json_file(json_3d_file)["3d_points"])

    cam_mat, _, rot, trans = load_calib_files(calib_file)

    cv2.namedWindow("Marker", cv2.WINDOW_AUTOSIZE)
    for points, points_3d in zip_longest(keypoints_2d, keypoints_3d, fillvalue=None):
        ret, frame = video_cap.read()

        if not ret:
            break
        if points is not None:
            frame = draw_pose(frame, points, threshold=0.0, color=(255, 0, 0))
        if points_3d is not None:

            projected_points = project_3d_markers(points_3d, cam_mat, rot, trans)

            frame = draw_pose(frame, projected_points, threshold=0.0, color=(0, 0, 255))

        video_out.write(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Marker", frame)

        cv2.waitKey(int(1000./fps))

    video_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drawing markers over undistorted video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", "--video", help="Input video filename", required=True)
    parser.add_argument("-j", "--json",
                        help="Folder name for resulting openpose reconstruction for the given video")
    parser.add_argument(
        "-d", "--marker_3d", help="Folder name that contains JSON files with 3d reconstruction")
    parser.add_argument("-o", "--output",
                        help="Output video name", required=True)
    parser.add_argument("-r", "--frame_rotate", help="Angle of rotation when openpose is run",
                        default=0, choices={0, 90, 180, 270}, type=int)
    parser.add_argument("-b", "--calib_file",
                        help="Calib files names")
    options = parser.parse_args()
    draw_markers_on_video(options.video, options.json,
                           options.frame_rotate, options.marker_3d, options.calib_file, options.output)
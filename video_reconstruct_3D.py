import argparse
import cv2
import numpy as np
from utils import load_json_file, project_3d_markers, draw_pose


def generate_3d_video(output_vid, markers_3d, focus, size, pos, rot):

    print("Output file: {}\nMarkers file: {}\nFocus: {}\nSize: {}\nPos: {}\nRot: {}\n".format(
        output_vid, markers_3d, focus, size, pos, rot
    ))

    video_out = cv2.VideoWriter(
        output_vid, cv2.VideoWriter_fourcc('M','J','P','G'), 15, size
    )

    intrinsic = np.asmatrix(
        [[focus, 0, size[0]/2], [0, focus, size[1]/2], [0, 0, 1]]
    )

    Rz, _ = cv2.Rodrigues((0, 0, rot[2] * np.pi / 180.))
    Ry, _ = cv2.Rodrigues((0, rot[1] * np.pi / 180., 0))
    Rx, _ = cv2.Rodrigues((rot[0] * np.pi / 180., 0, 0))

    rot_mat = np.matmul(Rz, np.matmul(Ry, Rx)).T
    trans = - np.matmul(rot_mat, np.array(pos)).reshape(3, 1)

    keypoints_3d = np.asarray(load_json_file(markers_3d)["3d_points"])

    frame = np.zeros([size[1], size[0], 3], dtype=np.uint8)
    cv2.namedWindow("Marker", cv2.WINDOW_KEEPRATIO)

    for keypt in keypoints_3d:
        frame.fill(0)

        points = project_3d_markers(keypt, intrinsic, rot_mat, trans)

        img = draw_pose(frame, points, threshold=0.5)

        video_out.write(img)
        cv2.imshow("Marker", img)
        cv2.waitKey(int(1000/15))

    video_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render the video with human pose from selected view point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-d", "--marker_3d", help="JSON file with 3D reconstruction file name", required=True)

    parser.add_argument(
        "-o", "--output", help="Output video filename", required=True)

    parser.add_argument(
        "-w", "--width", help="Image width", default=1280, type=int)

    parser.add_argument(
        "-t", "--height", help="Image height", default=720, type=int)

    parser.add_argument(
        "-f", "--focus", help="Camera focal point", default=800, type=int)

    parser.add_argument(
        "-x", "--x_coor", help="Camera X coordinate", default=0, type=float)

    parser.add_argument(
        "-y", "--y_coor", help="Camera Y coordinate", default=0, type=float)

    parser.add_argument(
        "-z", "--z_coor", help="Camera Z coordinate", default=0, type=float)

    parser.add_argument(
        "-r", "--roll", help="Camera roll", default=0, type=float)
    parser.add_argument(
        "-p", "--pitch", help="Camera pitch", default=0, type=float)
    parser.add_argument(
        "-a", "--yaw", help="Camera yaw", default=0, type=float)

    options = parser.parse_args()

    generate_3d_video(
        options.output,
        options.marker_3d,
        options.focus,
        (options.height, options.width),
        (options.x_coor, options.y_coor, options.z_coor),
        (options.roll, options.pitch, options.yaw)
    )


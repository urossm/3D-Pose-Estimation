import cv2
import argparse
from utils import rotate_image


def calib_image(img, matx, dist, rotate):
    img = rotate_image(img, rotate)
    undistorted = cv2.undistort(img, matx, dist, None, matx)
    return undistorted


def calib_video(video_file, calib_file, video_output, rotate):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeffs = fs.getNode('distortion').mat()

    video_cap = cv2.VideoCapture(video_file)
    video_w = int(video_cap.get(3))  # width
    video_h = int(video_cap.get(4))  # height

    if not video_cap.isOpened():
        print("Video couldn't be loaded. Shutting down...")
        exit()

    video_write = cv2.VideoWriter(
        video_output,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        15,
        (video_h, video_w))

    cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)

    counter = 1
    while True:
        ret, frame = video_cap.read()

        if ret:
            undistorted = calib_image(frame, cam_mat, dist_coeffs, rotate)
            video_write.write(undistorted)

            cv2.imshow('Frame', undistorted)

            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video_cap.release()
    video_write.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Video lens undistort',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--video",
                        help="Input video path",
                        required=True)

    parser.add_argument("-b", "--calib_file",
                        help="Calib file input path. Calib file must contain <intrinsic> and <distortion> tags",
                        required=True)

    parser.add_argument("-o", "--output",
                        help="Outputh video path",
                        required=True)

    parser.add_argument("-r", "--frame_rotate",
                        help="Angle of rotation so that human will stand upwards",
                        default=0, choices={0, 90, -90, 180}, type=int)

    options = parser.parse_args()

    calib_video(video_file=options.video,
                video_output=options.output,
                calib_file=options.calib_file,
                rotate=options.frame_rotate)

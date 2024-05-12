import cv2
from openpose_cv import OpenPose
import matplotlib.pyplot as plt
import time
import argparse
from utils import colors, POSE_PAIRS, BODY_PARTS


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detecting human pose using OpenPose and OpenCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--video",
                        help="Input video",
                        required=True
                        )

    parser.add_argument("-o", "--out_video",
                        help="Output video with pose",
                        required=True
                        )

    parser.add_argument("-j", "--json",
                        help="Output JSON file with pose",
                        required=True
                        )

    parser.add_argument("-m", "--model",
                        help="Name of caffe model network",
                        default="./models/pose/coco/pose_iter_440000.caffemodel")
    parser.add_argument("-p", "--prototxt",
                        help="Name of prototxt caffe model network file",
                        default="./models/pose/coco/pose_deploy_linevec.prototxt")

    parser.add_argument("-s", "--scale",
                        help="Scale for pixel values (normalization)",
                        default=1. / 255.)
    parser.add_argument("-t", "--threshold",
                        help="Threshold (confidence)",
                        default=0.5)
    parser.add_argument("-r", "--frame_rotate",
                        help="Angle of rotation so that human will stand upwards",
                        default=0, choices={0, 90, -90, 180}, type=int, required=True)

    options = parser.parse_args()
    video_file = options.video
    video_output = options.out_video
    json_file = options.json

    scale = options.scale
    thr = options.threshold
    rotate = options.frame_rotate

    prototxt = options.prototxt
    caffe_model = options.model

    video_cap = cv2.VideoCapture(video_file)
    video_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_write = cv2.VideoWriter(
        video_output,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        15,
        (video_w, video_h))

    inH = 368
    inW = 368#int((inH / video_h) * video_w)

    """ Create OPNEPOSE CLASS object"""
    open_pose = OpenPose(
        prototxt=prototxt,
        caffemodel=caffe_model,
        scale=scale,
        inW=inW,
        inH=inH,
        video_W=video_w,
        video_H=video_h,
        thr=thr
    )

    if not video_cap.isOpened():
        print("Video couldn't be loaded. Shutting down...")
        exit()

    cv2.namedWindow("Pose", cv2.WINDOW_KEEPRATIO)

    while True:
        ret, frame = video_cap.read()
        start = time.time()
        if ret:
            output_frame = open_pose.draw_pose(frame, rotate)

            #print("Time to forward 1 frame: {:3} s".format(time.time() - start))

            video_write.write(output_frame)

            cv2.putText(output_frame, 'FPS: {:.2f}'.format(1. / (time.time() - start)),
                        org=(30, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.imshow("Pose", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            open_pose.save_points_to_json(json_file)
            break

    video_cap.release()
    video_write.release()
    cv2.destroyAllWindows()
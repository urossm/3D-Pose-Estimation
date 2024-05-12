import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import rotate_image, BODY_PARTS, POSE_PAIRS, colors
import json


class OpenPose:

    def __init__(self, prototxt=None, caffemodel=None, scale=None,
                 inW=None, inH=None, video_W=None, video_H=None, thr=None):
        self.net = cv2.dnn.readNet(cv2.samples.findFile(prototxt), cv2.samples.findFile(caffemodel))

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")

        self.scale = scale
        self.inW = inW
        self.inH = inH
        self.video_W = video_W
        self.video_H = video_H
        self.thr = thr

        self.pad_bot = 0
        self.pad_right = 0

        self.points_list = list()

    def net_output(self, image):

        if self.video_H > self.video_W:
            self.pad_right = self.video_H - self.video_W
            self.pad_bot = 0
            image = cv2.copyMakeBorder(image, 0, 0, 0, self.pad_right, cv2.BORDER_CONSTANT)

        elif self.video_H < self.video_W:
            self.pad_right = 0
            self.pad_bot = self.video_W - self.video_H
            image = cv2.copyMakeBorder(image, 0, self.pad_bot, 0, 0, cv2.BORDER_CONSTANT)
        else:
            self.pad_right = 0
            self.pad_bot = 0

        input = cv2.dnn.blobFromImage(image, self.scale, (self.inW, self.inH), (0, 0, 0))
        self.net.setInput(input)

        output = self.net.forward()
        return output

    def get_keypoints(self, frame):
        out = self.net_output(frame)

        points = []

        for i in range(len(BODY_PARTS)):
            heat_map = out[0, i, :, :]
            heat_map = cv2.resize(heat_map, (self.video_W + self.pad_right, self.video_H + self.pad_bot))

            _, conf, _, point = cv2.minMaxLoc(heat_map)#heat_map)
            x = point[0]
            y = point[1]
            points.append((int(x), int(y), conf))

        self.points_list.append(points)
        return points

    def draw_pose(self, frame, rotate):
        image = rotate_image(frame, rotate)

        frame_copy = image.copy()

        points = self.get_keypoints(frame_copy)

        if len(points) < 2:
            return frame_copy

        for i, pair in enumerate(POSE_PAIRS):
            pairFrom = pair[0]
            pairTo = pair[1]
            assert (pairFrom in BODY_PARTS)
            assert (pairTo in BODY_PARTS)

            idFrom = BODY_PARTS[pairFrom]
            idTo = BODY_PARTS[pairTo]

            if points[idFrom][2] > self.thr and points[idTo][2] > self.thr:
                cv2.line(frame_copy, points[idFrom][:2], points[idTo][:2], colors[i], 3)
                cv2.circle(frame_copy, points[idFrom][:2], 3, colors[idFrom], -1, cv2.LINE_AA)
                cv2.circle(frame_copy, points[idTo][:2], 3, colors[idTo], -1, cv2.LINE_AA)

        return frame_copy

    def save_points_to_json(self, path):
        points = np.asarray(self.points_list).tolist()
        data = {"2d_points": points}

        with open(path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)

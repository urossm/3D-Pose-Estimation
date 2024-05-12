import cv2
import numpy as np
import json


BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    #"Background": 18
}

POSE_PAIRS = [  ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

colors = [
    [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
    [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
    [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]
]


def rotate_image(frame, rotate):
    if rotate == 90:
        image = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == -90:
        image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif abs(rotate) == 180:
        image = cv2.rotate(frame, cv2.ROTATE_180)
    else:
        image = frame

    return image


def load_json_file(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def load_calib_files(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeff = fs.getNode('distortion').mat()
    Rot = fs.getNode('Rot').mat()
    trans = fs.getNode('Trans').mat()

    fs.release()
    return cam_mat, dist_coeff, Rot, trans


def save_json_file(pose, filename):
    pose_list = np.asarray(pose).tolist()
    data = {"3d_points": pose_list}
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=2)


def project_marker(points, cam_mat, rot, trans):
    image_points = np.matmul(
        cam_mat, np.matmul(rot, points) + trans)

    ret = (image_points/image_points[2, :])[0:2, :]
    return ret


def project_3d_markers(keypoints, cam_mat, rot, trans):

    points = np.zeros((keypoints.shape[0], 3), dtype=float)
    projected = project_marker(keypoints[:, 0:3].T, cam_mat, rot, trans).T

    points[:, 0:2] = projected
    points[:, 2] = keypoints[:, -1].T

    return points


def draw_pose(frame, points, threshold = None, radius = 5, thickness = 3, color = None):

    frame_copy = frame.copy()

    for i, pair in enumerate(POSE_PAIRS):
        pairFrom = pair[0]
        pairTo = pair[1]
        assert (pairFrom in BODY_PARTS)
        assert (pairTo in BODY_PARTS)

        idFrom = BODY_PARTS[pairFrom]
        idTo = BODY_PARTS[pairTo]

        x_from = int(points[idFrom][0])
        y_from = int(points[idFrom][1])
        thr_from = (points[idFrom][2])

        x_to = int(points[idTo][0])
        y_to = int(points[idTo][1])
        thr_to = (points[idTo][2])

        color_from = colors[idFrom]
        color_to = colors[idTo]
        if color is None:
            color_line = colors[i]
        else:
            color_line = color

        if thr_from > threshold and thr_to > threshold:
            cv2.line(frame_copy, (x_from, y_from), (x_to, y_to), color_line, thickness=thickness)
            cv2.circle(frame_copy, (x_from, y_from), radius, color_from, -1, cv2.LINE_AA)
            cv2.circle(frame_copy, (x_to, y_to), 3, color_to, -1, cv2.LINE_AA)

    return frame_copy


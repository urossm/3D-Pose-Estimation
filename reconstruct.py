import numpy as np
import argparse
import utils

IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def reconstruct_pose(i, all_points, all_intrinsic, all_rot, all_cam_pos, threshold):
    num_cameras = len(all_points)

    points = list()
    for pt in all_points:
        points.append(pt[:, :, i])

    ray_list = list()
    confidense_list = list()

    for pt, intris, rot, cp in zip(points, all_intrinsic, all_rot, all_cam_pos):
        # pt = [x, y, conf] ----> [[x],[y],[conf]]
        q = pt.T.copy()
        # q = [x; y; 1]
        q[2, :] = 1

        ray = np.matmul(np.linalg.inv(np.matmul(intris, rot.T)), q)
        ray_list.append(ray)
        confidense_list.append(pt[:, 2])

    num_pts = ray.shape[1]

    res = np.zeros((num_pts, 4), dtype=float)
    lhs = np.zeros((3*num_cameras, num_cameras + 3), dtype=float)

    for idx in range(num_cameras):
        lhs[idx*3:idx*3+3, num_cameras:num_cameras+3] = np.identity(3, dtype=float)

    rhs = np.vstack(all_cam_pos)
    for j in range(num_pts):
        for idx, ray in enumerate(ray_list):
            lhs[3 * idx:3 * idx + 3, idx] = ray[:, j]
        row_mask = np.array([True] * 3 * num_cameras, dtype=bool)
        col_mask = np.array([True] * (num_cameras + 3), dtype=bool)
        num_above = 0
        mean_confidence = 0
        for idx, conf in enumerate(confidense_list):
            if conf[j] > threshold:
                num_above = num_above + 1
                mean_confidence = mean_confidence + conf[j]
            else:
                col_mask[idx] = False
                row_mask[3 * idx:3 * idx + 3] = False
        if num_above > 1:
            # izbaci one za koje je confidence mali
            L = lhs[row_mask, :][:, col_mask]
            R = rhs[row_mask, :]
            sol = np.matmul(np.linalg.pinv(L), R)
            res[j, 0:3] = sol[-3:].ravel()
            res[j, 3] = mean_confidence / num_above
        else:
            res[j, :] = 0
    return res


def reconstruct(json_file, calib_files, frame_rotate, output, threshold):
    all_points = list()
    all_instrinsic = list()
    all_rot = list()
    all_cam_pos = list()

    num_frames = 1e10
    for path, calib_file in zip(json_file, calib_files):
        # json loaded dimension: [[num_frames][num_joints][x,y,conf]]
        points = np.asarray(utils.load_json_file(path)["2d_points"])
        all_points.append(np.stack(points, axis=2))
        num_frames = min(num_frames, points.shape[0])

        intrinsic, _, rot, trans = utils.load_calib_files(calib_file)
        all_instrinsic.append(intrinsic)
        all_rot.append(rot.T)
        all_cam_pos.append(-np.matmul(rot.T, trans))

    all_poses = list()
    for i in range(num_frames):
        pose = reconstruct_pose(i, all_points, all_instrinsic, all_rot, all_cam_pos, threshold)
        all_poses.append(pose)
    utils.save_json_file(all_poses, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human pose reconstruction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-j", "--json", help="JSON files paths", nargs="+", required=True)
    parser.add_argument("-b", "--calib_files", help="Calib files paths", nargs="+", required=True)
    parser.add_argument("-r", "--frame_rotate",
                        help="Angle of rotation so that human will stand upwards",
                        default=0, choices={0, 90, -90, 180}, type=int, nargs="+", required=True)

    parser.add_argument("-o", "--output", help="JSON files save path", default="points_3d_{num:05d}.json")
    parser.add_argument("-t", "--threshold", help="Detection threshold", default=0.5, type=float)

    options = parser.parse_args()

    if len(options.json) != len(options.calib_files) or len(options.json) != len(options.frame_rotate):
        raise Exception(
            "Number of JSON files is not same as the calib files and angles of rotation"
        )

    reconstruct(options.json, options.calib_files, options.frame_rotate, options.output, options.threshold)

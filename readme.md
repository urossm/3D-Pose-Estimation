# 3D Pose Estimation

3D Pose Estimation is a personal Python project that utilizes the OpenCV library to detect motion and estimate poses of the human from video file or camera. 
It includes features for for camera calibration, video undistortion, motion reconstruction and rendering video with 3d skeleton motion.

## Usage and Features

- **SOURCE FILES**: Download source files from my Google Drive because it was too large for GitHub (https://drive.google.com/drive/folders/13LgNRNZ5iUbON0s7Uxzq4yXxqST7OSD1?usp=sharing)
- **Camera Calibration**: Calibrate camera and export intrinsix matrix in YAML file
- **Video Lens Undistortion**: Undistort lens curvature in video from camera calibration files
- **Camera Extrinsic Matrix**: Detect camera position in 3D space and export extrinsic matrix in YAML file
- **Human Pose Detection**: Detecting human pose using OpenPose and OpenCV
- **Human Pose Reconstruction**: Reconstruct human pose from video and export JSON file with 3D points
- **Reconstruct Markers on the Video**: Reconstruct markers and render video from human pose 3D points
- **Render Skeleton Video**: Render the video with human pose from selected point of view


## Result

- Result video with estimated poses is in **_RESULTS_** folder:

![Result video](https://raw.githubusercontent.com/urossm/3D-Pose-Estimation/main/RESULTS/thumbnail.png)

## Requirements

- Python (version 3.6)
- OpenCV (version 3.4.9)

## Installation

1. Clone the repository
2. Download source files from my Google Drive because it was too large for GitHub (https://drive.google.com/drive/folders/13LgNRNZ5iUbON0s7Uxzq4yXxqST7OSD1?usp=sharing)
3. Install the required dependencies from requirements.txt
4. Run and have fun :)




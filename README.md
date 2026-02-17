# automated-boxing-scoring-system
This Python script is an automated boxing scoring system using YOLOv8-pose and OpenCV. It tracks two players via dual-camera feeds, using HSV color masks to identify teams and IOU tracking to prevent identity swaps. Scores are triggered by a "pull-back" state machine that monitors hand-to-body distance in centimeters.


Dual-View Management: Processes two video streams (t1.mp4, t2.mp4) and automatically switches the active view if a player is obscured or a clinch occurs.

Team Identification: Uses HSV color filtering to distinguish between players (Red vs. Black) and maintains their identity using IOU (Intersection Over Union) tracking to prevent "swapping" during close-quarters combat.

"Pull-back" Scoring Logic: Implements a state machine that counts a hit only after a 100ms hold within range and requires a "pull-back" distance before the next score can be registered, preventing double-counting.

Environmental Adaptation: Features a SmoothAutoBrightness class that dynamically adjusts gain and applies CLAHE to handle varying gym lighting conditions.

Automated Dataset Generation: Automatically captures snapshots of every scored hit and logs the metadata (frame ID, team, hand side, distance) into a dataset.csv.

## Technical Stack
Deep Learning: YOLOv8s-pose (Ultralytics)

Computer Vision: OpenCV (Python)

Logic: Finite State Machine (FSM) for punch detection

Hardware Compatibility: Optimized for systems capable of running YOLO inference (CUDA recommended)

## Getting Started
### Prerequisites

Bash
pip install ultralytics opencv-python numpy
### Running the System

Place your video files as t1.mp4 and t2.mp4 in the root directory.

Ensure yolov8s-pose.pt is downloaded.

Execute the script:

Bash
python test.py
### Controls

p / s: Pause/Resume the video.

m: Toggle mask views for color calibration.

[ / ]: Decrease/Increase the punch detection range.

- / +: Calibrate PX_PER_CM for distance accuracy.

q: Quit the application.
![hit_0005_BLACK](https://github.com/user-attachments/assets/1ce7e0c3-3bb3-48ad-8afc-a0cd1cc57728)


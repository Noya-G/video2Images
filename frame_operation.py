import cv2
import os

import numpy as np


def extract_frames(video_path, output_dir, num_frames, file_type):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open the video file.")
        return
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames - {total_frames}")
    if num_frames > total_frames:
        print("Error: The requested number of frames exceeds the total frames in the video.")
        return

    interval = total_frames // num_frames
    extracted_frames = 0
    frame_count = 0
    while extracted_frames < num_frames and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{extracted_frames}.{file_type}")
            cv2.imwrite(frame_path, frame)
            extracted_frames += 1
        frame_count += 1
    video_capture.release()
    print(f"Successfully extracted {extracted_frames} frames to '{output_dir}'.")


def extract_frames_by_time(video_path, output_dir, frame_rate, file_type, percent, limit, memory):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open the video file.")
        return
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate)
    extracted_frames = 0
    frame_count = 0

    while video_capture.isOpened():
        if limit is not None and extracted_frames >= limit:
            break
        video_capture.set(cv2.CAP_PROP_POS_MSEC, frame_count * 1000 / fps)
        ret, frame = video_capture.read()
        if not ret:
            break
        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{extracted_frames}.{file_type}")
        cv2.imwrite(frame_path, frame)
        extracted_frames += 1
        frame_count += frame_interval
    video_capture.release()
    print(f"Successfully extracted {extracted_frames} frames to '{output_dir}'.")


def calculate_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    percentage_difference = np.mean(magnitude) / 255 * 100
    return percentage_difference

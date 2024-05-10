import cv2
import os


def extract_frames(file_type, video_path, output_dir, num_frames):
    # Open the video file
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
    # Release the video capture object
    video_capture.release()
    print(f"Successfully extracted {extracted_frames} frames to '{output_dir}'.")

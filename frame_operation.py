from video2Images.video2imageCLI import logger
import cv2
import os
from video2Images import iou_and_zoom
from tqdm import tqdm

def extract_frames(video_path, output_dir, num_frames, file_type):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        logger.error("Error: Could not open the video file.")
        return
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames - {total_frames}")
    if num_frames > total_frames:
        logger.info("Error: The requested number of frames exceeds the total frames in the video.")
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
    logger.info(f"Successfully extracted {extracted_frames} frames to '{output_dir}'.")


def extract_frames_by_time(video_path, output_dir, frame_rate, file_type, percent, limit, memory):
    logger.info("$$$$$ START $$$$$")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        logger.error("Error: Could not open the video file.")
        return
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate)
    extracted_frames = 0
    frame_count = 0

    video_capture.set(cv2.CAP_PROP_POS_MSEC, frame_count * 1000 / fps)
    ret, prev_frame = video_capture.read()
    if not ret:
        logger.error("Error occurred!")
        return
    while video_capture.isOpened():
        if limit is not None and extracted_frames >= limit:
            break
        video_capture.set(cv2.CAP_PROP_POS_MSEC, frame_count * 1000 / fps)
        ret, current_frame = video_capture.read()
        if not ret:
            break
        # Save the frame
        if percent is not None:
            change_diff = iou_and_zoom.calculate_iou(prev_frame, current_frame)
            logger.info(f"percent of change - {change_diff}")
            if change_diff > percent:
                frame_path = os.path.join(output_dir, f"frame_{extracted_frames}.{file_type}")
                cv2.imwrite(frame_path, current_frame)
                prev_frame = current_frame.copy()
                extracted_frames += 1
                logger.info(f"extracted frame number - {extracted_frames}")
        else:
            frame_path = os.path.join(output_dir, f"frame_{extracted_frames}.{file_type}")
            cv2.imwrite(frame_path, prev_frame)
            prev_frame = current_frame.copy()
            extracted_frames += 1
        frame_count += frame_interval
    video_capture.release()
    logger.info(f"Successfully extracted {extracted_frames} frames to '{output_dir}'.")

import logging
import math
import os
import re
import cv2
import numpy as np
from LogMannager import create_new_folder
from eanalyzeTool import *
import subprocess
from concurrent.futures import ThreadPoolExecutor

from src.analysisTools.keypointCented import keypoint_movement_towards_center

SKIP = 20 # Number of frames to skip
THRESHOLD = 10
def extract_frames(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    # Loop through each frame in the video
    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # If there are no more frames, break the loop
        if not success:
            break

        frame_count += 1

        # Append the frame to the list when the frame count is a multiple of skip
        if frame_count % SKIP == 0:
            frames.append(frame)

    # Release the video capture object
    video_capture.release()

    return frames


def estimate_camera_movement(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching (using SIFT)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match keypoints between the frames
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estimate transformation (homography)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Decompose transformation
    dx = H[0, 2]  # Translation in x direction
    dy = H[1, 2]  # Translation in y direction
    theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Rotation angle (in degrees)

    # Calculate camera movement
    translation_distance = np.sqrt(dx ** 2 + dy ** 2)

    return translation_distance, theta


def movement_estimator(frames):
    estimator = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(frames)-1):
            frame1 = frames[i]
            frame2 = frames[i+1]
            future = executor.submit(estimate_camera_movement, frame1, frame2)
            futures.append((i, i +1, future))  # Store the frame indexes along with the future object

        # Retrieve results
        for start_index, end_index, future in futures:
            translation_distance, theta = future.result()
            estimator.append((start_index, end_index, translation_distance, theta))  # Include the frame indexes in the results
    return estimator


def calculate_significance(frames, indexes):
    significance = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        translation_distance, theta = estimate_camera_movement(frame1, frame2)
        significance.append((indexes[i], indexes[i + 1], translation_distance, theta))
    return significance


def select_frames_with_dynamic_programming(significance, limit):
    n = len(significance)
    dp = [0] * n
    for i in range(n):
        dp[i] = max(dp[j] for j in range(i) if significance[i][1] - significance[j][1] > THRESHOLD) + 1

    # Find the index of the last frame in the selected sequence
    max_index = max(range(n), key=lambda x: dp[x])

    # Backtrack to find the indexes of the frames in the selected sequence
    selected_indexes = []
    while max_index >= 0 and len(selected_indexes) < limit:
        selected_indexes.append(significance[max_index][0])
        max_index = max(j for j in range(max_index) if dp[j] == dp[max_index] - 1 and significance[max_index][1] - significance[j][1] > THRESHOLD)

    # Return the selected frames with their indexes
    return [(index, significance[index][1]) for index in selected_indexes]



def select_frames_with_indexes(frames, indexes, s):
    significance = calculate_significance(frames, indexes)
    selected_frames_with_indexes = select_frames_with_dynamic_programming(significance, s)
    selected_indexes = [frame[0] for frame in selected_frames_with_indexes]
    selected_frames = [frames[indexes.index(index)] for index in selected_indexes]
    return selected_frames, selected_indexes


def select_frames(movement):
    selected_frames = []
    for start_index, end_index, translation_distance, theta in movement:
        if (translation_distance >THRESHOLD
                or abs(theta) > THRESHOLD):
            selected_frames.append(start_index)

    return selected_frames


def get_selected_frames(selected_frame_indexes, frames):
    selected_frames = []
    for index in selected_frame_indexes:
        selected_frames.append(frames[index])
    return selected_frames


def save_frames_as_photos(frames, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each frame as an image file
    for i, frame in enumerate(frames):
        filename = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(filename, frame)

def detect_drone_movement(selected_frames,indexes):
    frames =list(zip(indexes, selected_frames))
    frames_with_movement_indexes = []

    # Variables to track the drone's movement
    ascending = False
    turning_down = False

    for idx, frame in frames:
        # Convert frame to grayscale for easier processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges in the frame
        edges = cv2.Canny(gray_frame, 50, 150)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if drone is ascending (going up)
                if y2 > y1:
                    ascending = True
                else:
                    ascending = False

                # Check if drone is turning its camera down
                if x1 == x2:
                    turning_down = True

        # If the drone has finished ascending and turned its camera down
        if ascending and turning_down:
            frames_with_movement_indexes.append(idx)

    return frames_with_movement_indexes


def configure_logging(log_dir):
    log_file_path = os.path.join(log_dir, "chosen_frames.log")
    error_log_file_path = os.path.join(log_dir, "error.log")

    # Configure logging to write INFO-level messages to chosen_frames.log
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Configure logging to write ERROR-level messages to error.log
    error_log_handler = logging.FileHandler(error_log_file_path)
    error_log_handler.setLevel(logging.ERROR)
    error_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_log_handler.setFormatter(error_log_formatter)
    logging.getLogger().addHandler(error_log_handler)

def get_git_branch():
    try:
        # Run git command to get current branch
        result = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        # Decode bytes to string and remove whitespace
        branch = result.decode('utf-8').strip()
        return branch
    except Exception as e:
        print(f"Error getting git branch: {e}")
        return None



def get_git_commit_info():
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        # Get commit time
        commit_time = subprocess.check_output(['git', 'show', '-s', '--format=%ci', commit_hash]).decode('utf-8').strip()
        # Get commit message
        commit_message = subprocess.check_output(['git', 'show', '-s', '--format=%s', commit_hash]).decode('utf-8').strip()

        return commit_time, commit_message
    except Exception as e:
        print(f"Error getting git commit information: {e}")
        return None, None


def parse_log_file(log_file_path):
    import os

    # Check if the log file exists
    if not os.path.exists(log_file_path):
        # If the file doesn't exist, return default values
        return None, None, None, None, None

    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extract relevant information using regular expressions
    skip_match = re.search(r'SKIP: (\d+)', log_content)
    threshold_match = re.search(r'THRESHOLD: (\d+)', log_content)
    branch_match = re.search(r'Branch: (\w+)', log_content)
    estimator_match = re.findall(r'movement estimator: \[(.*?)\]', log_content, re.DOTALL)
    frames_direction_match = re.findall(r'frames direction detected: \[(.*?)\]', log_content, re.DOTALL)

    if skip_match and threshold_match and branch_match:
        skip = int(skip_match.group(1))
        threshold = int(threshold_match.group(1))
        branch = branch_match.group(1)
        # Parse the estimator list if it exists
        if estimator_match:
            estimator_data = estimator_match[0]
            # Split the string by parentheses and extract numeric values
            estimator = [
                (int(data.split(',')[0]), int(data.split(',')[1]), float(data.split(',')[2]), float(data.split(',')[3]))
                for data in re.findall(r'\((.*?)\)', estimator_data)]
        else:
            estimator = None

        # Parse the frames_direction data if it exists
        if frames_direction_match:
            frames_direction_data = frames_direction_match[0]
            # Split the string by parentheses and extract numeric values
            frames_direction = [(int(data.split(',')[0]), int(data.split(',')[1]), int(data.split(',')[2])) for data in
                                re.findall(r'\((.*?)\)', frames_direction_data)]
        else:
            frames_direction = None

        return skip, threshold, branch, estimator, frames_direction
    else:
        return None, None, None, None, None


def movement_direction(frames, indexes):
    frames_direction = []
    framesWI = list(zip(indexes, frames))
    for i in range(len(framesWI) - 1):  # Iterate over the range of indices
        index1, frame1 = framesWI[i]
        index2, frame2 = framesWI[i + 1]
        direc = keypoint_movement_towards_center(frame1, frame2)
        frames_direction.append((index1, index2, direc))
    return frames_direction

def longest_subarray_with_value(lst):
    if not lst:
        return []

    n = len(lst)
    dp = [0] * n
    max_length = 0
    end_index = 0

    for i in range(n):
        if lst[i][2] == 4:
            dp[i] = dp[i - 1] + 1 if i > 0 else 1
            if dp[i] > max_length:
                max_length = dp[i]
                end_index = i

    start_index = end_index - max_length + 1
    return lst[start_index:end_index + 1]

def find_most_significant_indexes(data, size):
    n = len(data)

    # Create a table to store the maximum change for each index
    max_change_table = [0] * n

    # Calculate the maximum change for each index
    for i in range(n - 1, -1, -1):
        max_change = 0
        for j in range(i + 1, min(i + size + 1, n)):
            change = abs(data[j][2] - data[i][2]) + abs(data[j][3] - data[i][3])
            max_change = max(max_change, change)
        max_change_table[i] = max_change

    # Backtrack to find the indexes with the most significant changes
    significant_indexes = []
    current_index = 0
    while len(significant_indexes) < size:
        significant_indexes.append(data[current_index][0])  # Append the first element of the chosen tuple
        next_index = current_index + 1
        for i in range(current_index + 1, min(current_index + size + 1, n)):
            if max_change_table[i] > max_change_table[next_index]:
                next_index = i
        current_index = next_index

    return significant_indexes


def largest_interval_subset(original_list, size=100):
    step = len(original_list) // size  # Calculate step size for subset

    if step == 0:  # If original list is smaller than size 100
        return original_list

    subset = []
    for i in range(0, len(original_list), step):
        subset.append(original_list[i])

    # Adjust the size of the subset to be exactly 100
    if len(subset) < size:
        last_index = len(original_list) - 1
        while len(subset) < size and last_index >= 0:
            subset.append(original_list[last_index])
            last_index -= step

    return subset[:size]


def select_frames_with_most_significant_movements(selected_frames, selected_frames_indexes, limit):
    # Calculate significance of movements
    significance = calculate_significance(selected_frames, selected_frames_indexes)

    # Select frames with the most significant movements using dynamic programming
    selected_frames_with_indexes = select_frames_with_dynamic_programming(significance, limit)

    # Extract selected frames and indexes
    selected_indexes = [frame[0] for frame in selected_frames_with_indexes]
    selected_frames = [selected_frames[selected_frames_indexes.index(index)] for index in selected_indexes]

    return selected_frames, selected_indexes

def get_log_file(cond,video_path, destPath,preLog=None):

    video_path = video_path
    video_name = last_six_chars = video_path[-6:]
    new_path = create_new_folder("/Users/noyagendelman/Desktop/choosingFrames/v2p", "chosen_frames")
    if cond is True:
        log_file_path = os.path.join(new_path, "chosen_frames.log")
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logERORR_file_path = os.path.join(new_path, 'error.log')
        logging.basicConfig(filename=logERORR_file_path, level=logging.ERROR)

        # Get current branch
        current_branch = get_git_branch()

        # Get commit information
        commit_time, commit_message = get_git_commit_info()

        if commit_time and commit_message:
            logging.info(f"Commit Time: {commit_time}")
            logging.info(f"Commit: {commit_message}")
        else:
            logging.info("Commit information not available")

        if current_branch:
            logging.info(f"Branch: {current_branch}")
        else:
            logging.info("Branch information not available")
        logging.info(f"SKIP: {SKIP}")
        logging.info(f"THRESHOLD: {THRESHOLD}")
        logging.info(f"Video name: {video_name}")

    # Extract frames from the video
    all_frames = extract_frames(video_path)
    if cond is True:
        logging.info(f"Total frames extracted: {len(all_frames)}")

        # Load the previous log file
        if preLog is not None:
            previous_log_file_path = '/Users/noyagendelman/Desktop/choosingFrames/v2p/chosen_frames_22/chosen_frames.log'  # Change this to the path of your previous log file
            previous_skip, previous_threshold, previous_branch, previous_estimator, previous_frames_direction = parse_log_file(
                previous_log_file_path)

            # Check if any previous parameter is None
            if any(param is None for param in
                   [previous_skip, previous_threshold, previous_branch, previous_estimator, previous_frames_direction]):
                # Execute the method to proceed with regular execution
                estimator = movement_estimator(all_frames)
                frames_indexes = select_frames(estimator)

                frames_direction = movement_direction(all_frames, frames_indexes)
            else:
                # Proceed with regular execution
                estimator = previous_estimator
                frames_direction = previous_frames_direction

        else:
            estimator = movement_estimator(all_frames)
            frames_indexes = select_frames(estimator)
            frames_direction = movement_direction(all_frames, frames_indexes)

        logging.info("Using previous estimator and frames_direction from the log file.")
        # Select frames with significant camera movement
        logging.info(f"Total pairs of frames processed: {len(estimator)}")
        logging.info(f"movement estimator: {estimator}")
        logging.info(f"frames direction detected: {frames_direction}")


    if cond is False:
        estimator = movement_estimator(all_frames)
        # Select frames with significant camera movement
        selected_frames_indexes = select_frames(estimator)
        frames_indexes = select_frames(estimator)
        frames_direction = movement_direction(all_frames, frames_indexes)

    if cond is True:
        logging.info(f"Total selected frames: {len(selected_frames_indexes)}")
        logging.info(f"selected frames: {selected_frames_indexes}")

    # Get selected frames
    selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    # Save selected frames as images
    # save_frames_as_photos(selected_frames, new_path)
    expected_translation_distance, expected_theta = calculate_expected_values(estimator,
                                                                              selected_frames,
                                                                              selected_frames_indexes)
    if cond is True:
        logging.info(f"expected translation distance: {expected_translation_distance}")
        logging.info(f"expected theta: {expected_theta}")
        # Plot translation distance
        plot_translation_distance(expected_translation_distance, estimator[:len(selected_frames_indexes) - 1],
                                  selected_frames_indexes, new_path)

        # Plot theta
        plot_theta(expected_theta, estimator[:len(selected_frames_indexes) - 1], selected_frames_indexes,
                   new_path)
        plot_theta_zoom_out(expected_theta, estimator[:len(selected_frames_indexes) - 1], selected_frames_indexes,
                            new_path)

    longestSubarray = longest_subarray_with_value(frames_direction)

    firstGoodFrame = longestSubarray[-1][1]
    if (firstGoodFrame < selected_frames_indexes[math.floor(len(selected_frames_indexes) / 2)]):
        selected_frames_indexes = selected_frames_indexes[selected_frames_indexes.index(firstGoodFrame):]
        if cond is True:
            logging.info(f"selected frames after cutting: {selected_frames_indexes}")
            logging.info(f"total selected frames after cutting: {len(selected_frames_indexes)}")
        selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    if (firstGoodFrame > selected_frames_indexes[math.floor(len(selected_frames_indexes) / 2)]):
        selected_frames_indexes = selected_frames_indexes[:selected_frames_indexes.index(firstGoodFrame)]
        if cond is True:
            logging.info(f"selected frames after cutting: {selected_frames_indexes}")
            logging.info(f"total selected frames after cutting: {len(selected_frames_indexes)}")
        selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    # estimator2 = movement_estimator(selected_frames)
    selected_indexes = largest_interval_subset(selected_frames_indexes)
    if cond is True:
        logging.info(f"final selected frames indexes: {selected_frames_indexes}")
    selected_frames = get_selected_frames(selected_indexes, all_frames)

    save_frames_as_photos(selected_frames, new_path)

if __name__ == "__main__":
    videoPath = ""
    destPath = ""
    get_log_file(False)






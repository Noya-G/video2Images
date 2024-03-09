import logging
import os
import re
import cv2
import numpy as np
from LogMannager import create_new_folder
from eanalyzeTool import plot_translation_distance, calculate_expected_values, plot_theta
import subprocess
from concurrent.futures import ThreadPoolExecutor

SKIP = 20 # Number of frames to skip
THRESHOLD = 10
def extract_frames(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []

    # Loop through each frame in the video
    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # If there are no more frames, break the loop
        if not success:
            break

        # Append the frame to the list
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
        for i in range(0, len(frames) - SKIP, SKIP):
            frame1 = frames[i]
            frame2 = frames[i + SKIP]
            future = executor.submit(estimate_camera_movement, frame1, frame2)
            futures.append((i, i + SKIP, future))  # Store the frame indexes along with the future object

        # Retrieve results
        for start_index, end_index, future in futures:
            translation_distance, theta = future.result()
            estimator.append((start_index, end_index, translation_distance, theta))  # Include the frame indexes in the results
    return estimator


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
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extract relevant information using regular expressions
    skip_match = re.search(r'SKIP: (\d+)', log_content)
    threshold_match = re.search(r'THRESHOLD: (\d+)', log_content)
    branch_match = re.search(r'Branch: (\w+)', log_content)
    estimator_match = re.findall(r'movement estimator: \[(.*?)\]', log_content, re.DOTALL)

    if skip_match and threshold_match and branch_match:
        skip = int(skip_match.group(1))
        threshold = int(threshold_match.group(1))
        branch = branch_match.group(1)
        # Parse the estimator list if it exists
        if estimator_match:
            estimator_data = estimator_match[0]
            # Split the string by parentheses and extract numeric values
            estimator = [(int(data.split(',')[0]), int(data.split(',')[1]), float(data.split(',')[2]), float(data.split(',')[3])) for data in re.findall(r'\((.*?)\)', estimator_data)]
        else:
            estimator = None

        return skip, threshold, branch, estimator
    else:
        return None, None, None, None



if __name__ == "__main__":
    # Configure logging to write to different files for INFO and ERROR
    #Configure logging
    # log_dir = "/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames"
    # configure_logging(log_dir)
    #
    # # Log initial parameters
    # logging.info(f"SKIP: {SKIP}")
    # logging.info(f"THRESHOLD: {THRESHOLD}")
    #
    # # Extract frames from the video
    # video_path = "/Users/noyagendelman/Desktop/choosingFrames/v1.mp4"
    # video_name = video_path[-6:]
    # logging.info(f"Video name: {video_name}")
    # all_frames = extract_frames(video_path)
    # logging.info(f"Total frames extracted: {len(all_frames)}")
    #
    # logging.info(f"SKIP: {SKIP}")
    # logging.info(f"THRESHOLD: {THRESHOLD}")
    # logging.info(f"Video name: {video_name}")
    video_path = "/Users/noyagendelman/Desktop/choosingFrames/v1.mp4"  ###Enter here thr mp4 file path
    video_name = last_six_chars = video_path[-6:]
    new_path = create_new_folder("/Users/noyagendelman/Desktop/choosingFrames/v1p","chosen_frames")
    log_file_path = new_path+"chosen_frames.log"
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logERORR_file_path = new_path+'error.log'
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
    logging.info(f"Total frames extracted: {len(all_frames)}")

    # Estimate camera movement between frames
    # estimator = movement_estimator(all_frames)

    # Load the previous log file
    previous_log_file_path = '/Users/noyagendelman/Desktop/choosingFrames/v1p/chosen_frames_60chosen_frames.log'  # Change this to the path of your previous log file
    previous_skip, previous_threshold, previous_branch, previous_estimator = parse_log_file(previous_log_file_path)

    # Current configuration
    current_skip = 20
    current_threshold = 10
    current_branch = 'master'  # Get the current branch using your existing function

    if previous_skip == current_skip and previous_threshold == current_threshold and previous_branch == current_branch:
        # Use the previous estimator
        estimator = previous_estimator
        print("Using previous estimator from the log file.")
    else:
        # Proceed with regular execution
        estimator = movement_estimator(all_frames)

    logging.info(f"Total pairs of frames processed: {len(estimator)}")
    logging.info(f"movement estimator: {estimator}")

    # Select frames with significant camera movement
    selected_frames_indexes = select_frames(estimator)
    logging.info(f"Total selected frames: {len(selected_frames_indexes)}")
    logging.info(f"selected frames: {selected_frames_indexes}")

    # Get selected frames
    selected_frames = get_selected_frames(selected_frames_indexes, all_frames)

    # Save selected frames as images
    # save_frames_as_photos(selected_frames, new_path)
    expected_translation_distance,expected_theta = calculate_expected_values(estimator,
                                                                             selected_frames,selected_frames_indexes)

    logging.info(f"expected translation distance: {expected_translation_distance}")
    logging.info(f"expected theta: {expected_theta}")
    # Plot translation distance
    plot_translation_distance(expected_translation_distance, estimator[:len(selected_frames_indexes) - 1],
                              selected_frames_indexes, new_path)

    # Plot theta
    plot_theta(expected_theta, estimator[:len(selected_frames_indexes) - 1], selected_frames_indexes,
               new_path)

    # logging.info(f"estimate_camera_movement {estimate_camera_movement_con}")
    # video_path = "/Users/noyagendelman/Desktop/choosingFrames/v1.mp4"
    # frames = extract_frames(video_path)
    # Configure logging to write to different files for INFO and ERROR
    # video_path = "/Users/noyagendelman/Desktop/choosingFrames/v1.mp4"
    # video_name = video_path[-6:]
    # new_path = create_new_folder("/Users/noyagendelman/Desktop/choosingFrames/v1p", "chosen_frames")
    # log_file_path = os.path.join(new_path, "chosen_frames.log")
    # log_error_file_path = os.path.join(new_path, "error.log")
    #
    # logging.basicConfig(filename=log_file_path, level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # # Add a separate handler for ERROR-level messages
    # error_log_handler = logging.FileHandler(log_error_file_path)
    # error_log_handler.setLevel(logging.ERROR)
    # error_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # error_log_handler.setFormatter(error_log_formatter)
    # logging.getLogger().addHandler(error_log_handler)
    #
    # logging.info(f"SKIP: {SKIP}")
    # logging.info(f"THRESHOLD: {THRESHOLD}")
    # logging.info(f"Video name: {video_name}")
    selected_f = [360, 400, 500, 520, 540, 560, 580, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2140, 2180, 2200, 2220, 2240, 2260, 2280, 2300, 2320, 2340, 2360, 2380, 2400, 2420, 2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580, 2600, 2620, 2640, 2660, 2680, 2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860, 2880, 2900, 2920, 2940, 2960, 2980, 3000, 3020, 3040, 3060, 3080, 3100, 3120, 3140, 3160, 3180, 3200, 3220, 3240, 3260, 3280, 3300, 3320, 3340, 3360, 3380, 3400, 3420, 3440, 3460, 3480, 3500, 3520, 3540, 3560, 3580, 3600, 3620, 3640, 3660, 3680, 3700, 3720, 3740, 3760, 3780, 3800, 3820, 3840, 3860, 3880, 3900, 3920, 3940, 3960, 3980, 4000, 4020, 4040, 4060, 4080, 4100, 5000, 5020, 5040, 5060, 5080, 5100, 5120, 5140, 5160, 5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320, 5340, 5360, 5380, 5400, 5420, 5440, 5460, 5480, 5500, 5520, 5540, 5560, 5580, 5600, 5620, 5640, 5660, 5680, 5700, 5720, 5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880, 5900, 5920, 5940, 5960, 5980, 6000, 6020, 6040, 6060, 6080, 6100, 6120, 6140, 6160, 6180, 6200, 6220, 6240, 6260, 6280, 6300, 6320, 6340, 6360, 6380, 6400, 6420, 6440, 6460, 6480, 6500, 6520, 6540, 6560, 6580, 6600, 6620, 6640, 6660, 6680, 6700, 6720, 6740, 6760, 6780, 6800, 6820, 6840, 6860, 6880, 6900, 6920, 6940, 6960, 6980, 7000, 7020, 7040, 7060, 7080, 7100, 7120, 7140, 7160, 7180, 7200, 7220, 7240, 7260, 7280, 7300, 7320, 7340, 7360, 7380, 7400, 7420, 7440, 7460, 7480, 7500, 7520, 7540, 7560, 7580, 7600, 7620, 7640, 7660, 7680, 7700, 7720, 7740, 7760, 7780, 7800, 7820, 7840, 7860, 7880, 7900, 7920, 7940, 7960, 7980, 8000, 8020, 8040, 8060, 8080, 8100, 8120, 8140, 8160, 8180, 8200, 8220, 8240, 8260, 8280, 8300, 8320, 8340, 8360, 8380, 8400, 8420, 8440, 8460, 8480, 8500, 8520, 8540, 8560, 8580, 8600, 8620, 8640, 8660, 8680]
    frames_from_v = get_selected_frames(selected_f,all_frames)
    dec = detect_drone_movement(frames_from_v,selected_f)
    logging.info(f"detected total: {len(dec)}")
    logging.info(f"detected frames: {dec}")
    # save_frames_as_photos(get_selected_frames(dec,all_frames),new_path)


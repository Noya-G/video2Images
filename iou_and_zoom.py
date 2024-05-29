import cv2
import numpy as np


def calculate_iou_and_show_overlap(image1, image2, ratio_threshold=0.75, min_good_matches=10, min_inliers=10):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    print(len(good_matches), min_good_matches)
    if len(good_matches) < min_good_matches:
        print("Not enough good matches found. Returning IoU of 0.")
        return 0.0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum()
    print(inliers, min_inliers)
    if inliers < min_inliers:
        print("Not enough inliers found. Returning IoU of 0.")
        return 0.0
    h, w, _ = image1.shape
    warped_image2 = cv2.warpPerspective(image2, M, (w, h))
    intersection = cv2.bitwise_and(image1, warped_image2)
    union = cv2.bitwise_or(image1, warped_image2)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    # visualise overlap
    overlap_image = cv2.addWeighted(image1, 0.5, warped_image2, 0.5, 0)
    cv2.imshow('Overlap Area', overlap_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return iou

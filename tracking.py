# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from cs231aApproachingOdt import utils as myutils
from PIL import Image
import os
def match_detections(prev_path, prev_detection, new_path, new_detection, size=(640, 480)):
    prev_range = [*range(len(prev_detection))]
    new_range = [*range(len(new_detection))]

    permutations = myutils.unique_permutations(prev_range, new_range)

    fig, ax = plt.subplots(1, 2)
    prev_img = myutils.load_resize(prev_path, size)
    new_img = myutils.load_resize(new_path, size)

    for old, new in permutations:
        [a.cla() for a in ax]
        draw_detection(prev_img, prev_detection[old], ax[0])
        ax[0].set_title(f"{os.path.basename(prev_path)}")

        draw_detection(new_img, new_detection[new], ax[1])
        ax[1].set_title(f"{os.path.basename(new_path)}")
        plt.pause(0.1)

        prev_crop = crop_detection(prev_img, prev_detection[old])
        new_crop = crop_detection(new_img, new_detection[new])

        keypoint_matching(prev_crop, new_crop)
    plt.close(fig)

def keypoint_matching(img1, img2):
    # Source: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    myutils.show(img1_gray)
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig_match, ax_match = plt.subplot()
    plt.imshow(img3)
    plt.show()
    plt.close(fig_match)

def crop_detection(img, detection):
    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    return crop

def draw_detection(img, detection, ax):
    ax.imshow(myutils.bgr2rgb(img))
    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
    box_w = x2 - x1
    box_h = y2 - y1
    # Create a Rectangle patch
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
    # Add the bbox to the plot
    ax.add_patch(bbox)
    ax.set_xticks([])
    ax.set_yticks([])

def tracking_by_detection(image_paths, img_detections, size=(640, 480)):
    # Iterate through images and save plot of detections
    print("In Tracking By Detection")
    path_detections_zip = zip(image_paths, img_detections)
    num_images = len(image_paths)
    tqdm_pbar = tqdm.tqdm(path_detections_zip, total=num_images)

    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})
        if img_i == 0:
            print("Initialize Detections")
            continue

        match_detections(prev_path=image_paths[img_i - 1], prev_detection=img_detections[img_i - 1],
                         new_path=path, new_detection=detections, size=size)

# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from cs231aApproachingOdt import utils as myutils
from PIL import Image
import os

import torch
import torchvision.ops.boxes as bops

def match_detections(prev_path, prev_detection, new_path, new_detection, size=(640, 480)):
    prev_range = [*range(len(prev_detection))]
    new_range = [*range(len(new_detection))]

    permutations = myutils.unique_permutations(prev_range, new_range)

    fig, ax = plt.subplots(1, 2)
    prev_img = myutils.load_resize(prev_path, size)
    new_img = myutils.load_resize(new_path, size)

    matching_pairs = []
    for old, new in permutations:
        [a.cla() for a in ax]
        draw_detection(prev_img, prev_detection[old], ax[0])
        ax[0].set_title(f"{os.path.basename(prev_path)}")

        draw_detection(new_img, new_detection[new], ax[1])
        ax[1].set_title(f"{os.path.basename(new_path)}")
        #plt.pause(0.1)
        iou = get_iou(prev_detection[old], new_detection[new])

        if iou < 0.7:
            continue
        prev_crop = crop_detection(prev_img, prev_detection[old])
        new_crop = crop_detection(new_img, new_detection[new])
        #keypoint_matching(prev_crop, new_crop)

        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        is_match = template_matching(new_crop, prev_crop, methods[3])

        if is_match == True:
            matching_pairs.append((old, new))


    plt.close(fig)
    return matching_pairs

def get_iou(prev_detection, new_detection):
    box1 = new_detection[:4].reshape((1, 4))
    box2 = prev_detection[:4].reshape((1, 4))
    iou = bops.box_iou(box1, box2)
    return iou

def template_matching(img1, template, method):
    fig_template, ax = plt.subplots()

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img = img1_gray.copy()
    w_t, h_t = template_gray.shape[::-1]
    w_i, h_i = img1_gray.shape[::-1]

    if (w_t > w_i) or (h_t > h_i):
        return False

    method = eval(method)
    # Apply template Matching
    res = cv2.matchTemplate(img1_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(f"\n{min_val}, {max_val}, {min_loc}, {max_loc}")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(img, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(method)
    # plt.show()
    # plt.close(fig_template)

    if max_val > 0.9:
        return True
    else:
        return False

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

def tracking_by_detection(img_folder, image_paths, img_detections, size=(640, 480)):
    # Iterate through images and save plot of detections
    print("In Tracking By Detection")
    path_detections_zip = zip(image_paths, img_detections)
    num_images = len(image_paths)
    tqdm_pbar = tqdm.tqdm(path_detections_zip, total=num_images)

    tracks_dict = dict()
    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})
        if img_i == 0:
            print("Initialize Detections")
            continue

        matching_pairs = match_detections(prev_path=image_paths[img_i - 1], prev_detection=img_detections[img_i - 1],
                         new_path=path, new_detection=detections, size=size)
        print(matching_pairs)
        tracks_dict[path] = matching_pairs

    myutils.pickle_save(os.path.join(img_folder, "output/tracks.pickle"), (tracks_dict, img_detections))
    return tracks_dict
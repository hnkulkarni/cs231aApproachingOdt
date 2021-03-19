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

    f, ax = plt.subplots(1, 2)
    prev_img = myutils.load_resize(prev_path, size)
    new_img = myutils.load_resize(new_path, size)

    for old, new in permutations:
        [a.cla() for a in ax]
        draw_detection(prev_img, prev_detection[old], ax[0])
        ax[0].set_title(f"{os.path.basename(prev_path)}")
        draw_detection(new_img, new_detection[new], ax[1])
        ax[1].set_title(f"{os.path.basename(new_path)}")
        plt.pause(0.1)


def draw_detection(img, detection, ax):
    ax.imshow(myutils.bgr2rgb(img))
    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
    box_w = x2 - x1
    box_h = y2 - y1
    # Create a Rectangle patch
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
    # Add the bbox to the plot
    ax.add_patch(bbox)

def tracking_by_detection(image_paths, img_detections, size=(640, 480)):
    # Iterate through images and save plot of detections
    print("In Tracking By Detection")
    path_detections_zip = zip(image_paths, img_detections)
    num_images = len(image_paths)
    tqdm_pbar = tqdm.tqdm(path_detections_zip, total=num_images)

    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})
        print(path)
        if img_i == 0:

            print("Initialize Detections")
            continue

        match_detections(prev_path=image_paths[img_i - 1], prev_detection=img_detections[img_i - 1],
                         new_path=path, new_detection=detections, size=size)

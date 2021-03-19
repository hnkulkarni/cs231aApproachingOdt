# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cs231aApproachingOdt import utils as myutils
from PIL import Image

def match_detections(prev_path, prev_detection, new_path, new_detection, size=(640, 480)):
    prev_range = [*range(len(prev_detection))]
    new_range = [*range(len(new_detection))]

    permutations = myutils.unique_permutations(prev_range, new_range)

    f, ax = plt.subplots(1, 2)
    prev_img = myutils.load_resize(prev_path, size)
    new_img = myutils.load_resize(new_path, size)

    for old, new in permutations:
        ax[0].imshow(prev_img, cmap=)
        ax[1].imshow(new_img)


def tracking_by_detection(image_paths, img_detections, size=(640, 480)):
    # Iterate through images and save plot of detections
    print("In Tracking By Detection")
    fig, ax = plt.subplots()
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

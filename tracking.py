# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def match_detections(old_detection, new_detection):
    x = 2

def tracking_by_detection(image_paths, img_detections):
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

        for det in img_detections[img_i - 1]:
            match_detections(old_detection=det, new_detection=detections)

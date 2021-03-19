# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#def match_detections(old_detection, new_detection):

def tracking_by_detection(image_paths, img_detections):
    # Iterate through images and save plot of detections
    print("In Tracking By Detection")
    fig, ax = plt.subplots()
    path_detections = zip(image_paths, img_detections)
    tqdm_pbar = tqdm.tqdm(path_detections, total=len(image_paths))
    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})
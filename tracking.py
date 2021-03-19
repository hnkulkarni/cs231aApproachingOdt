# This file will track detections
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#def match_detections(old_detection, new_detection):

def tracking_by_detection(image_paths, img_detections):
    # Iterate through images and save plot of detections
    fig, ax = plt.subplots()
    for img_i, (path, detections) in enumerate(tqdm.tqdm(zip(image_paths, img_detections))):
        print(f"{img_i} : {path}")
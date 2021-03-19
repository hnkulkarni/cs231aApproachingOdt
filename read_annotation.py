import numpy as np
import argparse
from pathlib import Path
import os
from PIL import Image, ImageDraw
import psutil
import cv2
from collections import defaultdict

# From: https://data.vision.ee.ethz.ch/cvl/aess/dataset/
# For each image, it lists a set of bounding boxes, separated by commas.
# The boxes contain upper-left and lower-right corner,
# but are not necessarily sorted according to this.
# A semicolon ends the list of bounding boxes for a single file, a period ends the file.
# "filename": (x1, y1, x2, y2), (x1, y1, x2, y2), ...;


def annotate_approaching(filepath):

    # Using readlines()
    file1 = open(filepath, 'r')
    Lines = file1.readlines()
    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        line = line.strip()
        img_path = get_imagepath(filepath, line)
        detections = get_detections(line)
        approaching_detections = label_detections(img_path, detections)
        write_detections(filepath, line, approaching_detections)

def read_detections(filepath):
    print(f"Reading annotations from {filepath}")
    # Using readlines()
    file1 = open(filepath, 'r')
    Lines = file1.readlines()
    count = 0

    detections = defaultdict(list)

    # Strips the newline character
    for line in Lines:
        count += 1
        line = line.strip()
        img_path = get_imagepath(filepath, line)
        detections[img_path] = get_detections(line)

    file1.close()
    return detections

def write_detections(filepath, line, approaching_detections):
    p = Path(filepath)
    annotation_path = Path(p.parents[0]).joinpath(f"{p.stem}_approaching.idl").__str__()
    name = line.split(":")[0].replace('"', '')
    detstrings = [convert_2_datasetformat(x) for x in approaching_detections]
    new_line = name + ":" + ",".join(detstrings) + ".\n"
    f = open(annotation_path, "a+")
    f.write(new_line)
    f.close()

def convert_2_datasetformat(np_detection):
    s = np.array2string(np_detection, separator=",")
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    return s

def load(f):
    img = cv2.imread(f)
    return img

def show(img):
    cv2.imshow('image', img)
    waitkey = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return chr(waitkey)

def label_detections(image_path, detections):

    img = load(image_path)
    approaching_detections = []

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d
        img_draw = img.copy()
        img_draw = cv2.rectangle(img_draw, (x1, y1), (x2,y2), (0,255,0),3)
        h = np.abs(y1 - y2)
        w = np.abs(x1 - x2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_draw = cv2.putText(img_draw, f"h:{h} w:{w}", (10, 300), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        is_approaching = show(img_draw)

        if is_approaching == "y":
            approaching_detections.append(d)

    return approaching_detections

def get_detections(line):
    detections_list_str = line.split(":")[1].strip().split("),")
    detections = []
    for d in detections_list_str:
        dsingle_str = d.strip().replace("(", "")
        dsingle_str = dsingle_str.replace(");", "")
        dsingle_np = np.fromstring(dsingle_str, sep=",", dtype=int)
        detections.append(dsingle_np)
    return np.asarray(detections)


def get_imagepath(filepath, line):
    p = Path(filepath)
    annotation_dirpath = p.parents[0].__str__()
    filename = line.split(":")[0].replace('"', '')

    # List all the folders
    camera_name = filename.split("/")[0]
    image_name = filename.split("/")[1]
    img_fldr = get_image_folder(camera_name, annotation_dirpath)
    image_path = os.path.join(img_fldr, image_name)
    return image_path


def get_image_folder(camera_name, annotation_dirpath):
    folders = [x[0] for x in os.walk(annotation_dirpath)]
    # Find the ones with end with left
    for fdr in folders:
        if fdr.endswith(camera_name):
            # We just need one folder hence returning over here
            return fdr


# def main():
#     parser = argparse.ArgumentParser(description="Lets make world a better place")
#     parser.add_argument("--filepath", type=str, help="File containing all annotations")
#
#     args = parser.parse_args()
#     annotate_approaching(args.filepath)
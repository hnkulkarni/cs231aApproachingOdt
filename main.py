## This is the code for the CS231A class project to detect approaching objects
import argparse

import pandas as pd
import tqdm
import numpy as np

import ObjDetector

def main():
    print(f"The main of the project")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")

    opt = parser.parse_args()
    print(opt)

    imgs, img_detections = ObjDetector.detections(image_folder=opt.image_folder, batch_size=24)
    print(imgs)
    print(img_detections)


if __name__ == '__main__':
    main()
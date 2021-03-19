## This is the code for the CS231A class project to detect approaching objects
import argparse
from cs231aApproachingOdt import ObjDetector
from cs231aApproachingOdt import MonacularDepthAdabins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")

    opt = parser.parse_args()
    print(opt)

    imgs, img_detections = ObjDetector.detections(image_folder=opt.image_folder, batch_size=24)
    bin_centers, predicted_depth, depth_viz = MonacularDepthAdabins.get_depth(imgs[0])
    print(bin_centers)

if __name__ == '__main__':
    main()

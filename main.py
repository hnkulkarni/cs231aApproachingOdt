## This is the code for the CS231A class project to detect approaching objects
import argparse
from cs231aApproachingOdt import ObjDetector
from cs231aApproachingOdt import MonacularDepthAdabins
from cs231aApproachingOdt import tracking
from cs231aApproachingOdt import utils as myutils
from PIL import Image
import cs231aApproachingOdt.mytracking as mytracking
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")

    opt = parser.parse_args()
    print(opt)

    files = myutils.files_from_folder(folderpath=opt.image_folder, ext=".png")
    pil_img = Image.open(files[0])

    image_paths, img_detections = ObjDetector.detections(image_folder=opt.image_folder,
                                                         batch_size=24,
                                                         size=pil_img.size)
    #bin_centers, predicted_depth, depth_viz = MonacularDepthAdabins.get_depth(image_paths[0])
    pickle_track = os.path.join(opt.image_folder, "output/tracks.pickle")
    if os.path.exists(pickle_track):
        tracks_dict, img_detections = myutils.pickle_load(pickle_track)
    else:
        tracks_dict = tracking.tracking_by_detection(img_folder=opt.image_folder,
                                       image_paths=image_paths, img_detections=img_detections, size=pil_img.size)
        #myt = mytracking.TracksSet(image_paths[0], img_detections[0])


    MonacularDepthAdabins.detections_depth(image_paths, img_detections )



    print(tracks_dict)
if __name__ == '__main__':
    main()

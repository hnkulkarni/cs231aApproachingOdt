from AdaBins.infer import InferenceHelper
from PIL import Image
import tqdm
from cs231aApproachingOdt import opflow
import numpy as np
from cs231aApproachingOdt import utils as myutils
import matplotlib.pyplot as plt
from cs231aApproachingOdt import tracking

from cs231aApproachingOdt import audio
from multiprocessing import Pool

from cs231aApproachingOdt import read_annotation

def get_depth(imagepath):
    infer_helper = InferenceHelper(dataset='kitti')
    pil_img = Image.open(imagepath)
    pil_img = pil_img.resize((640,480))
    bin_centers, predicted_depth, depth_viz = infer_helper.predict_pil(pil_img, visualized=True)
    return bin_centers, predicted_depth, depth_viz

def crop_detection(img, detection):
    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    return crop

def detections_depth(image_paths, img_detections):
    path_detections_zip = zip(image_paths, img_detections)
    num_images = len(image_paths)
    tqdm_pbar = tqdm.tqdm(path_detections_zip, total=num_images)

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    height_ratio = 0.3
    audio_ctr = 5
    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})

        if img_i == 0:
            print("Initialize Detections")
            continue

        im = myutils.load(path)
        img_h = im.shape[0]
        ax[0].imshow(myutils.bgr2rgb(im))
        ax[0].set_title("Input Image")

        f1 = image_paths[img_i - 1]
        f2 = path
        flow, magnitude, angle = opflow.get_flow(f1, f2)
        ax[1].imshow(myutils.bgr2rgb(flow))
        ax[1].set_title("Dense Optical Flow")

        bin_centers, predicted_depth, depth_viz = get_depth(path)
        ax[2].imshow(depth_viz)
        ax[2].set_title("Adabins-Depth")

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            tracking.draw_detection(im, [x1, y1, x2, y2, conf, cls_conf, cls_pred], ax[3])
            box_w = x2 - x1
            box_h = y2 - y1
            if (img_i > 0) and (box_h > img_h * height_ratio):
                cr = angle[int(y1):int(y1 + box_h), int(x1):int(x1 + box_w)]

                cr_mean = cr.mean()
                if np.isnan(cr_mean):
                    cr_mean = 0

                if cr_mean < 90:
                    mode = "Approaching"
                    detect_string =  f" {mode} Angle {int(cr_mean)}"
                else:
                    mode = "Leaving"
                    detect_string = f" {mode} Angle {int(cr_mean)}"

                # Add label
                ax[3].text(
                    x1,
                    y1,
                    s=detect_string,
                    color="white",
                    verticalalignment="top",
                    bbox={"color": 'red', "pad": 0},
                )
                ax[3].set_title("Approaching Object Detection")


                if mode == "Approaching" and audio_ctr == 0:
                    pool = Pool(processes=1)
                    play_str = detect_string
                    res = pool.apply_async(audio.play, (play_str,))
                    # audio.play(play_str)
                    audio_ctr = 5


            if audio_ctr > 0:
                audio_ctr -= 1

        plt.savefig(f"Output-{img_i}.png")
        plt.pause(0.1)
        plt.cla()








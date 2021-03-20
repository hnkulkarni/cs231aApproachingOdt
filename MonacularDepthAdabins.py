from AdaBins.infer import InferenceHelper
from PIL import Image
import tqdm
from cs231aApproachingOdt import opflow

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

def detections_depth(image_paths, img_detections)
    path_detections_zip = zip(image_paths, img_detections)
    num_images = len(image_paths)
    tqdm_pbar = tqdm.tqdm(path_detections_zip, total=num_images)

    for img_i, (path, detections) in enumerate(tqdm_pbar):
        tqdm_pbar.set_postfix({"Processing ": path})

        if img_i == 0:
            print("Initialize Detections")
            continue

        f1 = path[img_i - 1]
        f2 = path[img_i]
        flow, magnitude, angle = opflow.get_flow(f1, f2)




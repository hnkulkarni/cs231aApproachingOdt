import datetime
import os
import time
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from PyTorchYOLOv3.models import Darknet
from PyTorchYOLOv3.utils.augmentations import DEFAULT_TRANSFORMS, Resize
from PyTorchYOLOv3.utils.datasets import ImageFolder
from PyTorchYOLOv3.utils.utils import load_classes, non_max_suppression, rescale_boxes
from cs231aApproachingOdt import utils as myutils

def detections(image_folder, batch_size, size=(640, 480)):
    print("Getting Yolo Detections")
    YOLO_HOME = "/home/hnkulkarni/nn/opensource/PyTorchYOLOv3"
    model_def = os.path.join(YOLO_HOME, "config/yolov3.cfg")
    weights_path = os.path.join(YOLO_HOME, "weights/yolov3.weights")
    class_path = os.path.join(YOLO_HOME, "data/coco.names")
    nms_thres = 0.4
    img_size = 416
    conf_thres = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)
    # Load darknet weights
    model.load_darknet_weights(weights_path)

    # Set the model in Eval Mode
    model.eval()

    dataloader = DataLoader(
        ImageFolder(image_folder, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    classes = load_classes(class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure Input
        input_imgs = Variable(input_imgs.type(Tensor))

        # get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres=conf_thres, nms_thres=nms_thres)

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print(f" {batch_i} : Inference Time =  {inference_time}")

        # Append image paths and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    imgs, img_detections = scale_detections(imgs=imgs, img_detections=img_detections,
                                            img_size=img_size, size=size)
    return imgs, img_detections

def scale_detections(imgs, img_detections, img_size, size=(640, 480)):

    print(f"Resizing detections to {size}")

    pil_img = Image.open(imgs[0])
    pil_img = pil_img.resize(size)
    img = np.array(pil_img)

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            img_detections[img_i] = detections

    return imgs, img_detections

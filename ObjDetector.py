import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import time
import datetime

from PyTorchYOLOv3.models import Darknet
from PyTorchYOLOv3.utils.datasets import ImageFolder
from PyTorchYOLOv3.utils.augmentations import DEFAULT_TRANSFORMS, Resize
from PyTorchYOLOv3.utils.utils import load_classes, non_max_suppression

def detections(image_folder, batch_size):
    print("Getting Yolo Detections")
    YOLO_HOME = "/home/hnkulkarni/nn/opensource/PyTorchYOLOv3"
    model_def = os.path.join(YOLO_HOME, "config/yolov3.cfg")
    weights_path = os.path.join(YOLO_HOME,"weights/yolov3.weights")
    class_path = os.path.join(YOLO_HOME,"data/coco.names")
    nms_thres = 0.4
    img_size = 416
    conf_thres = 0.8


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
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
        num_workers=4,
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
    return imgs, img_detections



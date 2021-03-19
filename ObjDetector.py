import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import datetime

from PyTorchYOLOv3.models import Darknet
from PyTorchYOLOv3.utils.datasets import ImageFolder
from PyTorchYOLOv3.utils.augmentations import DEFAULT_TRANSFORMS, Resize
from PyTorchYOLOv3.utils.utils import load_classes

def detections(image_folder, batch_size):
    print("Getting Yolo Detections")
    model_def = "/home/hnkulkarni/nn/opensource/PyTorchYOLOv3/config/yolov3.cfg"
    weights_path = "/PyTorchYOLOv3/weights/yolov3.weights"
    class_path = "/PyTorchYOLOv3/data/coco.names"
    nms_thres = 0.4
    img_size = 416

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

    print("\nPerforming object detection:")
    prev_time = time.time()


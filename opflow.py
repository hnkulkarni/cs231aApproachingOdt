"""
Get Optical flow between two images
"""

import utils
import cv2
import numpy as np

def load(f):
    img = cv2.imread(f)
    return img

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_flow(f1, f2):
    im1 = load(f1)
    im2 = load(f2)

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(im1)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(im1_gray, im2_gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    #show(rgb)
    return rgb,mask[..., 2], mask[..., 0]




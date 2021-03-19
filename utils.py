import glob
import os
import natsort
import cv2


def files_from_folder(folderpath, ext):
    files = [f for f in glob.glob(os.path.join(folderpath, f"*{ext}"))]
    files = natsort.natsorted(files)
    return files

def load(f):
    img = cv2.imread(f)
    return img

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
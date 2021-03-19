import glob
import itertools
import os

import cv2
import natsort


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


def unique_permutations(list1, list2):
    # Source From: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
    result = [list(zip(x, list2)) for x in itertools.permutations(list1, len(list2))]
    return result

import cv2
import cs231aApproachingOdt.utils as myutils
# BOOSTING, MIL, KCF, TLD, MEDIANFLOW
class TracksSet():
    # Adapted from: https://learnopencv.com/object-tracking-using-opencv-cpp-python/
    def __init__(self, filepath, detections):
        frame = myutils.load(filepath)
        self.alltracks = []
        if detections is not None:
            for detection in detections:
                single_track = Track(frame, detection)
                self.alltracks.append(single_track)



class Track():
    def __init__(self):
        self
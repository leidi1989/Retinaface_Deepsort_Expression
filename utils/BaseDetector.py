'''
Description: 
Version: 
Author: Leidi
Date: 2021-03-12 18:10:49
LastEditors: Leidi
LastEditTime: 2021-03-15 10:54:03
'''
from tracker import update_tracker
import cv2


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):

        retDict = {
            'frame': None,
            'boxes': None
        }
        self.frameCounter += 1

        im, outputs = update_tracker(self, im)

        retDict['frame'] = im
        retDict['boxes'] = outputs
        
        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")

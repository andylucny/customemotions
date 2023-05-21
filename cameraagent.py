import cv2 as cv
import numpy as np
from agentspace import Agent, space

class CameraAgent(Agent):

    def __init__(self,cameraId,nameImage):
        self.cameraId = cameraId
        self.nameImage = nameImage
        super().__init__()

    def init(self): 
        self.camera = cv.VideoCapture(0)
        while True:
            hasFrame, frame = self.camera.read() 
            if hasFrame: 
                space(validity=0.15)[self.nameImage] = frame
    
    def senseSelectAct(self):
        pass


# import the necessary packages
import numpy as np
import cv2 as cv 
from agentspace import Agent, space

class EmotionAgent(Agent):

    def __init__(self,nameImage,nameFacePosition,nameFaceImage,nameFaceEmotion):
        self.nameImage = nameImage
        self.nameFacePosition = nameFacePosition
        self.nameFaceImage = nameFaceImage
        self.nameFaceEmotion = nameFaceEmotion
        super().__init__()

    def init(self): 
        print("faceDetector: loading model")
        face_architecture = 'deploy.prototxt'
        face_weights = 'res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv.dnn.readNetFromCaffe(face_architecture,face_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("faceDetector: model loaded")

        print("emotionDetector: loading model")
        self.net2 = cv.dnn.readNet('mobilenet_7_backbone.pbtxt','mobilenet_7_backbone.pb')
        print("emotionDetector: model loaded")

        space.attach_trigger(self.nameImage, self)

    def senseSelectAct(self):
        image = space[self.nameImage]
        if image is None:
            return

        height = 300
        width = 300
        mean = (104.0, 177.0, 123.0)
        threshold = 0.5
        h, w = image.shape[:2] 

        # convert to RGB
        rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)

        # blob preparation
        blob = cv.dnn.blobFromImage(cv.resize(image,(width,height)),1.0,(width,height),mean)

        # passing blob through the network to detect and pridiction
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        rects = []
        faces = []
        for i in range(detections.shape[2]):
            # extract the confidence and prediction
            confidence = detections[0, 0, i, 2]
            # filter detections by confidence greater than the minimum
            if confidence > threshold:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype(np.int)
                if startX < 0:
                    startX = 0
                if startY < 0:
                    startY = 0
                if endX > w:
                    endX = w
                if endY > h:
                    endY = h
                if (startY+1 < endY) and (startX+1 < endX):
                    rects.append((startX, startY, endX, endY, confidence))
                    face = np.copy(image[startY:endY,startX:endX,:])
                    faces.append(face)
        
        result = np.copy(image)

        if len(rects) > 0:
            
            # select the best face
            best = np.argmin([rect[4] for rect in rects])

            # transform the image to suitable input of the MobileNet DNN
            blob = cv.dnn.blobFromImage(faces[best], 1.0, (224, 224), (123.68, 116.779, 103.939), True, False)
            # put the input to the network
            self.net2.setInput(blob)
            # launch the network and get the produced output
            features = self.net2.forward()[0]

            # output the find info to the blackboard
            space(validity=0.2)[self.nameFacePosition] = rects[best][:4]
            space(validity=0.2)[self.nameFaceImage] = faces[best]
            space(validity=0.2)[self.nameFaceEmotion] = features

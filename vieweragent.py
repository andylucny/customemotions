from agentspace import Agent, space
import cv2 as cv
import time

class ViewerAgent(Agent):

    def __init__(self, nameImage, nameRoi, nameEmotion, nameInvitation):
        self.nameImage = nameImage
        self.nameRoi = nameRoi
        self.nameEmotion = nameEmotion
        self.nameInvitation = nameInvitation
        super().__init__()

    def init(self):
        self.labels = None
        labelsFile = "labels.txt"
        with open(labelsFile, 'rt') as f:
            self.labels = f.read().rstrip('\n').split('\n')
        space.attach_trigger(self.nameImage,self)
            
    def senseSelectAct(self):
        image = space[self.nameImage]
        if image is None:
            return

        roi = space[self.nameRoi]
        if roi is not None:
            startX, startY, endX, endY = roi
            cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
            emotion = space[self.nameEmotion]
            if emotion is not None:
                cv.putText(image,self.labels[emotion],(10,32),0,1.5,(0,0,255),2)

            invitation = space[self.nameInvitation]
            if invitation is not None:
                cv.putText(image,'expose '+self.labels[invitation],(image.shape[1]//2,32),0,0.8,(255,255,255),2)

        cv.imshow("custom emotions",image)
        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite(str(time.time())+'.png',image)

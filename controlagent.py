from agentspace import Agent, space
import numpy as np
import time
from beeply.notes import beeps
from speak import speak

def rad(fi):
    return np.pi * fi / 180.0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def Attention(query,keys,values,d):
    keys_matrix = np.array(keys,np.float32)
    values_matrix = np.array(values,np.float32)
    c = softmax(query.dot(keys_matrix.T)/d)
    output = c.dot(values_matrix)
    return output, c

class ControlAgent(Agent):

    def __init__(self,nameFeatures,nameEmotion,nameInvitation):
        self.nameFeatures = nameFeatures
        self.nameEmotion = nameEmotion
        self.nameInvitation = nameInvitation
        super().__init__()

    def init(self):
        fis = [180,270,220,110,0,310,70]
        self.keys = []
        self.values = np.array([[np.cos(rad(fi)),np.sin(rad(fi))] for fi in fis],np.float32)
        space.attach_trigger(self.nameFeatures,self)

    def senseSelectAct(self):
        features = space[self.nameFeatures]
        if features is not None:
            if space(default=False)[self.nameInvitation+'-confirmed']:
                self.keys.append(features)
                space[self.nameInvitation+'-confirmed'] = False
            if len(self.keys) > 0:
                embed, _ = Attention(features,self.keys,self.values[:len(self.keys)],len(features)**0.5)
                _, coefs = Attention(embed,self.values,self.values,1.0/len(embed))
                emotion = np.argmax(coefs)
                space(validity=0.4)[self.nameEmotion] = emotion

class InvitationAgent(Agent):

    def __init__(self,nameFeatures,nameInvitation):
        self.nameFeatures = nameFeatures
        self.nameInvitation = nameInvitation
        super().__init__()

    def init(self):
        self.labels = None
        labelsFile = "labels.txt"
        with open(labelsFile, 'rt') as f:
            self.labels = f.read().rstrip('\n').split('\n')
        beeper = beeps(250)
        for i in range(len(self.labels)):
            space(validity=4)[self.nameInvitation] = i
            speak("Expose "+self.labels[i])
            time.sleep(1)
            beeper.hear("C_")
            time.sleep(1)
            beeper.hear("C_")
            time.sleep(1)
            beeper.hear("C__")
            space[self.nameInvitation+'-confirmed'] = True
            time.sleep(3)

    def senseSelectAct(self):
        pass
 
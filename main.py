from agentspace import space
import signal 
import os
import time
def quit():
    os._exit(0)

def signal_handler(signal, frame):
    quit()
    
signal.signal(signal.SIGINT, signal_handler)

from cameraagent import CameraAgent
from emotionagent import EmotionAgent
from controlagent import ControlAgent, InvitationAgent
from vieweragent import ViewerAgent

from download import download_espeak
download_espeak()

print('starting camera')
CameraAgent(1,'colorImage')
time.sleep(1)
print('starting face detector')
EmotionAgent('colorImage','facePosition','faceImage','features')
time.sleep(1)
print('starting control')
ControlAgent('features','emotion','invitation')
InvitationAgent('features','invitation')
time.sleep(1)
ViewerAgent('colorImage','facePosition','emotion','invitation')
print('started')

# os._exit(0)


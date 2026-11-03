import initSettings
from camera import Camera as Camera
import dModel
from math import exp as exp
import math

dt = 0.1
vid0 = []
vid1 = []
for i in range(1000):
    vid0.append((dt * i, exp(dt * i)))
    vid1.append((0, exp(dt * i) + 2))

initSettings.Cameras[0] = Camera(vid=vid0, xLoc=0, yLoc=0, compassOr=90, heightOr=90)
initSettings.Cameras[1] = Camera(vid=vid1, xLoc=0, yLoc=0, compassOr=180, heightOr=90)
print(f"vid0 = {vid0}")
print(f"vid1 = {vid1}")
print(initSettings.Cameras)
dModel.model()
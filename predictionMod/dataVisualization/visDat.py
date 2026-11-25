import cv2
import numpy as np
from predictionMod.dataPreparation.dataPreparation import loadTrain, loadTest
import predictionMod.latestModelAPI

def visualData(data, label):
    # creating a blank image
    width = 1920 // 2
    height = 1080 // 2

    predictionMod.latestModelAPI.initVid()

    for frame in data:
        start = (int(frame[0][2].item() * width), int(frame[0][3].item() * height))
        end = (int(frame[0][0].item() * width), int(frame[0][1].item() * height))

        baseImg = np.zeros((height, width, 3), dtype=np.uint8)
        # adding the bottle arrow
        img = cv2.arrowedLine(baseImg, start, end, (0, 0, 255), 3)
        # adding the true z value
        img = cv2.line(img, (0, int(label.item() * height)), (width, int(label.item() * height)), (0, 255, 0), 3)

        pred = predictionMod.latestModelAPI.predictFrame(frame)
        print("-" * 30)
        print(frame)
        print(f"pred = {pred[0][0]}")
        print(label.item(), (label.item() - pred[0][0]).item())

        img = cv2.line(img, (0, int(pred * height)), (width, int(pred * height)), (255, 0, 0), 3)

        # Display the image
        cv2.imshow('state', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

trainDataset = loadTrain()

while True:
    lmts = len(trainDataset) - 1
    userInput = -1

    while int(userInput) < 0 or int(userInput) > lmts:
        userInput = input(f"enter Vid num(0~{lmts}):")
        try:
            userInput = int(userInput)
        except:
            userInput = -1
    vid, label = trainDataset.getVid(userInput)

    visualData(vid, label)
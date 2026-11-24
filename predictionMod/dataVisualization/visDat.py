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


[20.70285479987867, 16.16210950407789, 15.813707526728635, 12.825892500252195, 11.267226606590143, 10.538021304408245, 8.175803980762057, 7.5512279619268465, 6.219606152025776, 5.735835383393611, 5.563258598104946, 5.0882328671343595, 4.677481154388672, 4.438625662384532, 4.0513065516620905, 4.154486015228266, 4.200032600190017, 4.416108692815677, 4.3395106990387005]
[3711.375643524695, 3172.12402619862, 2740.694361706736, 2335.103481581196, 1993.6613399759476, 1674.8659985277518, 1422.5337191796825, 1182.3275464517196, 1015.3132304390411, 900.1641295574348, 835.6745910477548, 800.2001432559733, 769.0755023798608, 740.3056985290849, 723.1429082827282, 721.067447426163, 727.8693689028354, 711.7666852012045, 715.1550517463813]
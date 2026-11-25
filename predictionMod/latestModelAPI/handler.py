from .model import model
import torch

hidden = None

import operator

def rindex(lst, value):
    return len(lst) - operator.indexOf(reversed(lst), value) - 1

def initVid():
    global hidden
    hidden = model.init_hidden()

def csvToTensors(path):
    inputSize = 7
    outputSize = 2

    with open(path, "r") as file:
        dataLines = file.readlines()

    counter = 0
    tens = {}
    zTrues = {}

    for data in dataLines:
        data = data.split(",")
        # removing the name and the seperator "2"
        # print(data)
        try:
            data = data[1:rindex(data, "2")]
        except:
            data = data[1:rindex(data, "2\n")]

        # separating the data from the labels
        data, label = data[2:], data[:2]

        # creating the tensor in the rite shape
        # creating an empty tensor in the right shape
        vidTensor = torch.zeros((int(len(data) / inputSize), 1, inputSize))
        for frameNum in range(int(len(data) / inputSize)):
            # loading the data
            for argIdx in range(inputSize):
                vidTensor[frameNum][0][argIdx] = float(data[frameNum * inputSize + argIdx])

        labelTensor = torch.zeros((1, inputSize))
        for argIdx in range(outputSize):
            labelTensor[0][argIdx] = float(label[argIdx])

        if vidTensor.size(0) > 60:
            zTrues[counter] = labelTensor
            tens[counter] = vidTensor
            counter += 1

    return tens, zTrues

def predictFrame(frame):
    global hidden
    output, hidden = model(frame, hidden)

    return output

def predToHeight(pred):
    """
    this function transform the model's prediction into real-world height
    """
    W_px=1920
    H_px=1080
    z_bottom_px = 716
    z_middle_px = 624
    z_top_px = 527
    bottle_offset_cm = 10.25


    z_bottom_norm = (H_px - z_bottom_px) / H_px
    z_middle_norm= (H_px - z_middle_px) / H_px
    z_top_norm = (H_px - z_top_px) / H_px

    delta_z_cm = 22 # CHANGE ME
    delta_z_norm = z_top_norm - z_bottom_norm

    NORM_FACTOR = (delta_z_cm / delta_z_norm)
    pred_norm = (H_px - pred) / H_px

    return (pred_norm - z_middle_norm) * NORM_FACTOR - bottle_offset_cm
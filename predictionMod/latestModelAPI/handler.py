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

    z_bottom = 0.1 # CHANGE ME
    z_top = 0.1  # CHANGE ME
    z_rest = 0.5 # CHANGE ME
    delta_z_cm = 40 # CHANGE ME
    delta_z_pixel = z_top - z_bottom

    H = (delta_z_cm / delta_z_pixel)

    return (pred - z_rest) * H
from .model import model
import torch

hidden = None

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
            data = data[1:data.index("2")]
        except:
            data = data[1:data.index("2\n")]

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

def processFrame(frame):
    global hidden
    return model(frame, hidden)
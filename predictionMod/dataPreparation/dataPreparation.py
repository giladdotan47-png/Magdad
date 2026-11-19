import random
from .constants import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
import numpy as np
import os

def splitSingleFile(training_data_path, labels_data_path) -> None:
    # loading data
    with open(training_data_path, "r") as file:
        newDataLines = file.readlines()

    # loading labels
    with open(labels_data_path, "r") as file:
        newLabelsLines = file.readlines()
    if len(newDataLines) != len(newLabelsLines): raise MemoryError("data and labels don't have same number of lines")
    n = len(newLabelsLines)

    # calculating how much lines each dataset will get
    trainingLines = int(TRAIN_SIZE * n)
    validationLines = int(VAL_SIZE * n)

    # adding the new training data
    with open(TRAINING_DATA_FILENAME, "r+") as file:
        oldTrainingData = file.readlines()
        oldTrainingData.extend(newDataLines[:trainingLines])
        file.writelines(oldTrainingData)

    # adding the new training labels
    with open(TRAINING_LABELS_FILENAME, "r+") as file:
        oldTrainLabels = file.readlines()
        oldTrainLabels.extend(newLabelsLines[:trainingLines])
        file.writelines(oldTrainLabels)

    # adding the new validation data
    with open(VALIDATION_DATA_FILENAME, "r+") as file:
        oldValData = file.readlines()
        oldValData.extend(newDataLines[trainingLines:trainingLines+validationLines])
        file.writelines(oldValData)

    # adding the new validation labels
    with open(VALIDATION_LABELS_FILENAME, "r+") as file:
        oldValLabels = file.readlines()
        oldValLabels.extend(newLabelsLines[trainingLines:trainingLines+validationLines])
        file.writelines(oldValLabels)

    # adding the new test data
    with open(TEST_DATA_FILENAME, "r+") as file:
        oldTestData = file.readlines()
        oldTestData.extend(newDataLines[trainingLines+validationLines:])
        file.writelines(oldTestData)

    # adding the new test labels
    with open(TEST_LABELS_FILENAME, "r+") as file:
        oldTestLabels = file.readlines()
        oldTestLabels.extend(newLabelsLines[trainingLines+validationLines:])
        file.writelines(oldTestLabels)

def splitNewData() -> None:


    # specifying the path to csv files
    path = "newData"

    # csv files in the path
    print(f'opening {NEW_DATA_PATH + "/*.csv"}')
    dataFiles = glob.glob(NEW_DATA_PATH + "/*.csv")
    labelsFiles = glob.glob(NEW_LABELS_PATH + "/*.csv")

    for dataFilename, labelsFilename in zip(dataFiles, labelsFiles):
        # reading content of csv file
        # content.append(filename)
        splitSingleFile(training_data_path=dataFilename, labels_data_path=labelsFilename)


# creating pytorch's datasets
class bottleDataset(Dataset):
    _filename = None
    _xdf = None
    _ydf = None
    _dataInputSize = None
    _labelsInputSize = None
    _xTensors = {}
    _yTensors = {}
    _name = ""

    def __init__(self, data_path, labels_path, dataInputSize, labelsInputSize):
        self._filename = data_path

        self._dataInputSize = dataInputSize
        self._labelsInputSize = labelsInputSize

        # opening the data and the labels
        print(f"loading from \"{data_path}\"")
        print(f"loading from \"{labels_path}\"")
        self._loadDatasetToTensors(data_path, labels_path)

        print(len(self._xTensors.keys()), len(self._yTensors.keys()))

    def _loadDatasetToTensors(self, data_path, labels_path):
        with open(data_path, "r") as file:
            dataLines = file.readlines()
        with open(labels_path, "r") as file:
            labelsLines = file.readlines()

        counter = 0
        for data, label in zip(dataLines, labelsLines):
            data = data.split(",")
            label = label.split(",")
            # removing the name and the seperator "2"
            data = data[1:-1]
            label = label[1:]

            # creating the tensor in the rite shape
            vidTensor = torch.zeros((int(len(data) / self._dataInputSize), 1, self._dataInputSize))
            for frameNum in range(int(len(data) / self._dataInputSize)):
                # loading the data
                for argIdx in range(self._dataInputSize):
                    vidTensor[frameNum][0][argIdx] = float(data[frameNum * self._dataInputSize + argIdx])

            labelTensor = torch.zeros((1, self._labelsInputSize))
            for argIdx in range(self._labelsInputSize):
                labelTensor[0][argIdx] = float(label[argIdx])

            if vidTensor.size(0) > 60:
                self._yTensors[counter] = labelTensor
                self._xTensors[counter] = vidTensor
                counter += 1


    def __len__(self):
        return len(self._xTensors.items())

    def getRand(self):
        idx = random.randint(0, len(self._xTensors) - 1)
        return self.getVid(idx)


    def getVid(self, idx):
        return self._xTensors[idx], self._yTensors[idx]

    def __getitem__(self, idx):
        return self.getVid(idx)


def loadSets():
    # creating pytorch's dataloaders
    trainDataset = bottleDataset(data_path=TRAINING_DATA_FILENAME, labels_path=TRAINING_LABELS_FILENAME,
                                 dataInputSize=7, labelsInputSize=2)

    validationDataset = bottleDataset(data_path=VALIDATION_DATA_FILENAME, labels_path=VALIDATION_LABELS_FILENAME,
                                 dataInputSize=7, labelsInputSize=2)

    testDataset = bottleDataset(data_path=TEST_DATA_FILENAME, labels_path=TEST_LABELS_FILENAME,
                                      dataInputSize=7, labelsInputSize=2)

    return trainDataset, validationDataset, testDataset
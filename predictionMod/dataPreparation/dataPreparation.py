import random
from .constants import *
import torch
from torch.utils.data import Dataset
import glob
import operator

def _splitSingleFile(training_data_path) -> None:
    """
    this file loads a single file of data, and splits it between the training/validation/test datasets
    """
    # loading data
    with open(training_data_path, "r") as file:
        newDataLines = file.readlines()

    n = len(newDataLines)

    # calculating how much lines each dataset will get
    trainingLines = int(TRAIN_SIZE * n)
    validationLines = int(VAL_SIZE * n)

    # adding the new training data
    with open(TRAINING_DATA_FILENAME, "r+") as file:
        oldTrainingData = file.readlines()
        oldTrainingData.extend(newDataLines[:trainingLines])
        file.writelines(oldTrainingData)

    # adding the new validation data
    with open(VALIDATION_DATA_FILENAME, "r+") as file:
        oldValData = file.readlines()
        oldValData.extend(newDataLines[trainingLines:trainingLines+validationLines])
        file.writelines(oldValData)

    # adding the new test data
    with open(TEST_DATA_FILENAME, "r+") as file:
        oldTestData = file.readlines()
        oldTestData.extend(newDataLines[trainingLines+validationLines:])
        file.writelines(oldTestData)


def splitNewData() -> None:
    # specifying the path to csv files
    path = "newData"

    # checking fo all the csv files in the path
    print(f'opening {NEW_DATA_PATH + "/*.csv"}')
    dataFiles = glob.glob(NEW_DATA_PATH + "/*.csv")

    for dataFilename in dataFiles:
        # reading content of each csv file and processing
        _splitSingleFile(training_data_path=dataFilename)

# creating pytorch's datasets
class bottleDataset(Dataset):
    def __init__(self, data_path, dataInputSize, labelsInputSize):
        self._filename = None
        self._xdf = None
        self._ydf = None
        self._dataInputSize = None
        self._labelsInputSize = None
        self._xTensors = {}
        self._yTensors = {}
        self._name = ""

        self._filename = data_path

        self._dataInputSize = dataInputSize
        self._labelsInputSize = labelsInputSize

        # opening the data and the labels
        print(f"loading from \"{data_path}\"")
        self._loadDatasetToTensors(data_path)

        print(len(self._xTensors.keys()), len(self._yTensors.keys()))

    def rindex(self, lst, value):
        return len(lst) - operator.indexOf(reversed(lst), value) - 1

    def _loadDatasetToTensors(self, data_path):
        with open(data_path, "r") as file:
            dataLines = file.readlines()

        counter = 0
        a = 0
        for data in dataLines:
            data = data.split(",")

            # removing the name and the seperator "2"
            try:
                data = data[1:self.rindex(data, "2\n")]
            except:
                data = data[1:self.rindex(data, "2")]


            # separating the data from the labels
            data, label = data[2:], data[:2]

            # creating the tensor in the rite shape
            vidTensor = torch.zeros((int(len(data) / DATA_PER_FRAME), 1, self._dataInputSize))
            for frameNum in range(int(len(data) / DATA_PER_FRAME)):
                # loading the data for each frame
                frameDataIdx = 0
                if LOAD_CAP_X:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 0])
                    frameDataIdx += 1
                if LOAD_CAP_Y:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 1])
                    frameDataIdx += 1
                if LOAD_CAP_Z:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 2])
                    frameDataIdx += 1
                if LOAD_BOTTLE_X:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 3])
                    frameDataIdx += 1
                if LOAD_BOTTLE_Y:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 4])
                    frameDataIdx += 1
                if LOAD_BOTTLE_Z:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 5])
                    frameDataIdx += 1
                if LOAD_DURATION:
                    vidTensor[frameNum][0][frameDataIdx] = float(data[frameNum * DATA_PER_FRAME + 6])

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


def loadTrain():
    trainDataset = bottleDataset(data_path=TRAINING_DATA_FILENAME, dataInputSize=5, labelsInputSize=1)
    return trainDataset

def loadTest():
    testDataset = bottleDataset(data_path=TEST_DATA_FILENAME, dataInputSize=5, labelsInputSize=1)
    return testDataset

def loadSets():
    # creating pytorch's dataloaders
    trainDataset = bottleDataset(data_path=TRAINING_DATA_FILENAME, dataInputSize=5, labelsInputSize=1)

    validationDataset = bottleDataset(data_path=VALIDATION_DATA_FILENAME, dataInputSize=5, labelsInputSize=1)

    testDataset = bottleDataset(data_path=TEST_DATA_FILENAME, dataInputSize=5, labelsInputSize=1)

    return trainDataset, validationDataset, testDataset
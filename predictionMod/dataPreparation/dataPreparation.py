import random
from .constants import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
import numpy as np
import os

def splitSingleFile(training_data_path=None, labels_data_path=None) -> None:
    if training_data_path is not None:
        trainingData = pd.read_csv(training_data_path).to_numpy()
        newXTrain, newValTestTrain = train_test_split(trainingData, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
        newValTrain, newTestTrain = train_test_split(newValTestTrain, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)

        # add save to files
        xTrainFile = pd.read_csv(TRAINING_DATA_FILENAME, header = None)
        xTrainFile = xTrainFile.append(pd.DataFrame(newXTrain, columns=xTrainFile.columns), ignore_index=True)
        xTrainFile.to_csv(TRAINING_DATA_FILENAME, index=False)

    if labels_data_path is not None:
        labelsData = pd.read_csv(training_data_path).to_numpy()
        newXLabels, newValTestLabels = train_test_split(labelsData, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
        newValTrain, newTestTrain = train_test_split(newValTestLabels, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)

        # add save to files

def splitNewData() -> None:


    # specifying the path to csv files
    path = "newData"

    # csv files in the path
    dataFiles = glob.glob(NEW_DATA_PATH + "/*.csv")
    labelsFiles = glob.glob(NEW_LABELS_PATH + "/*.csv")

    for filename in dataFiles:
        # reading content of csv file
        # content.append(filename)
        print(filename)


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

        # opening the file
        self._xdf = pd.read_csv(data_path, header=None).to_numpy()
        self._ydf = pd.read_csv(labels_path, header=None).to_numpy()
        print(f"loading from \"{data_path}\"")
        print(f"loading from \"{labels_path}\"")

        # replacing the empy cells
        # self._df[pd.isna(self._df)] = -1

        self._transformAllVidsToTensor()

    def _transformAllVidsToTensor(self):
        counter = 0
        for vidIdx in range(len(self._xdf)):
            suc = self._splitVidToTensor(vidIdx, counter)
            if suc: counter += 1

        print(f"total vids after selection: {len(self._xTensors.values())}")
        print(f"total vids after selection: {len(self._yTensors.values())}")

    def __len__(self):
        return len(self._xTensors.items())

    def _splitVidToTensor(self, vidIdx : int, dsidx) -> bool:
        """
        this function transforms how the videos are saved in the csv into
        trainable tensors
        """

        vid = self._xdf[vidIdx]
        # print(f"adding video {vid[0]} with {self._ydf[vidIdx][0]}")
        # removing the video's name
        vid = vid[1:].astype(float)

        # trimming the video until the first np.nan value
        nanIndices = list(np.argwhere(np.isnan(vid)))
        if not (nanIndices == []):
            vid = vid[:nanIndices[0][0]]

        # creating the tensor in the rite shape
        vidTensor = torch.zeros((int(len(vid) / self._dataInputSize), 1, self._dataInputSize))
        for frameNum in range(int(len(vid) / self._dataInputSize)):
            # loading the data
            for argIdx in range(self._dataInputSize):
                vidTensor[frameNum][0][argIdx] = float(vid[frameNum * self._dataInputSize + argIdx])

        labelTensor = torch.zeros((1, self._labelsInputSize))
        for i in range(self._labelsInputSize):
            labelTensor[0][i] = self._ydf[vidIdx][i+1]

        # adding the tensor to the saved tensors
        if vidTensor.shape[0] != 0:
            self._xTensors[dsidx] = vidTensor
            self._yTensors[dsidx] = labelTensor
            return True
        return False

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
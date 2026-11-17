import random
from constants import *
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def splitNewData(data):
    train, valTest = train_test_split(data, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
    val, test = train_test_split(valTest, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)

# add save to files

# creating pytorch's datasets
class bottleDataset(Dataset):
    _filename = None
    _df = None
    _infoSize = 7
    _tensDf = {}

    def __init__(self, data_path):
        self._filename = data_path

        # opening the file
        self._df = pd.read_csv(data_path).to_numpy()

        # replacing the empy cells
        self._df[pd.isna(self._df)] = -1

    def __len__(self):
        return self._df.shape[0]

    def _splitVidToTensor(self, vidIdx : int) -> None:
        """
        this function transforms how the videos are saved in the csv into
        trainable tensors
        """

        vid = self._df[vidIdx]
        # removing the video's name
        vid = vid[1:].astype(float)

        # creating the the tensor in the rite shape
        vidTensor = torch.zeros((int(len(vid) / self._infoSize), 1, self._infoSize))
        for frameNum in range(int(len(vid) / self._infoSize)):
            # loading the data
            for argIdx in range(self._infoSize):
                vidTensor[frameNum][0][argIdx] = int(vid[frameNum * self._infoSize + argIdx])

        # adding the tensor to the saved tensors
        self._tensDf[vidIdx] = vidTensor

    def getRandVid(self):
        idx = random.randint(0, len(self._df) - 1)
        return self.getVid(idx)

    def getVid(self, idx):
        if not (idx in self._tensDf.items()):
            # if we haven't converted the video to tensor, we do it now
            self._splitVidToTensor(idx)
        return self._tensDf[idx]

    def __getitem__(self, idx):
        return self.getVid(idx)


# creating pytorch's dataloaders
xTrain = bottleDataset(TRAINING_DATA_FILENAME)
yTrain = bottleDataset(TRAINING_LABELS_FILENAME)

xVal = bottleDataset(VALIDATION_DATA_FILENAME)
yVal = bottleDataset(VALIDATION_LABELS_FILENAME)

xTest = bottleDataset(TEST_DATA_FILENAME)
yTest = bottleDataset(TEST_LABELS_FILENAME)
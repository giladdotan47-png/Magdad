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
        # , header=None, nrows=1)
        self._df = pd.read_csv(data_path).to_numpy()
        print(type(self._df), type(self._df[0]), type(self._df[0]))
        self._df[pd.isna(self._df)] = -1
        print(f"df = {self._df}\nshape = {self._df.shape}, {type(self._df)}")
    def __len__(self):
        return self._df.shape[0]

    def _splitVidToTensor(self, vidIdx : int) -> None:
        vid = self._df[vidIdx]
        # removing the video's name
        vid = vid[1:].astype(float)
        #print(f"vid = {vid}")

        vidTensor = torch.zeros((int(len(vid) / self._infoSize), 1, self._infoSize))
        for frameNum in range(int(len(vid) / self._infoSize)):
            for argIdx in range(self._infoSize):
                vidTensor[frameNum][0][argIdx] = int(vid[frameNum * self._infoSize + argIdx])

        #print(f"added {vidTensor[:5][:3]}")

        # replacing the video with a tensor
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
trainDat = bottleDataset(TRAINING_FILENAME)
#valDat = bottleDataset(VALIDATION_FILENAME)
#testDat = bottleDataset(TEST_FILENAME)
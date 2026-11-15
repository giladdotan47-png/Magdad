from constants import *
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def splitNewData(data):
    train, valTest = train_test_split(data, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
    val, test = train_test_split(valTest, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)

# add save to files

# creating pytorch's datasets
class bottleDataset(Dataset):
    _filename = None
    _df = None
    _infoSize = 7
    _toTensorSet = set()
    def __init__(self, data_path):
        self._filename = data_path
        # opening the file
        self._df = pd.read_csv(data_path, header=None).to_numpy()#, nrows=1)
    def __len__(self):
        return self._df.shape[0]

    def _splitVidToTensor(self, vidIdx : int) -> None:
        vid = self._df[vidIdx]
        # removing the video's name
        vidName = vid[0]
        vid = vid[1:]

        vidTensor = torch.zeros(len(vid) / self._infoSize, 1, self._infoSize)
        for frameNum in range(len(vid) / self._infoSize):
            for argIdx in range(self._infoSize):
                vidTensor[frameNum][0][argIdx] = vid[frameNum * self._infoSize + argIdx]

        # replacing the video with a tensor
        self._df[vidIdx] = vidTensor

        # saving that we converted this video
        self._toTensorSet.add(vidIdx)

    def __getitem__(self, idx):
        if not (idx in self._toTensorSet):
            # if we haven't converted the video to tensor, we do it now
            self._splitVidToTensor(idx)
        return self._df[idx]


# creating pytorch's dataloaders
trainDat = bottleDataset(TRAINING_FILENAME)
valDat = bottleDataset(VALIDATION_FILENAME)
testDat = bottleDataset(TEST_FILENAME)
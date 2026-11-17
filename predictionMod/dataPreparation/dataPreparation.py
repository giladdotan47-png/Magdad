import random
from .constants import *
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset

def splitNewData() -> None:
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


def loadSets():
    # creating pytorch's dataloaders
    xTrain = bottleDataset(TRAINING_DATA_FILENAME)
    yTrain = bottleDataset(TRAINING_LABELS_FILENAME)

    xVal = bottleDataset(VALIDATION_DATA_FILENAME)
    yVal = bottleDataset(VALIDATION_LABELS_FILENAME)

    xTest = bottleDataset(TEST_DATA_FILENAME)
    yTest = bottleDataset(TEST_LABELS_FILENAME)

    return xTrain, yTrain, xVal, yVal
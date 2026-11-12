from constants import *
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

data = None
train, valTest = train_test_split(data, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
val, test = train_test_split(valTest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# add save to files

# creating pytorch's datasets
class bottleDataset(Dataset):
    def __init__(self, data_path):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):


# creating pytorch's dataloaders

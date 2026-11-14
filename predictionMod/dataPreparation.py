from constants import *
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader

data = None
train, valTest = train_test_split(data, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
val, test = train_test_split(valTest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# add save to files

# creating pytorch's datasets
class bottleDataset(Dataset):
    _filename = None
    _df = None
    def __init__(self, data_path):
        self._filename = data_path
        # opening the file
        self._df = pd.read_csv(data_path, header=None).to_numpy()#, nrows=1)
    def __len__(self):
        return self._df.shape[0]
    def __getitem__(self, idx):


# creating pytorch's dataloaders

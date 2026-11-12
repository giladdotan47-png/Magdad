from constants import *
from sklearn.model_selection import train_test_split

data = None
train, valTest = train_test_split(data, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_STATE)
val, test = train_test_split(valTest, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# add save to files

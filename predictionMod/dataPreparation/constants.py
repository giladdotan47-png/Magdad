import os

RANDOM_STATE = 42

# data constants
TRAIN_SIZE = 0.8
VAL_SIZE = 0.15
TEST_SIZE = 0.05

NEW_DATA_PATH = "newData"
NEW_LABELS_PATH = "newLabels"

TRAINING_DATA_FILENAME = f"Training_Data.csv"
VALIDATION_DATA_FILENAME = "Validation_Data.csv"
TEST_DATA_FILENAME = "Test_Data.csv"

# making the paths absolute values
script_dir = os.path.dirname(__file__)

NEW_DATA_PATH = os.path.join(script_dir, NEW_DATA_PATH)

TRAINING_DATA_FILENAME = os.path.join(script_dir, TRAINING_DATA_FILENAME)
VALIDATION_DATA_FILENAME = os.path.join(script_dir, VALIDATION_DATA_FILENAME)
TEST_DATA_FILENAME = os.path.join(script_dir, TEST_DATA_FILENAME)

import os

RANDOM_STATE = 42

# data constants
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

NEW_DATA_PATH = "newData"
NEW_LABELS_PATH = "newLabels"

TRAINING_DATA_FILENAME = f"Training_Data.csv"
TRAINING_LABELS_FILENAME = f"Training_Labels.csv"

VALIDATION_DATA_FILENAME = "Validation_Data.csv"
VALIDATION_LABELS_FILENAME = "Validation_Labels.csv"

TEST_DATA_FILENAME = "Test_Data.csv"
TEST_LABELS_FILENAME = "Test_Labels.csv"

# making the paths absolute values
script_dir = os.path.dirname(__file__)
TRAINING_DATA_FILENAME = os.path.join(script_dir, TRAINING_DATA_FILENAME)
TRAINING_LABELS_FILENAME = os.path.join(script_dir, TRAINING_LABELS_FILENAME)

VALIDATION_DATA_FILENAME = os.path.join(script_dir, VALIDATION_DATA_FILENAME)
VALIDATION_LABELS_FILENAME = os.path.join(script_dir, VALIDATION_LABELS_FILENAME)

TEST_DATA_FILENAME = os.path.join(script_dir, TEST_DATA_FILENAME)
TEST_LABELS_FILENAME = os.path.join(script_dir, TEST_LABELS_FILENAME)


# importing the different models:
from trainingModels import horizontalFrameModel
from trainingModels import nextFrameModel
from predictionMod.dataPreparation.dataPreparation import splitNewData, loadSets

# first, we check if we have new data parsed:
userInput = "n"
while userInput not in ["y", "n"]:
    print("do we have new video to process\n(the video must be in the \"preprocessed\" folder)")
    userInput = input("(y/n):")

if userInput == "y":
    splitNewData()
    # separate the data
    # label the data
    # add to one csv file
    # split into training / validation / testing
    pass

trainDataset, validationDataset, testDataset = loadSets()

userInput = ""
while userInput not in ["1", "2"]:
    print("which model do you want to train?")
    print("1. predict the next frame")
    print("2. predict the final height and time")
    userInput = input(":")

if userInput == "1":
    nextFrameModel.trainModel(trainDataset, validationDataset)
if userInput == "2":
    horizontalFrameModel.trainModel(trainDataset, validationDataset)
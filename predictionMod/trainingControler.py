# importing the different models:
from trainingModels.trainModel_nextFrame import trainModel_nextFrame
from trainingModels.trainModel_nextFrame import trainModel_nextFrame
from trainingModels.trainModel_horizontalFrame import trainModel_horizontalFrame
from dataPreparation import xTrain, yTrain, xVal, yVal, xTest, yTest

# first, we check if we have new data parsed:
userInput = ""
while userInput not in ["y", "n"]:
    print("do we have new video to process\n(the video must be in the \"preprocessed\" folder)")
    userInput = input("(y/n):")

if userInput == "y":
    # separate the data
    # label the data
    # add to one csv file
    # split into training / validation / testing
    pass

userInput = ""
while userInput not in ["1", "2"]:
    print("which model do you want to train?")
    print("1. predict the next frame")
    print("2. predict the final height and time")
    userInput = input(":")

if userInput == "1":
    trainModel_nextFrame.trainModel(xTrain, xVal)
if userInput == "1":
    trainModel_horizontalFrame.trainModel(xTrain, yTrain, xVal, yVal)
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

trainDataset, validationDataset, testDataset = loadSets()

userInput = "2"
while userInput not in ["1", "2"]:
    print("which model do you want to train?")
    print("1. predict the next frame")
    print("2. predict the final height and time")
    userInput = input(":")

if userInput == "1":
    nextFrameModel.trainModel(trainDataset, validationDataset)
if userInput == "2":
    horizontalFrameModel.trainModel(trainDataset, validationDataset)

[229863.12496450907, 219515.55062982359, 203068.10862727827, 182254.61867794438, 189270.31118389545, 142111.4035077153, 159905.39872219643, 118891.00152130944, 123900.27229240895, 122180.05541821024, 117220.87357919572, 97988.98233598108, 86899.38816427108, 108295.37350973621, 101767.27266878526, 87159.97528535228, 90923.52205417425, 89956.4487712709]
[17055946.976987027, 16342207.691946194, 15563100.368687535, 14573749.59461671, 13410986.284218695, 12315979.077394571, 11088950.525529737, 10252297.486284083, 9655541.682447739, 9173480.270367835, 8728544.50182606, 8379471.389665139, 8066250.359470242, 7733262.658420539, 7365976.506029359, 6998294.952897634, 6774429.237421893, 6536708.063667612]
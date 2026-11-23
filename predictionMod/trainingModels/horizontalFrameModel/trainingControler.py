import torch
from .trainModel_horizontalFrame import trainModel, evalAtVal

def trainModelCont(trainDataset, validationDataset, testDataset):
    chosenModel = "1"

    availableModels = ["1"]

    while chosenModel not in availableModels:
        print("available models are:")
        for mod in availableModels:
            print(f"-> {mod}")
        chosenModel = input("choose model from the above: ")

    if chosenModel == "1":
        from .model1.constants import N_ITERS, BATCH_SIZE, PRINT_STEPS
        from .model1.rnnModel import model, optimizer, criterion

    userInput = "n"
    while userInput not in ["y", "n"]:
        userInput = input("do you want to load the pretrained weights? (y/n) ")

    if userInput == "y":
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, "model1")
        dir_path = os.path.join(dir_path, "model_weights_final.pt")
        model.load_state_dict(torch.load(dir_path, weights_only=True))
        model.eval()

    userInput = ""
    while userInput not in ["1", "2", "3"]:
        print("choose from the following:")
        print("1. train the mocel (or continue training)")
        print("2. evaluate at validation set")
        print("3. evaluate at test set")

        userInput = input(": ")

    if userInput == "1":
        trainModel(trainDataset, validationDataset, model, optimizer, criterion, N_ITERS, BATCH_SIZE, PRINT_STEPS)

    if userInput == "2":
        print(evalAtVal(validationDataset, model, optimizer, criterion))

    if userInput == "3":
        print(evalAtVal(testDataset, model, optimizer, criterion))


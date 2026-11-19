from .rnnModel import model, optimizer, criterion
import torch
from .constants import N_ITERS, BATCH_SIZE, PRINT_STEPS
import matplotlib.pyplot as plt
import random


def singleStep(xTrain, yTrain):
    # initializing the model for the step
    hidden = model.init_hidden()

    # stacking the
    preStacked = []
    for i in range(xTrain.size(0)):
        preStacked.append(yTrain[0]+i)

    tOutputs = torch.stack(tuple(preStacked))
    pOutputs = []

    for i in range(xTrain.size(0)):
        output, hidden = model(xTrain[i], hidden)
        pOutputs.append(output[0]+i)

    pOutputs = torch.stack(pOutputs)
    loss = criterion(pOutputs[40:], tOutputs[40:])

    loss.backward()

    return loss.item() / (xTrain.size(0) - 40)


def trainModel(trainDataset, validationDataset):
    all_losses = []
    epoch_loss = 0


    for iter in range(N_ITERS):
        # getting the training data and label
        vid, label = trainDataset.getRand()

        # adding the video and label to the device
        vidCuda = vid.cuda() if torch.cuda.is_available() else vid
        labelCuda = label.cuda() if torch.cuda.is_available() else label

        if (iter + 1) % BATCH_SIZE == 0:
            optimizer.zero_grad()

        loss = singleStep(vidCuda, labelCuda)

        if (iter + 1) % BATCH_SIZE == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss


        if (iter+1) % PRINT_STEPS == 0:
            # saving the average of the losses
            avg_loss = epoch_loss / PRINT_STEPS
            all_losses.append(avg_loss)
            print(f"Epoch {(iter + 1) // PRINT_STEPS}/{N_ITERS//PRINT_STEPS}, Avg Loss: {avg_loss:.6f}")
            epoch_loss = 0

    print(all_losses)
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(all_losses)
    plt.show()
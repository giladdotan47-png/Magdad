from rnnModel import model, optimizer, criterion
import torch
from constants import N_ITERS
import matplotlib.pyplot as plt
import random


def singleStep(xTrain, yTrain):
    # initializing the model for the step
    hidden = model.init_hidden()
    optimizer.zero_grad()

    # stacking the
    preStacked = []
    for i in range(xTrain.size(0)):
        preStacked.append(yTrain)

    tOutputs = torch.stack(tuple(preStacked))
    pOutputs = []

    for i in range(xTrain.size(0)):
        output, hidden = model(xTrain[i], hidden)
        pOutputs.append(output)

    pOutputs = torch.stack(pOutputs)
    loss = criterion(pOutputs, tOutputs)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def trainModel(xTrain, yTrain, xVal, yVal):
    all_losses = []
    epoch_loss = 0


    for epoch in range(N_ITERS):
        untrainedVids = list(range(len(xTrain)))
        while len(untrainedVids) > 0:
            # choosing random video
            idx = random.randint(0, len(untrainedVids) - 1)

            # getting the training data and label
            vid = xTrain[idx]
            label = yTrain[idx]

            # removing the index from the unused videos
            del untrainedVids[idx]

            # adding the video and label to the device
            vidCuda = vid.cuda() if torch.cuda.is_available() else vid
            labelCuda = label.cuda() if torch.cuda.is_available() else label

            loss = singleStep(vidCuda, labelCuda)
            epoch_loss += loss

        # saving the average of the losses
        avg_loss = epoch_loss / len(xTrain)
        all_losses.append(avg_loss)
        print(f"Epoch {(epoch + 1) // len(xTrain)}, Avg Loss: {avg_loss:.6f}")
        epoch_loss = 0

    print(all_losses)
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(all_losses)
    plt.show()
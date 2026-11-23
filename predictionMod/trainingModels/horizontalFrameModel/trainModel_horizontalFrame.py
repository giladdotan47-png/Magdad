from .rnnModel import model, optimizer, criterion
import torch
from .constants import N_ITERS, BATCH_SIZE, PRINT_STEPS
import matplotlib.pyplot as plt

def singleStep(xTrain, yTrain):
    # initializing the model for the step
    hidden = model.init_hidden()

    # stacking the
    preStacked = []
    for i in range(xTrain.size(0)):
        preStacked.append(yTrain*i)

    tOutputs = torch.stack(tuple(preStacked))
    pOutputs = []

    for i in range(xTrain.size(0)):
        output, hidden = model(xTrain[i], hidden)
        pOutputs.append(output*i)

    pOutputs = torch.stack(pOutputs)
    pOutputs = pOutputs[40:]
    tOutputs = tOutputs[40:]
    #print(pOutputs.size(), tOutputs.size())
    loss = criterion(pOutputs, tOutputs)

    loss.backward()

    return loss.item() / (xTrain.size(0) - 40)

def evalAtVal(validationDataset):
    lss = 0
    for i in range(len(validationDataset)):
        hidden = model.init_hidden()
        x, y = validationDataset[i]

        x = x.cuda() if torch.cuda.is_available() else x
        y = y.cuda() if torch.cuda.is_available() else y

        # stacking the
        preStacked = []
        for i in range(x.size(0)):
            preStacked.append(y * i)

        tOutputs = torch.stack(tuple(preStacked))
        pOutputs = []

        for i in range(x.size(0)):
            output, hidden = model(x[i], hidden)
            pOutputs.append(output * i)

        pOutputs = torch.stack(pOutputs)
        #print(pOutputs.size(), tOutputs.size())
        lss += criterion(pOutputs[40:], tOutputs[40:]).item() / (x.size(0) - 40)

    return lss

def trainModel(trainDataset, validationDataset):
    all_losses = []
    all_val_losses = []
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
            print(f"iter {(iter + 1) // PRINT_STEPS}/{N_ITERS//PRINT_STEPS}, Avg Loss: {avg_loss:.6f}")
            epoch_loss = 0
            valLos = evalAtVal(validationDataset)
            all_val_losses.append(valLos)
            print(f"validation loss = {valLos}")
            torch.save(model.state_dict(), f"model_weights_{(iter + 1) // PRINT_STEPS}.pt")

    print(all_losses)
    print(all_val_losses)
    x = list(range(len(all_losses)))
    plt.plot(x, all_losses, color='r', label="trainingLosses")
    plt.plot(x, all_val_losses, color='b', label="validationLosses")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model_weights.pt")
    #plt.figure()
    #plt.xlabel("epochs")
    #plt.ylabel("loss")
    #plt.plot(all_losses)
    #plt.show()
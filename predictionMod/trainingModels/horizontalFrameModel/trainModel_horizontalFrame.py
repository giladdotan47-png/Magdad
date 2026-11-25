import torch
import matplotlib.pyplot as plt
import numpy as np

def singleStep(xTrain, yTrain, lasts, model, criterion):
    # initializing the model for the step
    hidden = model.init_hidden()

    # stacking the
    preStacked = []
    for i in range(xTrain.size(0)):
        preStacked.append(yTrain * i)

    tOutputs = torch.stack(tuple(preStacked))
    pOutputs = []

    for i in range(xTrain.size(0)):
        output, hidden = model(xTrain[i], hidden)
        pOutputs.append(output * i)

    pOutputs = torch.stack(pOutputs)
    pOutputs = pOutputs[70:]
    tOutputs = tOutputs[70:]

    loss = criterion(pOutputs, tOutputs)
    std = 0

    eps = 1e-6
    target_std = 10
    lambda_var = 20

    std = torch.sqrt(pOutputs.var(dim=0, unbiased=False) + eps)

    variance_penalty = torch.relu(target_std - std).mean()

    loss = loss + (lambda_var * variance_penalty)
    if std == 0.0010:
        loss = loss + 1000
    if len(lasts) % 100 == 1:
        print(f"std = {std}")

    if len(lasts) >= 2 and False:
        lasts = torch.tensor(lasts)
        std = torch.sqrt(lasts.var(dim=0, unbiased=False) + eps)

        variance_penalty = torch.relu(target_std - std).mean()

        loss = loss + (lambda_var * variance_penalty)

        std = std.item()

    if len(lasts) % 100 == 1 and False:
        print(f"{len(lasts)}, added loss = {loss}")

    loss.backward()

    return loss.item(), pOutputs[-1]  # / (xTrain.size(0) - 70), pOutputs[-1], std

def evalAtVal(validationDataset, model, criterion):
    lss = 0
    for i in range(len(validationDataset)):
        hidden = model.init_hidden()
        x, y = validationDataset[i]

        x = x.cuda() if torch.cuda.is_available() else x
        y = y.cuda() if torch.cuda.is_available() else y

        # stacking the
        preStacked = []
        for i in range(x.size(0)):
            preStacked.append(y)

        tOutputs = torch.stack(tuple(preStacked))
        pOutputs = []

        for i in range(x.size(0)):
            output, hidden = model(x[i], hidden)
            pOutputs.append(output)

        pOutputs = torch.stack(pOutputs)
        # print(pOutputs.size(), tOutputs.size())
        lss += criterion(pOutputs[70:], tOutputs[70:]).item() / (x.size(0) - 70)

    return lss

def trainModel(trainDataset, validationDataset, model, optimizer, criterion, N_ITERS, BATCH_SIZE, PRINT_STEPS):
    all_losses = []
    all_val_losses = []
    epoch_loss = 0
    lasts = []
    varss = []

    for iter in range(N_ITERS):
        # getting the training data and label
        vid, label = trainDataset.getRand()

        # adding the video and label to the device
        vidCuda = vid.cuda() if torch.cuda.is_available() else vid
        labelCuda = label.cuda() if torch.cuda.is_available() else label

        if (iter + 1) % BATCH_SIZE == 0:
            optimizer.zero_grad()

        loss, lst = singleStep(xTrain=vidCuda, yTrain=labelCuda, lasts=lasts, model=model, criterion=criterion)
        lst = float(lst.detach().numpy()[0][0].item())
        # loss, lst = singleStep(vidCuda, labelCuda)
        lasts.append(lst)

        if (iter + 1) % BATCH_SIZE == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss += loss

        if (iter + 1) % 100 == 0:
            varss.append(np.var(lasts))
            lasts = []

        if (iter + 1) % PRINT_STEPS == 0:
            # saving the average of the losses
            avg_loss = epoch_loss / PRINT_STEPS
            all_losses.append(avg_loss)
            print(f"iter {(iter + 1) // PRINT_STEPS}/{N_ITERS // PRINT_STEPS}, Avg Loss: {avg_loss:.6f}")
            epoch_loss = 0
            valLos = evalAtVal(validationDataset, model=model, criterion=criterion)
            all_val_losses.append(valLos)
            print(f"validation loss = {valLos}")

            torch.save(model.state_dict(), f"model_weights_{(iter + 1) // PRINT_STEPS}.pt")

            print(all_losses)
            print(all_val_losses)
            print(varss)
    x = list(range(len(all_losses)))
    plt.plot(x, all_losses, color='r', label="trainingLosses")
    plt.plot(x, all_val_losses, color='b', label="validationLosses")
    plt.legend()
    plt.show()

    # Save final model to Google Drive
    torch.save(model.state_dict(), "model_weights_final")
    print(f"Saved final model to: model_weights_final")
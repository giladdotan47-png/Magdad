from predictionMod.trainingModels.trainModel_horizontalFrame.rnnModel import model, optimizer, criterion
import torch
from constants import N_ITERS
import matplotlib.pyplot as plt
from predictionMod.dataPreparation import trainDat

def singleStep(line_tensor):
    # initializing the analysis
    hidden = model.init_hidden()
    optimizer.zero_grad()

    # separating to true next frames and hypothesis next frames
    tOutputs = line_tensor[1:]
    pOutputs = []

    # run on each frame
    for i in range(line_tensor.size(0) - 1):
        output, hidden = model(line_tensor[i], hidden)
        pOutputs.append(output)

    pOutputs = torch.stack(pOutputs)  # stacking the hypothesises
    loss = criterion(pOutputs, tOutputs)  # comparing to the real frames

    # updating the weights
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def trainModel(vidsTrain, vidsValidation):
    all_losses = []
    epoch_loss = 0

    for i in range(N_ITERS):
        vid = vidsTrain[i % len(trainDat)]
        # adding the video to the device
        seq = vid.cuda() if torch.cuda.is_available() else vid

        loss = singleStep(seq)
        epoch_loss += loss

        # saving the average of the losses
        if (i + 1) % len(trainDat) == 0:
            avg_loss = epoch_loss / len(trainDat)
            all_losses.append(avg_loss)
            print(f"Epoch {(i + 1) // len(trainDat)}, Avg Loss: {avg_loss:.6f}")
            epoch_loss = 0


    print(all_losses)
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(all_losses)
    plt.show()
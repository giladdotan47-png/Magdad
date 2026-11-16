from constants import *
from dataPreparation import *
from rnnModel import *
import matplotlib.pyplot as plt


def test_train(line_tensor):
    # 1. Save a copy of a weight tensor before stepping
    before = next(model.parameters()).clone()

    optimizer.zero_grad()
    hidden = model.init_hidden()

    targets = line_tensor[1]
    output, hidden = model(line_tensor[0], hidden)
    output = torch.nan_to_num(output, nan=-1.0)

    loss = criterion(output, targets)
    loss.backward()

    # 2. Check if gradients exist and are non-zero
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for {name} (Graph broken or frozen)")
        elif torch.all(param.grad == 0):
            print(f"Zero gradient for {name} (Vanishing grad or dead ReLU)")

    optimizer.step()

    # 3. Check if the weights actually changed
    after = next(model.parameters())
    print("Weights updated:", not torch.equal(before, after))

    return loss.item()

def true_train(line_tensor):
    hidden = model.init_hidden()
    optimizer.zero_grad()
    totalLoss = 0
    tOutputs = line_tensor[:-1]
    pOutputs = torch.zeros((line_tensor.shape[0]-1, line_tensor.shape[1], line_tensor.shape[2]))
    for i in range(line_tensor.size()[0] - 1):
        output, hidden = model(line_tensor[i], hidden)
        pOutputs[i] = output

    pOutputs = torch.nan_to_num(pOutputs, nan=-1.0)

    loss = criterion(pOutputs, tOutputs)
    totalLoss += loss.item()

    loss.backward()
    optimizer.step()

    return totalLoss

current_loss = 0
all_losses = []
for i in range(N_ITERS):
    vid = trainDat.getRandVid()
    vid = trainDat[i % len(trainDat)]
    loss = train(vid)

    all_losses.append(loss)

    if (i + 1) % PRINT_STEPS == 0:
        print(f"iter: {i}, loss = {loss}")

print(all_losses)
plt.figure()
plt.plot(all_losses)
plt.show()


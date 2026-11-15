from constants import *
from dataPreparation import *
from rnnModel import *

# running for the first time to the parameters vectors will not be None
hidden_tensor = model.init_hidden()
output, next_hidden = model(trainDat[0][0], hidden_tensor)

def train(line_tensor, y_true):
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()



current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    #print(line_tensor.shape, category_tensor.shape)
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i + 1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()
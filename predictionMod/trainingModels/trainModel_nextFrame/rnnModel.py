from constants import *
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = torch.tanh(self.i2h(combined))  # Prevents exploding values
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        device = next(self.parameters()).device
        return torch.zeros(1, self.hidden_size, device=device)


# generating the model
model = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# checking whether we have a GPU available
if torch.cuda.is_available():
    model.cuda()

# deciding on the loss
criterion = nn.MSELoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
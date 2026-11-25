from predictionMod.trainingModels.horizontalFrameModel.model1.constants import *
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2hSig = nn.Sigmoid()
        self.i2h1 = nn.Linear(hidden_size, hidden_size)
        self.i2hSig1 = nn.Sigmoid()
        self.i2h2 = nn.Linear(hidden_size, hidden_size)
        #self.i2hSig2 = nn.Sigmoid()
        #self.i2h3 = nn.Linear(hidden_size, hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2oSig = nn.Sigmoid()
        self.i2o1 = nn.Linear(hidden_size, hidden_size)
        self.i2oSig1 = nn.Sigmoid()
        self.i2o2 = nn.Linear(hidden_size, output_size)
        #self.i2oSig2 = nn.Sigmoid()
        #self.i2o3 = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        #hidden = torch.tanh(self.i2h(combined))  # Prevents exploding values
        hidden = self.i2h(combined)
        hidden = self.i2hSig(hidden)
        hidden = self.i2h1(hidden)
        hidden = self.i2hSig1(hidden)
        hidden = self.i2h2(hidden)
        #hidden = self.i2hSig2(hidden)
        #hidden = self.i2h3(hidden)

        output = self.i2o(combined)
        output = self.i2oSig(output)
        output = self.i2o1(output)
        output = self.i2oSig1(output)
        output = self.i2o2(output)
        #output = self.i2oSig2(output)
        #output = self.i2o3(output)

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

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.01)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#for param in model.parameters():
#    param.requires_grad =True
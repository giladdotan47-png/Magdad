from constants import *
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)

        output = self.i2o(combined)
        return output, hidden


    def init_hidden(self):
        return torch.zeros((1, self.hidden_size))
    ################################################################################################################################################################################################################################################################################################################################
#    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#        super(RNNModel, self).__init__()
#        # Hidden dimensions
#        self.hidden_dim = hidden_dim
#
#        # Number of hidden layers
#        self.layer_dim = layer_dim
#
#        # Building your RNN
#        # batch_first=True causes input/output tensors to be of shape
#        # (batch_dim, seq_dim, feature_dim)
#        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
#
#        # Readout layer
#        self.fc = nn.Linear(hidden_dim, output_dim)
#
#    def forward(self, x):
#        # Initialize hidden state with zeros
#        #######################
#        #  USE GPU FOR MODEL  #
#        #######################
#        if torch.cuda.is_available(): h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
#        else: h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
#
#        # One time step
#        out, hn = self.rnn(x, h0)
#
#        # Index hidden state of last time step
#        # out.size() --> 100, 28, 100
#        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
#        out = self.fc(out[:, -1, :])
#        # out.size() --> 100, 10
#        return out

# generating the model
model = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

#if torch.cuda.is_available():
#    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# checking whether we have a GPU available
#device = torch.device('cpu')
#if torch.cuda.is_available():
#    device = torch.device('cuda')

#torch.set_default_device(device)

# deciding on the loss
criterion = nn.MSELoss()
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
LSTM units follow the formulae:

i = sigmoid(x.mm(W_i) + old_h.mm(U_i) + b_i)
f = sigmoid(x.mm(W_f) + old_h.mm(U_f) + b_f)
c = f*old_c + i*tanh(x.mm(W_c) + old_h.mm(U_c) + b_c)
o = sigmoid(x.mm(W_o) + old_h.mm(U_o) + b_o)
h = o*tanh(c)

Where x is the new, incoming data old_h is the h at the previous time step, and 
old_c is the c at the previous time step.
Each of the W_ and U_ terms are weight matrices and the b_ terms are biases specific
to the quantity being calculated.

"""

class LSTM(nn.Module):
    def __init__(self, state_size=256):
        super(LSTM, self).__init__()

        self.state_size = state_size

        # Internal LSTM Entry Parameters
        means = torch.zeros(4, state_size, state_size)
        self.W = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM State Parameters
        means = torch.zeros(4, state_size, state_size)
        self.U = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM Bias Parameters
        self.b = nn.Parameter(torch.zeros(4,1,state_size), requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, old_h, old_c):
        """
        x - New data coming into LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_h - short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)

        returns:
            h - new short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c - new long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        """

        i = self.sigmoid(x.mm(self.W[0]) + old_h.mm(self.U[0]) + self.b[0])
        f = self.sigmoid(x.mm(self.W[1]) + old_h.mm(self.U[1]) + self.b[1])
        c = f*old_c + i*self.tanh(x.mm(self.W[2]) + old_h.mm(self.U[2]) + self.b[2])
        o = self.sigmoid(x.mm(self.W[3]) + old_h.mm(self.U[3]) + self.b[3])
        h = o*self.tanh(c)

        return h, c

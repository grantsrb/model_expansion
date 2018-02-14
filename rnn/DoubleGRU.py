import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
GRU units follow the formulae:

z = sigmoid(W_z.mm(x) + U_z.mm(old_h) + b_z)
r = sigmoid(W_r.mm(x) + U_r.mm(old_h) + b_r
h = z*old_h + (1-z)*tanh(W_h.mm(x) + U_h.mm(r*old_h) + b_h)

Where x is the new, incoming data and old_h is the h at the previous time step.
Each of the W_ and U_ terms are weight matrices and the b_ terms are biases.
In this implementation, all of the W_ terms are combined into a single variable. Same
with the U_ and b_ terms.
Also note that we are stacking 2 GRU units, so we create 2 W_, 2 U_, and 2 b_ parameters.

"""

class DoubleGRU(nn.Module):
    def __init__(self, state_size=256):
        super(DoubleGRU, self).__init__()

        self.state_size = state_size

        # Internal GRU Entry Parameters
        means = torch.zeros(6, state_size, state_size)
        self.W = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal GRU State Parameters
        means = torch.zeros(6, state_size, state_size)
        self.U = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal GRU Bias Parameters
        self.b = nn.Parameter(torch.zeros(6,1,state_size), requires_grad=True)

        # Internal Dropout and BatchNorm
        self.mid_dropout = nn.Dropout(p=.5)
        self.mid_h_bnorm = nn.BatchNorm1d(state_size)

        # Stacking Middle Parameters
        means = torch.zeros(state_size, state_size)
        self.mid = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)
        self.mid_bnorm = nn.BatchNorm1d(state_size)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, old_h):
        """
        old_h - running state of GRU. FloatTensor Variable with shape (batch_size, state_size)
        x - New data coming into GRU. FloatTensor Variable with shape (batch_size, state_size)
        """

        z = self.sigmoid(x.mm(self.W[0]) + old_h.mm(self.U[0]) + self.b[0])
        r = self.sigmoid(x.mm(self.W[1]) + old_h.mm(self.U[1]) + self.b[1])
        mid_h = z*old_h + (1-z)*self.tanh(x.mm(self.W[2]) + (r*old_h).mm(self.U[2]) + self.b[2])

        mid_h = self.mid_dropout(mid_h)
        mid_h = self.mid_h_bnorm(mid_h)
        mid_x = self.relu(mid_h.mm(self.mid))
        mid_x = self.mid_bnorm(mid_x)

        z = self.sigmoid(mid_x.mm(self.W[3]) + mid_h.mm(self.U[0]) + self.b[0])
        r = self.sigmoid(mid_x.mm(self.W[4]) + mid_h.mm(self.U[4]) + self.b[4])
        h = z*mid_h + (1-z)*self.tanh(mid_x.mm(self.W[5]) + (r*mid_h).mm(self.U[5]) + self.b[5])

        return h

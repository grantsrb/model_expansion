import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rnn.LSTM

"""
Nested LSTM units are very similar to LSTMs but use a nested LSTM for the c term.
Nested LSTMs follow the formulae:

i = sigmoid(x.mm(W_i) + old_h.mm(U_i) + b_i)
f = sigmoid(x.mm(W_f) + old_h.mm(U_f) + b_f)

old_h_hat = f*old_c
x_hat = i*tanh(x.mm(W_c) + old_h.mm(U_c) + b_c)

i_hat = sigmoid(x_hat.mm(W_i_hat) + old_h_hat.mm(U_i_hat) + b_i_hat)
f_hat = sigmoid(x_hat.mm(W_f_hat) + old_h_hat.mm(U_f_hat) + b_f_hat)
c_hat = f_hat*old_c_hat + i_hat*tanh(x_hat.mm(W_c_hat) + old_h_hat.mm(U_c_hat) + b_c_hat)
o_hat = sigmoid(x_hat.mm(W_o_hat) + old_h_hat.mm(U_o_hat) + b_o_hat)
h_hat = o_hat*tanh(c_hat)
c = h_hat
o = sigmoid(x.mm(W_o) + old_h.mm(U_o) + b_o)
h = o*tanh(c)

Where x is the new, incoming data old_h is the h at the previous time step, and 
old_c is the c at the previous time step, and old_c_hat is the c_hat at the previous time step.
Each of the W_ and U_ terms are weight matrices and the b_ terms are biases specific
to the quantity being calculated.

"""

class NestedLSTM(nn.Module):
    def __init__(self, state_size=256, nested_state_size=256):
        super(NestedLSTM, self).__init__()

        self.state_size = state_size
        self.nested_state_size = nested_state_size

        # Internal LSTM Entry Parameters
        means = torch.zeros(4, state_size, state_size)
        self.W = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM State Parameters
        means = torch.zeros(4, state_size, state_size)
        self.U = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM Bias Parameters
        self.b = nn.Parameter(torch.zeros(4,1,state_size), requires_grad=True)

        # Nested LSTM
        self.LSTM = LSTM.LSTM(nested_state_size)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, old_h, old_c, old_c_hat):
        """
        x - New data coming into LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_h - short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of nested LSTM. FloatTensor Variable with shape (batch_size, state_size)

        returns:
            h - new short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c - new long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c_hat - new long term memory of nested LSTM. FloatTensor Variable of shape (batch_size, state_size)
        """

        i = self.sigmoid(x.mm(self.W[0]) + old_h.mm(self.U[0]) + self.b[0])
        f = self.sigmoid(x.mm(self.W[1]) + old_h.mm(self.U[1]) + self.b[1])
        old_h_hat = f*old_c
        x_hat = i*self.tanh(x.mm(self.W[2]) + old_h.mm(self.U[2]) + self.b[2])
        h_hat, c_hat = self.LSTM.forward(x_hat, old_h_hat, old_c_hat)
        c = h_hat
        o = self.sigmoid(x.mm(self.W[3]) + old_h.mm(self.U[3]) + self.b[3])
        h = o*self.tanh(c)

        return h, c, c_hat

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import DoubleGRU


class RecurrentUnit(nn.Module):
    def __init__(self, n_words, emb_size, state_size=256):
        super(RecurrentUnit, self).__init__()

        self.n_words = n_words
        self.emb_size = emb_size
        self.state_size = state_size

        # Embedding Sets
        means = torch.zeros(n_words,emb_size)
        self.embeddings = nn.Parameter(torch.normal(means, std=0.01), requires_grad=True)

        # Entry Parameters
        self.entry_bnorm = nn.BatchNorm1d(emb_size)
        means = torch.zeros(emb_size, state_size)
        self.entry = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)
        self.preGRU_bnorm = nn.BatchNorm1d(state_size)

        self.GRU = DoubleGRU.DoubleGRU(state_size)

        # Exit Parameters
        self.exit_bnorm = nn.BatchNorm1d(state_size)
        means = torch.zeros(state_size, state_size)
        self.exit = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Classifier
        self.classifier_bnorm = nn.BatchNorm1d(state_size)
        means = torch.zeros(state_size, n_words)
        self.classifier = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, old_h, emb_idxs):
        """
        old_h - FloatTensor Variable with dimensions (batch_size, state_size)
        emb_idxs - LongTensor Variable with dimensions (batch_size,)
        """
        embeddings = self.embeddings[emb_idxs]

        embeddings = self.entry_bnorm(embeddings)
        x = self.relu(embeddings.mm(self.entry))
        x = self.preGRU_bnorm(x)

        h = self.GRU(old_h, x)

        exit_x = self.exit_bnorm(h)
        exit_x = self.relu(exit_x.mm(self.exit))

        output = self.classifier_bnorm(exit_x)
        output = exit_x.mm(self.classifier)

        return h, output

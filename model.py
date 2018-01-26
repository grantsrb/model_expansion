import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import RecurrentUnit
import gc
import resource
import sys

class Model(nn.Module):
    def __init__(self, word_set1, emb_size1, word_set2, emb_size2, core_size=256, expanded_size=256, lr=0.001):
        super(Model, self).__init__()

        self.stateful = True # used to determine if h is persistent through entire epoch

        self.word_set1 = word_set1
        self.emb_size1 = emb_size1
        self.word_set2 = word_set2
        self.emb_size2 = emb_size2
        self.core_size = core_size
        self.expanded_size = expanded_size

        self.core = RecurrentUnit.RecurrentUnit(len(word_set1), emb_size1, core_size)
        self.expanded = RecurrentUnit.RecurrentUnit(len(word_set2), emb_size2, expanded_size)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def step(self, h, seq, labels, RUnit):
        h,output = RUnit.forward(h,Variable(seq))
        loss = self.cross_entropy(output, Variable(labels))
        loss.backward(retain_graph=True)
        return h, loss

    def optimize(self, X, Y, RUnit):
        """
        X - LongTensor of shape (-1, GRU sequence length, batch size)
        Y - LongTensor of shape (-1, GRU sequence length, batch_size)
        RUnit - the RecurrentUnit to be trained (core or expanded)
        """

        next_h = torch.zeros(X.size(-1), RUnit.state_size)
        if torch.cuda.is_available():
            next_h = next_h.cuda()

        loss = None 
        for i in range(X.size(0)):
            x,y = X[i], Y[i]
            avg_loss = 0
            del loss # Free graph

            if not self.stateful:
                h = Variable(torch.zeros(len(x[0]), RUnit.state_size))
                if torch.cuda.is_available():
                    h = h.cuda()
            else:
                h = Variable(next_h) # Free graph

            for j in range(len(x)):
                seq, labels = x[j], y[j]
                self.optim.zero_grad()
                h, loss = self.step(h, seq, labels, RUnit)
                if j == 0:
                    next_h = h.data.clone() # Forces GRU to do more than memorize the sequence
                avg_loss += loss.data[0]
                print(j,"/",len(x), "– Loss:", loss.data[0], end="\r")
                self.optim.step()
            print("Step", i,"/", X.size(0), "– Avg Loss:", avg_loss/len(x))

            # Check for memory leaks
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print("Memory Used: {:.2f} units".format(max_mem_used / 1024))


    def sync_expanded(self):
        """
        Updates the expanded Recurrent Unit's embeddings and first few operations to match
        the core's parameters' data.
        """
        self.expanded.embeddings.data[:self.core.n_words, :self.core.emb_size] = self.core.embeddings.data
        for core_p, exp_p in zip(self.core.entry_bnorm.parameters(), self.expanded.entry_bnorm.parameters()):
            exp_p.data[:core_p.data.size(0)] = core_p.data
        self.expanded.entry.data[:self.core.entry.size(0), :self.core.entry.size(1)] = self.core.entry.data
        for core_p, exp_p in zip(self.core.preGRU_bnorm.parameters(), self.expanded.preGRU_bnorm.parameters()):
            exp_p.data[:core_p.data.size(0)] = core_p.data

    def sync_core(self, average=False):
        """
        Updates the core Recurrent Unit's embeddings and first few operations to match
        the expanded's parameters' data.

        average - boolean denoting if core values should be averaged with their previous values
                    during the sync.
        """
        if average:
            self.core.embeddings.data = (self.core.embeddings.data + self.expanded.embeddings.data[:self.core.n_words,:self.core.emb_size])/2
            for core_p, exp_p in zip(self.core.entry_bnorm.parameters(), self.expanded.entry_bnorm.parameters()):
                core_p.data = (core_p.data + exp_p.data[:core_p.data.size(0)])/2
            self.core.entry.data = (self.core.entry.data + self.expanded.entry.data[:self.core.entry.size(0), :self.core.entry.size(1)])/2
            for core_p, exp_p in zip(self.core.preGRU_bnorm.parameters(), self.expanded.preGRU_bnorm.parameters()):
                core_p.data = (core_p.data + exp_p.data[:core_p.data.size(0)])/2
        else:
            self.core.embeddings.data = self.expanded.embeddings.data[:self.core.n_words, :self.core.emb_size]
            for core_p, exp_p in zip(self.core.entry_bnorm.parameters(), self.expanded.entry_bnorm.parameters()):
                core_p.data = exp_p.data[:core_p.data.size(0)]
            self.core.entry.data = self.expanded.entry.data[:self.core.entry.size(0), :self.core.entry.size(1)]
            for core_p, exp_p in zip(self.core.preGRU_bnorm.parameters(), self.expanded.preGRU_bnorm.parameters()):
                core_p.data = exp_p.data[:core_p.data.size(0)]

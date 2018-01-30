import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import DoubleGRU


class RecurrentUnit(nn.Module):
    def __init__(self, n_words, emb_size, state_size=256, seq_len=30):
        super(RecurrentUnit, self).__init__()

        self.n_words = n_words
        self.emb_size = emb_size
        self.state_size = state_size
        self.seq_len = seq_len

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
        means = torch.zeros(state_size, emb_size)
        self.exit = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Classifier
        self.classifier_bnorm = nn.BatchNorm1d(emb_size)
        means = torch.zeros(emb_size, n_words)
        self.classifier = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(emb_size))), requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.mse = nn.MSELoss()
        self.softmax = nn.Softmax(-1)

    def forward(self, old_h, emb_idxs):
        """
        old_h - FloatTensor Variable with dimensions (batch_size, state_size)
        emb_idxs - LongTensor Variable with dimensions (batch_size,)
        """

        embeddings = self.embeddings[emb_idxs]

        h, exit_x = self.subforward(old_h, embeddings)

        output = self.classifier_bnorm(exit_x)
        output = exit_x.mm(self.classifier)

        return h, output

    def subforward(self, old_h, embeddings):
        """
        Helper function for forward
        """

        embeddings = self.entry_bnorm(embeddings)
        x = self.relu(embeddings.mm(self.entry))
        x = self.preGRU_bnorm(x)

        h = self.GRU(old_h, x)

        exit_x = self.exit_bnorm(h)
        exit_x = self.relu(exit_x.mm(self.exit))

        return h, exit_x

    def generate_text(self, seed, generation_len):
        """
        Generates text conditioned on the seed. Returns indexes rather than actual text.

        seed - LongTensor of length > sequence length that was used for training.
                Ideally prepend a few '\0' characters to the seed.
        """
        
        self.eval()
        # Build h
        h = Variable(torch.zeros(1,self.state_size))
        for i in range(len(seed)):
            h, output = self.forward(Variable(h.data), Variable(seed[i:i+1]))

        # Generate Text
        generated_idxs = torch.zeros(seed.size(0)+generation_len).long()
        generated_idxs[:seed.size(0)] = seed
        _, new_idx = torch.max(self.softmax(output), 0)
        generated_idxs[seed.size(0)] = new_idx.data[0]
        for i in range(seed.size(0)+1,generated_idxs.size(0)):
            h, output = self.forward(Variable(h.data), Variable(generated_idxs[i-1:i]))
            _, new_idx = torch.max(self.softmax(output),0)
            generated_idxs[i] = new_idx.data[0]
        self.train(mode=True)
        return generated_idxs
        
    
    def verbalize(self, semantic_tensor, sentence_len, lr=0.01, start_std=0.01, epochs=50):
        """
        This function seeks to describe in a sentence a tensor that contains semantics
        within the scope of the training data. Returns embedding indexes rather than actual text.

        semantic_tensor - FloatTensor to be verbalized (in a single sentence)
        sentence_len - length of desired generated sentence
        lr - learning rate for sentence updates
        start_std - starting standard deviation for sentence matrix
        epochs - num training epochs for sentence training
        """

        h = Variable(torch.zeros(1,state_size).float())
        means = torch.zeros(sentence_size,self.emb_size)
        sentence = Variable(torch.normal(means, std=start_std), requires_grad=True)
        for p in self.parameters():
            p.requires_grad = False
            
        for epoch in range(epochs):
            exit_sum = Variable(torch.zeros(1,self.emb_size))
            for i in range(sentence_len):
                h, exit_x = self.subforward(Variable(h.data),sentence)
                exit_sum += exit_x
            loss = self.mse(exit_sum, Variable(semantic_tensor))
            loss.backward()
            sentence.data = sentence.data - lr*sentence.data.grad

        idxs = []
        for sent in sentence.data:
            idxs.append(self.closest_emb_idx(sent))

        for p in self.parameters():
            p.requires_grad = True

        return idxs 
                    
    def closest_emb_idx(self, tensor):
        """
        Finds closest embedding tensor to the argued tensor. Distance is guaged using 
        mean squared difference.

        tensor - FloatTensor with shape (self.emb_size,) that will be compared with each embedding tensor
        """
        tensor = tensor.squeeze()
        smallest_mse = torch.sum((tensor-self.embeddings.data[1])**2)
        emb_idx = 0
        for i,emb in enumerate(self.embeddings.data[1:]: 
            mse = torch.sum((tensor-emb)**2)
            if mse < smallest_mse:
                emb_idx = i
        return emb_idx

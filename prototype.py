import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.optim as optim
import nltk.tokenize
from utils import split_token
from keras.utils.data_utils import get_file
import model

# Hyperparameters
seq_len = 30
embedding_size1 = 20
embedding_size2 = 30
batch_size = 64
core_epochs = 1
expanded_epochs = 1
combined_epochs = 1

lr = 0.01
optimizer = "adam"


# Get and prepare data
data_path1 = get_file('mc500.train.tsv', origin="https://raw.githubusercontent.com/mcobzarenco/mctest/master/data/MCTest/mc500.train.tsv")
data_path2 = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
data1 = open(data_path1, 'r')
data2 = open(data_path2, 'r')
stories = []
for story in data1:
    tokens = nltk.tokenize.word_tokenize(story.split('\t')[2])
    words = []
    for token in tokens:
        if "\\" in token:
            words += split_token(token)
        else:
            words.append(token)
    stories.append(words)

print("Num Stories:", len(stories))

words1 = set()
words1.add("\0")
for i,story in enumerate(stories):
    for j,word in enumerate(story):
        words1.add(word)
print("Num unique words data1:", len(words1))

text2 = nltk.tokenize.word_tokenize(data2.read())
words2 = set(text2)
print("Num unique words data2:", len(words2))

words = words1.union(words2)
print("Num unique words total:", len(words))

word_to_idx = {w:i for i,w in enumerate(words1)}
idx_to_word = {i:w for i,w in enumerate(words1)}
word_to_idx = {**word_to_idx, **{w:i+len(words1) for i,w in enumerate(words2-words1)}}
idx_to_word = {**idx_to_word, **{i+len(words1):w for i,w in enumerate(words2-words1)}}

# Flatten dataset1 (specific to MCTest dataset)
text1 = []
for story in stories:
    text1  += ["\0"]*8 # Used to seperate stories
    text1 += story


net = model.Model(words1, embedding_size1, words2|words1, embedding_size2)

if torch.cuda.is_available():
    print("Cuda enabled")
    torch.FloatTensor = torch.cuda.FloatTensor
    torch.LongTensor = torch.cuda.LongTensor
    net = net.cuda()

# Reshape into batches
X1 = [[word_to_idx[text1[i+j]] for j in range(seq_len)] for i in range(len(text1)-seq_len-1)]
Y1 = [[word_to_idx[text1[i+j]] for j in range(seq_len)] for i in range(1,len(text1)-seq_len)]

X1 = torch.LongTensor(X1[:(len(X1)//batch_size)*batch_size])
Y1 = torch.LongTensor(Y1[:(len(X1)//batch_size)*batch_size])

X1 = X1.view(batch_size, X1.size(0)//batch_size, seq_len).permute(1,2,0)
Y1 = Y1.view(batch_size, Y1.size(0)//batch_size, seq_len).permute(1,2,0)

assert len(X1) == len(Y1)
print("X1 shape:", list(X1.size()))
print("Y1 shape:", list(Y1.size()))

X2 = [[word_to_idx[text2[i+j]] for j in range(seq_len)] for i in range(len(text2)-seq_len-1)]
Y2 = [[word_to_idx[text2[i+j]] for j in range(seq_len)] for i in range(1,len(text2)-seq_len)]

X2 = torch.LongTensor(X2[:(len(X2)//batch_size)*batch_size])
Y2 = torch.LongTensor(Y2[:(len(X2)//batch_size)*batch_size])

X2 = X2.view(batch_size, X2.size(0)//batch_size, seq_len).permute(1,2,0)
Y2 = Y2.view(batch_size, Y2.size(0)//batch_size, seq_len).permute(1,2,0)

assert len(X2) == len(Y2)
print("X2 shape:", list(X2.size()))
print("Y2 shape:", list(Y2.size()))


# Train core model
print("Begin Core Training")
for epoch in range(core_epochs):
    print("Begin Epoch", epoch)
    net.optimize(X1, Y1, net.core)

# Train expanded model
print("Begin Expanded Training")
for epoch in range(expanded_epochs):
    print("Begin Epoch", epoch)
    net.sync_expanded()
    net.optimize(X2, Y2, net.expanded)

# Intermittently train core and expanded models
print("Begin Combined Training")
for epoch in range(combined_epochs):
    print("Begin Epoch", epoch)
    net.sync_expanded()
    net.optimize(X2,Y2,net.expanded)
    net.sync_core(average=True)
    net.optimize(X1, Y1, net.core)



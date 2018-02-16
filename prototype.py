import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.optim as optim
from tokenizer import split_token, tokenize
from keras.utils.data_utils import get_file
import model
import sys


if __name__ == '__main__':
    # Hyperparameters
    seq_len = 30
    emb_size1 = 20
    emb_size2 = 30
    batch_size = 64
    core_epochs = 10
    expanded_epochs = 10
    combined_epochs = 10
    lr = 0.01
    generation_len = 100
    rnn_type = "nested_lstm"
    train_type = "core_expanded_combined"
    save_file = "test_net.p"
    resume = False
    debug = False

    if len(sys.argv) > 1:
        for arg in sys.argv:
            if "seq_len=" in arg: seq_len = int(str(arg[len("seq_len="):]))
            if "emb_size1=" in arg: emb_size1 = int(str(arg[len("emb_size1="):]))
            if "emb_size2=" in arg: emb_size2 = int(str(arg[len("emb_size2="):]))
            if "core_epochs=" in arg: core_epochs = int(str(arg[len("core_epochs="):]))
            if "expanded_epochs=" in arg: expanded_epochs = int(str(arg[len("expanded_epochs="):]))
            if "combined_epochs=" in arg: combined_epochs = int(str(arg[len("combined_epochs="):]))
            if "lr=" in arg: lr = float(str(arg[len("lr="):]))
            if "rnn_type=" in arg: rnn_type = str(arg[len("rnn_type="):])
            if "train_type=" in arg: train_type = str(arg[len("train_type="):])
            if "save_file=" in arg: save_file = str(arg[len("save_file="):])
            if "generation_len=" in arg: generation_len = int(str(arg[len("generation_len="):]))
            if "resume" in arg and "False" not in arg: resume=True
            if "debug" in arg and "False" not in arg: debug=True

    if debug:
        core_epochs = 2
        expanded_epochs = 2
        combined_epochs = 2

    print("seq_len:", seq_len)
    print("emb_size1:", emb_size1)
    print("emb_size2:", emb_size2)
    print("batch_size:", batch_size)
    print("core_epochs:", core_epochs)
    print("expanded_epochs:", expanded_epochs)
    print("combined_epochs:", combined_epochs)
    print("lr:", lr)
    print("generation_len:", generation_len)
    print("rnn_type:", rnn_type)
    print("train_type:", train_type)
    print("save_file:", save_file)
    print("debug:", debug)
    print("resume:", resume)

    # Get and prepare data
    data_path1 = get_file('mc500.train.tsv', origin="https://raw.githubusercontent.com/mcobzarenco/mctest/master/data/MCTest/mc500.train.tsv")
    data_path2 = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    data1 = open(data_path1, 'r')
    data2 = open(data_path2, 'r')
    stories = []
    for story in data1:
        tokens = tokenize(story.split('\t')[2])
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

    text2 = tokenize(data2.read())
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


    net = model.Model(words1, emb_size1, words2|words1, emb_size2, seq_len=seq_len, rnn_type=rnn_type)

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

    if debug:
        X1, Y1 = X1[:1], Y1[:1]

    assert len(X1) == len(Y1)
    print("X1 shape:", list(X1.size()))
    print("Y1 shape:", list(Y1.size()))

    X2 = [[word_to_idx[text2[i+j]] for j in range(seq_len)] for i in range(len(text2)-seq_len-1)]
    Y2 = [[word_to_idx[text2[i+j]] for j in range(seq_len)] for i in range(1,len(text2)-seq_len)]

    X2 = torch.LongTensor(X2[:(len(X2)//batch_size)*batch_size])
    Y2 = torch.LongTensor(Y2[:(len(X2)//batch_size)*batch_size])

    X2 = X2.view(batch_size, X2.size(0)//batch_size, seq_len).permute(1,2,0)
    Y2 = Y2.view(batch_size, Y2.size(0)//batch_size, seq_len).permute(1,2,0)

    if debug:
        X2, Y2 = X2[:1], Y2[:1]

    assert len(X2) == len(Y2)
    print("X2 shape:", list(X2.size()))
    print("Y2 shape:", list(Y2.size()))

    seed1 = torch.LongTensor([word_to_idx[text1[i]] for i in range(seq_len+20)])

    # Train core model
    if "core" in train_type:
        print("Begin Core Training")
        for epoch in range(core_epochs):
            print("Begin Epoch", epoch)
            net.optimize(X1, Y1, net.core)
            torch.save(net.state_dict(), save_file)
            if generation_len > 0:
                gen_idxs = net.core.generate_text(seed1, generation_len)
                gen_text = [idx_to_word[idx] for idx in gen_idxs.tolist()]
                txt = " ".join(gen_text)+"ENDTEXT"
                if not torch.cuda.is_available():
                    print(txt)
                net.savetxt(txt, save_file[:-len(".p")]+"_gentxt.txt", "core", epoch)
        net.core.flush_log(save_file[:-len(".p")]+"_core.csv")
        net = model.Model(words1, emb_size1, words2|words1, emb_size2, seq_len=seq_len, rnn_type=rnn_type)

    # Train expanded model
    if "expanded" in train_type:
        print("Begin Expanded Training")
        for epoch in range(expanded_epochs):
            print("Begin Epoch", epoch)
            net.optimize(X2, Y2, net.expanded)
            torch.save(net.state_dict(), save_file)
            if generation_len > 0:
                gen_idxs = net.expanded.generate_text(seed1, generation_len)
                gen_text = [idx_to_word[idx] for idx in gen_idxs.tolist()]
                txt = " ".join(gen_text)+"ENDTEXT"
                if not torch.cuda.is_available():
                    print(txt)
                net.savetxt(txt, save_file[:-len(".p")]+"_gentxt.txt", "expanded", epoch)
        net.expanded.flush_log(save_file[:-len(".p")]+"_expanded.csv")
        net = model.Model(words1, emb_size1, words2|words1, emb_size2, seq_len=seq_len, rnn_type=rnn_type)

    if "combined" in train_type:
        # Intermittently train core and expanded models
        print("Begin Combined Training")
        for epoch in range(combined_epochs):
            print("Begin Epoch", epoch)
            net.sync_expanded()
            net.optimize(X2,Y2,net.expanded)
            torch.save(net.state_dict(), save_file)
            net.sync_core(average=True)
            if generation_len > 0:
                gen_idxs = net.expanded.generate_text(seed1, generation_len)
                gen_text = [idx_to_word[idx] for idx in gen_idxs.tolist()]
                txt = " ".join(gen_text)
                if not torch.cuda.is_available():
                    print(txt)
                net.savetxt(txt, save_file[:-len(".p")]+"_gentxt.txt", "expanded_combined", epoch)
            net.optimize(X1, Y1, net.core)
            torch.save(net.state_dict(), save_file)
            if generation_len > 0:
                gen_idxs = net.core.generate_text(seed1, generation_len)
                gen_text = [idx_to_word[idx] for idx in gen_idxs.tolist()]
                txt = " ".join(gen_text)+"ENDTEXT"
                if not torch.cuda.is_available():
                    print(txt)
                net.savetxt(txt, save_file[:-len(".p")]+"_gentxt.txt", "core_combined", epoch)
        net.core.flush_log(save_file[:-len(".p")]+"_combined_core.csv")
        net.expanded.flush_log(save_file[:-len(".p")]+"_combined_expanded.csv")

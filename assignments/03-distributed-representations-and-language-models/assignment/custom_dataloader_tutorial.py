from collections import namedtuple
from random import shuffle
from IPython import embed
from toolz import take
from itertools import chain, groupby
import torch

def load_data(path):
    dataset = []
    labels = []
    words = []
    with open(path) as f:
        for line in f.readlines():
            text, _, upos, _ = line.rstrip().split('\t')
            dataset.append((text.split(), upos.split()))
            labels += upos.split()
            words += text.split()    

    return dataset, {k:i+1 for i, k in enumerate(set(labels))}, {k:i+1 for i, k in enumerate(set(words))}

def batcher(data, word2i, label2i, batch_size=4):
    shuffle(data)
    datalen = len(data)
    data = iter(data)
    Batch = namedtuple('Batch', ['sentence', 'labels'])

    for i in range(int(datalen/batch_size)+1):
		# select batch_size of sentences from data
        batch = take(batch_size, data)
        batch = list(batch)
        if not batch:
            continue
        
        sentences, tags = zip(*batch)
        # how much to pad
        max_len = max([len(x) for x in sentences])
		
        # convert words/tags to indices and pad
        sentences = torch.tensor([[word2i[y] for y in x]+[0]*(max_len-len(x)) for x in sentences])
        tags = torch.tensor([[label2i[y] for y in x]+[0]*(max_len-len(x)) for x in tags])
        
        yield Batch(sentences, tags)

if __name__ == '__main__':
    data, label2i, word2i = load_data('/home/adamek/git/ud-attention-tagging/data/UD_English-EWT/dev.csv')         
    for x in batcher(data, word2i, label2i):
        embed()
        assert False

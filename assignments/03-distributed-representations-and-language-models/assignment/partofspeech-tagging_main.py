import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
from IPython import embed
import numpy as np

# for GPU: cuda:INT
device = torch.device('cpu')

def main():
    train_iter, test_iter, words_v, labels_v = get_data()

    num_words  = len(words_v.vocab)
    num_labels = len(labels_v.vocab)

    ## create model, optimizer and loss function
    model = PosModel(num_words, num_labels, 50, 50).to(device)
    # Adam is cool!
    opt   = optim.Adam(model.parameters(), lr=0.002) 
    # Cross Entropy combines Negative Log Likelihood loss and Log Softmax
    # reduce = mean loss across examples, i.e. mean( sum( loss(x_i) for x_i in examples) )
    loss  = nn.CrossEntropyLoss(reduction='mean')

    num_params = sum([p.numel() for p in model.parameters()])
    print(model)
    print(f'Model parameters: {num_params}')

    # tell model we're going to train
    model.train()
    for e in range(3):
        # loss for this epoch
        epoch_loss = 0

        for i, batch in enumerate(train_iter):
            sentences = batch.tokens
            labels    = batch.labels
            
            # run sentences through the model
            output = model(sentences)
            
            #embed()
            #assert False

            # computer loss
            # output: from (B, L, C) to (B*L, C)
            # labels: from (B, C) to (B*C)
            # where B = batch size, L = sequence length and C the number of labels
            batch_loss  = loss(output.view(-1,num_labels), labels.view(-1))
            epoch_loss += batch_loss.item()

            # report results
            print(e, (i+1)*sentences.size(0), np.round(epoch_loss/(i+1),4),
                  end='\r')

            # calculate gradients
            batch_loss.backward()
            # update model weights
            opt.step()
            # reset gradients
            opt.zero_grad()
        print()

    ## Test the model on the test_iter
    test_loss = 0
    # tell the model to go into evaluation mode
    model.eval()
    # iterate over the test data and compute the class probabilities, same
    # procedure as before, but now we don't backpropagate
    for i, batch in enumerate(test_iter):
        sentences = batch.tokens
        labels    = batch.labels

        with torch.no_grad(): # dont collect gradients when testing
            output = model(sentences)
        batch_loss = loss(output.view(-1,num_labels), labels.view(-1))
        test_loss += batch_loss.item()

    print('>', np.round(test_loss/(i+1), 4))


class PosModel(nn.Module):
    """Documentation for PosModel
    """
    def __init__(self, num_words, num_labels, i_dim, o_dim):
        super(PosModel, self).__init__()
        self.embedd  = nn.Embedding(num_words, i_dim) # matrix containing the word embeddings
        self.rnn     = nn.LSTM(i_dim, o_dim, batch_first=True) # rnn module
        self.predict = nn.Linear(o_dim, num_labels) # predict which class an example belongs to
        self.dropout = nn.Dropout(0.2)

    def forward(self, sentence):

        #embed()
        #assert False

        # embedd the words in out sentence
        embedded_sentence = self.embedd(sentence)

        # create contextualized representations for our words
        contextualized_representations, (_, _) = self.rnn(embedded_sentence)
        # Why do we do this?
        contextualized_representations = self.dropout(contextualized_representations)

        # return the model predictions
        # each word the model saw is assigned a score (logit) for each of
        # the different classes (part-of-speech tags) in our (train) dataset
        return self.predict(contextualized_representations)

def get_data():
    ddir = 'data/'
    whitespacer = lambda x: x.split(' ')

    # "fields" that process the different columns in our CSV files
    TOKENS = Field(tokenize    = whitespacer,
                   lower       = True,
                   batch_first = True) # enforce the (batch, words) structure

    LABELS = Field(tokenize    = whitespacer,
                   batch_first = True)

    # read the csv files
    train, test = TabularDataset.splits(path   = ddir,
                                        train  = 'train1.csv',
                                        test   = 'test1.csv',
                                        format = 'csv',
                                        fields = [('tokens', TOKENS),
                                                  ('labels', LABELS)],
                                        skip_header       = True,
                                        csv_reader_params = {'delimiter':'\t',
                                                             'quotechar':'½'})

    # build vocabularies based on what our csv files contained and create word2id mapping
    TOKENS.build_vocab(train)
    LABELS.build_vocab(train)

    # create batches from our data, and shuffle them for each epoch
    train_iter, test_iter = BucketIterator.splits((train, test),
                                                  batch_size        = 8,
                                                  sort_within_batch = True,
                                                  sort_key          = lambda x: len(x.tokens),
                                                  shuffle           = True,
                                                  device            = device)

    return train_iter, test_iter, TOKENS, LABELS

if __name__ == '__main__':
    main()

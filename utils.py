import os, sys
from argparse import ArgumentParser
from torchtext import data
from torchtext import datasets
import nltk
import re
from custom_snli_loader import CustomSNLI
import torch
import torch.optim as O
import torch.nn as nn


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def tokenize(sent):
    '''
    data_reader.tokenize('a#b')
    ['a', '#', 'b']
    '''
    sent = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", sent) #.
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]


def load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache):
    inputs = data.Field(lower=False, fix_length=max_seq_len, tokenize=tokenize, init_token='<sos>')
    #inputs = data.Field(lower=False, fix_length=max_seq_len, init_token='<sos>')
    #inputs = data.Field(lower=False, fix_length=None)
    answers = data.Field(sequential=False)
    train, dev, test = CustomSNLI.splits(inputs, answers)
    print('custom data loading debug:')
    print(len(train))
    print(len(dev))
    print(len(test))
    inputs.build_vocab(train, dev, test, max_size=vocab_size)
    answers.build_vocab(train)
    if os.path.isfile(vector_cache):
        inputs.vocab.vectors = torch.load(vector_cache)
    else:
        print('Loading word embeddings...')
        inputs.vocab.load_vectors(word_vectors)
        makedirs(os.path.dirname(vector_cache))
        torch.save(inputs.vocab.vectors, vector_cache)


    print('vocab debug:')
    print('ntokens:%d'%len(inputs.vocab))
    print(inputs.vocab.stoi['What'])
    print(inputs.vocab.freqs['What'])
    print(inputs.vocab.stoi['the'])
    print(inputs.vocab.freqs['the'])
    print(inputs.vocab.itos[0])
    print(inputs.vocab.itos[1])
    print(inputs.vocab.itos[2])
    print(inputs.vocab.itos[3])


    train, dev, test = CustomSNLI.splits(inputs, answers)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                (train, dev, test), batch_size=batch_size, device=0)
    print(len(inputs.vocab))
    print(len(train_iter))
    print(len(val_iter))
    print(len(test_iter))
    train_iter.repeat = False

    return inputs, train_iter, val_iter, test_iter
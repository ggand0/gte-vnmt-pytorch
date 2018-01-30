import os, sys
import time
import math
import glob

import torch
import torch.optim as O
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets

import torch.optim as optim

import nltk
import re
from custom_snli_loader import CustomSNLI
from enc_dec import EncDec
from vnmt import VRAE_VNMT, AttnGRU_VNMT


import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from utils import get_args, makedirs, tokenize, load_dataset
from gte import create_example, reverse_input, show_plot, plot_losses, show_attention


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import io
import torchvision
from PIL import Image
import visdom
vis = visdom.Visdom()

import socket
hostname = socket.gethostname()

##################################
# Load the entailment only snli
##################################
reverse = False

SOS_TOKEN = 2
EOS_TOKEN = 1
batch_size = 250#128
max_seq_len = 52#35
vocab_size = 10000
word_vectors = 'glove.42B.300d'
vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
opt = get_args()
inputs, train_iter, val_iter, test_iter = load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache)

config = opt
d_embed = 300
n_hid = 256#512 # becuase we'll concat two hidden tensors later
n_layers = 1#1#3#1 ## IMPORTANT
dropout = 0.5
if reverse:
    model_name = 'vnmt_pretrain_reverse'
else:
    model_name = 'vnmt_pretrain'
#rnn_type = 'LSTM'
rnn_type = 'GRU'
#dec_type = 'attn'
dec_type = 'vanilla'
config.n_embed = len(inputs.vocab)
ntokens = len(inputs.vocab)
gpu = 0
torch.cuda.set_device(gpu)


##################################
#    Load model
##################################
model = AttnGRU_VNMT(rnn_type, d_embed, n_hid, config.n_embed, max_seq_len, n_layers=n_layers, dropout=dropout)#, word_dropout=0.5)
model.encoder.embeddings.weight.data = inputs.vocab.vectors
model.decoder.embeddings.weight.data = inputs.vocab.vectors
model.encoder.embeddings.weight.requires_grad = False
model.decoder.embeddings.weight.requires_grad = False
print(model.encoder.hidden_dim)
print(model.decoder.hidden_dim)

# setup optimizer
lr=1e-3
epochs = 26
clip = 5.0
log_interval=50
save_interval = 5
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999))
model.cuda()


def evaluate(val_iter, model, n_tokens, eval_batch_size, wv=None):
    """
    Eval acc, bleu, etc.
    """

    # Turn on evaluation mode which disables dropout.
    model.encoder.eval()
    model.decoder.eval()
    total_loss = 0
    loss = 0

    for batch_idx, batch in enumerate(val_iter):
        batch.premise.data = batch.premise.data.transpose(1,0)
        batch.hypothesis.data = batch.hypothesis.data.transpose(1,0)
        loss += model.batchNLLLoss(batch.premise, batch.hypothesis, train=False)

    return loss / float(len(val_iter))


def train(reverse=False, pretrain=False):
    DEBUG=False
    print('gte_vae_pretrain.train')

    model.train()
    total_loss = 0
    total_acc = 0
    # for plotting
    train_losses = []
    val_losses = []

    ntokens = len(inputs.vocab)
    best_val_loss = float('inf')
    if reverse:
        sents = [
            'People are celebrating a birthday.',
            'There are two woman in this picture.'#'Two women who just had lunch hugging and saying goodbye.',
        ]
    else:
        sents = [
            'People are celebrating a victory on the square.',
            'Two women who just had lunch hugging and saying goodbye.',
        ]
    val_loss = evaluate(val_iter, model, ntokens, opt.batch_size)
    val_loss = val_loss.data[0]
    print(val_loss)
    example0 = create_example(inputs, sents[0], max_seq_len)
    example1 = create_example(inputs, sents[1], max_seq_len)
    for i, sent in enumerate(sents):
        sent = '<sos> ' + sent + ' <pad>'
        example = create_example(inputs, sent, max_seq_len)
        output, attns = model.generate(inputs, ntokens, example, max_seq_len)
        show_attention('attn_vis%d'%i, sent, output, attns)

    start_time = time.time()
    iteration = 0


    for epoch in range(epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        total_loss = 0
        train_loss = 0


        for batch_idx, batch in enumerate(train_iter):
            # Turn on training mode which enables dropout.
            model.train()
            optimizer.zero_grad()

            #print(batch.text.data.shape) # 35 x 64
            #batch.text.data = batch.text.data.view(-1, max_seq_len) # -1 instead of opt.batch_size to avoid reshaping err at the end of the epoch
            batch.premise.data = batch.premise.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            batch.hypothesis.data = batch.hypothesis.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            if reverse:
                loss = model.batchNLLLoss(batch.hypothesis, batch.premise, train=True)
            else:
                loss = model.batchNLLLoss(batch.premise, batch.hypothesis, train=True)

            iteration += 1
            loss.backward()
            optimizer.step()
            batch_loss = loss.data
            total_loss += batch_loss
            train_loss += batch_loss

            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss[0] / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, len(train_iter) // max_seq_len, lr,
                    elapsed * 1000 / log_interval, cur_loss, 0))#math.exp(cur_loss)
                total_loss = 0
                start_time = time.time()

        print('Evalating...')
        val_loss = evaluate(val_iter, model, ntokens, opt.batch_size)
        print(val_loss.data[0])
        for i, sent in enumerate(sents):
            sent = '<sos> ' + sent + ' <pad>'
            example = create_example(inputs, sent, max_seq_len)
            output, attns = model.generate(inputs, ntokens, example, max_seq_len)
            show_attention('attn_vis%d_%d'%(epoch,i), sent, output, attns)


        print('Epoch train loss:')
        print(train_loss[0])
        train_loss = train_loss / float(len(train_iter))
        print(train_loss[0])
        train_losses.append(train_loss[0])


        val_loss = evaluate(val_iter, model, ntokens, opt.batch_size)
        val_loss = val_loss.data[0]
        val_losses.append(val_loss)
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            with open('%s_%s_gte_best.pkl'%(model_name, rnn_type.lower()), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #lr /= 4.0
            #print('lr annealed: %f'%lr)
            pass
        if epoch % save_interval == 0:
            with open('%s_%s_gte_e%d.pkl'%(model_name, rnn_type.lower(), epoch), 'wb') as f:
                torch.save(model, f)

        # save train/val loss lists
        with open('train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('val_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
        plot_losses(train_losses, 'train', 'train_loss.eps')
        plot_losses(val_losses, 'validation', 'val_loss.eps')
        show_plot(train_losses, val_losses, 'train-val_loss.eps')


    print(train_losses)
    print(val_losses)

    # save train/val loss lists
    with open('train_losses.pickle', 'wb') as f:
        pickle.dump(train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_losses.pickle', 'wb') as f:
        pickle.dump(val_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    show_plot(train_losses, val_losses, 'train-val_loss.eps')



if __name__ == "__main__":
    print('Pre-training attentive GRU for VNMT...')
    train(reverse=False)
    #train(reverse=True)



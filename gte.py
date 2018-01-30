# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

import os, sys, time, glob
import math, re, nltk, pickle

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from argparse import ArgumentParser

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from custom_snli_loader import CustomSNLI
from enc_dec import EncDec
from utils import get_args, makedirs, tokenize, load_dataset

import io
import torchvision
from PIL import Image
import visdom
import socket
vis = visdom.Visdom()
hostname = socket.gethostname()


def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf).convert("RGB")), win=attn_win, opts={'title': attn_win})

def show_attention(filename, input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    #plt.show()
    plt.savefig(filename)
    plt.close()

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
        _loss = model.batchNLLLoss(batch.premise, batch.hypothesis)
        loss += _loss

    return loss / float(len(val_iter))


def create_example(inputs, sent, max_seq_len):
    example = [2]
    #example += [1] * (N - len(example)) # 1: <pad>, 2: <sos>
    words = sent.split() # e.g.'Two women and one man are drinking beer in a bar.'
    for i, w in enumerate(words):
        example.append(inputs.vocab.stoi[w])
    example.append(1)
    return example

def reverse_input(inp, target):
    # tensor: batch_size x seq_len
    samples = [0, 100]
    for index in samples:
        inp_out = []
        target_out = []
        for i, word_idx in enumerate(inp.data[index]):
            inp_out.append(inputs.vocab.itos[word_idx])
            target_out.append(inputs.vocab.itos[ target.data[index][i]] )
        print('#'*50)
        print(inp_out)
        print(target_out)


def show_plot(train_losses, val_losses, filename):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    ax.legend()
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross entropy loss')
    plt.savefig(filename)

def plot_losses(losses, label, filename):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(losses, label=label)
    ax.legend()
    plt.xlabel('Number of epochs')
    plt.ylabel('Cross entropy loss')
    plt.savefig(filename)



def train():
    DEBUG = False

    # Turn on training mode which enables dropout.
    model.encoder.train()
    model.decoder.train()
    total_loss = 0
    # for plotting
    train_losses = []
    val_losses = []

    ntokens = len(inputs.vocab)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')


    sents = [
        #'People are celebrating a victory on the square.',
        #'Two women who just had lunch hugging and saying goodbye.',
        'A boy is jumping on skateboard in the middle of a red bridge.',
        'A small group of church-goers watch a choir practice.',
        'An indian woman is washing and cleaning dirty laundry at a lake in the background is a kid who appears to have jumped into the lake .',
    ]
    val_loss = evaluate(val_iter, model, ntokens, batch_size)
    val_loss = val_loss.data[0]
    print('initial val loss: %f'%val_loss)
    print('Generating example sentences...')
    for i, sent in enumerate(sents):
        example = create_example(inputs, sent, max_seq_len)
        output, attns = model.generate(inputs, ntokens, example, max_seq_len)
        print(output)
        if dec_type == 'attn':
            show_attention('attn_vis%d'%i, '<sos> ' + sent, output, attns)
    start_time = time.time()


    for epoch in range(epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        total_loss = 0
        train_loss = 0


        for batch_idx, batch in enumerate(train_iter):
            # Turn on training mode which enables dropout.
            model.encoder.train()
            model.decoder.train()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            #print(batch.text.data.shape) # 35 x 64
            batch.premise.data = batch.premise.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            batch.hypothesis.data = batch.hypothesis.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            loss = model.batchNLLLoss(batch.premise, batch.hypothesis)


            loss.backward()
            torch.nn.utils.clip_grad_norm(model.encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm(model.decoder.parameters(), clip)
            enc_optimizer.step()
            dec_optimizer.step()


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
        val_loss = evaluate(val_iter, model, ntokens, batch_size)
        print(val_loss.data[0])
        print('Generating example sentences...')
        for i, sent in enumerate(sents):
            example = create_example(inputs, sent, max_seq_len)
            output, attns = model.generate(inputs, ntokens, example, max_seq_len)
            print(output)
            if dec_type == 'attn':
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
            with open('%d-%s_gte_best.pkl'%(n_layers, dec_type), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #lr /= 4.0
            #print('lr annealed: %f'%lr)
            pass
        if epoch % save_interval == 0:
            with open('%d-%s_gte_e%d.pkl'%(n_layers, dec_type, epoch), 'wb') as f:
                torch.save(model, f)

        # save train/val loss lists
        with open('train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('val_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
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
    ##################################
    # Load the entailment only snli
    ##################################
    batch_size = 250#128
    max_seq_len = 35
    vocab_size = 10000
    word_vectors = 'glove.42B.300d'

    vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
    opt = get_args()
    inputs, train_iter, val_iter, test_iter = load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache)


    config = opt
    d_embed = 300
    n_hid = 512
    n_layers = 3#2#3
    dropout = 0.5
    #rnn_type = 'LSTM'
    rnn_type = 'GRU'
    dec_type = 'attn'
    #dec_type = 'vanilla'
    config.n_embed = len(inputs.vocab)
    ntokens = len(inputs.vocab)

    gpu = 0
    torch.cuda.set_device(gpu)

    ##################################
    #    Load model
    ##################################
    #model = model.RNNmodel('LSTM', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    #model = RNNmodel('LSTM', ntokens, n_embed, n_hid, n_layers, dropout, tied)
    #model = Generator(rnn_type, d_embed, opt.d_hidden, config.n_embed, max_seq_len)
    model = EncDec(rnn_type, dec_type, d_embed, n_hid, config.n_embed, max_seq_len, n_layers=n_layers, dropout=dropout)#, word_dropout=0.5)
    model.encoder.embeddings.weight.data = inputs.vocab.vectors
    model.decoder.embeddings.weight.data = inputs.vocab.vectors
    model.encoder.embeddings.weight.requires_grad = False
    model.decoder.embeddings.weight.requires_grad = False

    # setup optimizer
    lr=1e-3#5e-4#0.0001#1e-2#1e-4
    epochs = 26#101
    clip = 5.0#.25
    log_interval=50
    save_interval = 5
    #enc_optimizer = optim.Adam(model.encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    #dec_optimizer = optim.Adam(model.decoder.parameters(), lr=lr, betas=(0.9, 0.999))
    enc_parameters = filter(lambda p: p.requires_grad, model.encoder.parameters())
    #enc_optimizer = optim.Adam(enc_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)
    enc_optimizer = optim.Adam(enc_parameters, lr=lr, betas=(0.9, 0.999))
    dec_parameters = filter(lambda p: p.requires_grad, model.decoder.parameters())
    #dec_optimizer = optim.Adam(dec_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)
    dec_optimizer = optim.Adam(dec_parameters, lr=lr, betas=(0.9, 0.999))

    model.encoder.cuda()
    model.decoder.cuda()


    print('Training seq-to-seq RNN...')
    train()



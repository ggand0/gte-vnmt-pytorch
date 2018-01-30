import os, sys
import time
import math
import glob
import copy

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
from vnmt import VRAE_VNMT

import matplotlib
matplotlib.use('qt5agg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from utils import get_args, makedirs, tokenize, load_dataset
from gte import create_example, reverse_input, show_plot, plot_losses


##################################
# Load the entailment only snli
##################################
SOS_TOKEN = 2
EOS_TOKEN = 1
batch_size = 250#256#128
max_seq_len = 52#35
vocab_size = 10000
word_vectors = 'glove.42B.300d'
vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
opt = get_args()
inputs, train_iter, val_iter, test_iter = load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache)


config = opt
d_embed = 300
n_hid = 256#512 # becuase we'll concat two hidden tensors later
n_layers = 1 ## IMPORTANT
dropout = 0.5
model_name = 'vnmt'
#rnn_type = 'LSTM'
rnn_type = 'GRU'
#dec_type = 'attn'
dec_type = 'vanilla'
config.n_embed = len(inputs.vocab)
ntokens = len(inputs.vocab)
gpu = 0
torch.cuda.set_device(gpu)
finetune = True


##################################
#    Load model
##################################
model = VRAE_VNMT(rnn_type, d_embed, n_hid, config.n_embed, max_seq_len, n_layers=n_layers, dropout=dropout)#, word_dropout=0.5)
model.encoder_prior.embeddings.weight.data = inputs.vocab.vectors
model.encoder_post.embeddings.weight.data = inputs.vocab.vectors
model.decoder.embeddings.weight.data = inputs.vocab.vectors
model.encoder_prior.embeddings.weight.requires_grad = False
model.encoder_post.embeddings.weight.requires_grad = False
model.decoder.embeddings.weight.requires_grad = False

if finetune:
    # Initialize enc/dec's weights with the pretrained model
    loaded_model = torch.load('vnmt_pretrain_gru_gte_best.pkl', map_location=lambda storage, locatoin: storage.cuda(gpu))
    print(loaded_model.encoder.hidden_dim)

    #loaded_model = torch.load('vnmt_pretrain_3-gru_12162017/vnmt_pretrain_gru_gte_best.pkl', map_location=lambda storage, locatoin: storage.cuda(gpu))
    #model.encoder_prior = loaded_model.encoder
    model.decoder = loaded_model.decoder
    model.encoder_prior =  copy.deepcopy(loaded_model.encoder)
    model.encoder_post =  copy.deepcopy(loaded_model.encoder)
    print(model.encoder_prior.hidden_dim)#512
    print(model.encoder_post.hidden_dim)#512
    #loaded_reverse = torch.load('vnmt_pretrain_reverse_1-gru1e-3_12082017/vnmt_pretrain_reverse_gru_gte_e10.pkl', map_location=lambda storage, locatoin: storage.cuda(gpu))
    #model.encoder_post = loaded_reverse.encoder

    model.encoder_prior.cuda()
    model.encoder_post.cuda()
    model.decoder.cuda()
    model.encoder_prior.embeddings.weight.requires_grad = False
    model.encoder_post.embeddings.weight.requires_grad = False
    model.decoder.embeddings.weight.requires_grad = False


# setup optimizer
lr = 1e-4#5e-5
epochs = 26
clip = 5.0
log_interval = 50
save_interval = 1
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999))
model.cuda()


def evaluate(val_iter, model, n_tokens, eval_batch_size, kld_weight=1.0, wv=None):
    """
    Eval acc, bleu, etc.
    """

    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.encoder_prior.eval()
    model.encoder_post.eval()
    model.decoder.eval()
    total_loss = 0
    loss = 0

    for batch_idx, batch in enumerate(val_iter):
        batch.premise.data = batch.premise.data.transpose(1,0)
        batch.hypothesis.data = batch.hypothesis.data.transpose(1,0)
        _loss, _kld = model.batchNLLLoss(batch.premise, batch.hypothesis, train=False)
        loss += _loss + kld_weight * _kld # full kld when evaluation?

    return loss / float(len(val_iter))



def kld_coef(i, batch_size):
    #return (math.tanh((i - 17500)/1000) + 1)/2 # 700 minibatches * 25 epochs = 17500
    return (math.tanh( (i - int(3500/(batch_size/float(32))) ) / 1000) + 1)/2 # bs: 256 vs 32. 256/32=8. 3500/8 = 437.5


def plot_vae_loss(nlls, klds, kld_weights, filename):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    ax.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Negative log likelihood loss')
    plt.savefig(filename)


def train(pretrain=False, kld_annealing=True):
    DEBUG=False
    print('gte_vae.train')
    print('lr=%F'%lr)

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_acc = 0
    # for plotting
    train_losses = []
    val_losses = []
    kld_values = [] # unweighted values
    kld_weights = []
    nlls = []

    ntokens = len(inputs.vocab)
    best_val_loss = float('inf')

    sents = [
        'People are celebrating a victory on the square.',
        'Two women who just had lunch hugging and saying goodbye.',
    ]

    iteration = 0
    if kld_annealing:

        kld_weight = kld_coef(iteration, batch_size)
    else:
        kld_weight = 1.0
    val_loss = evaluate(val_iter, model, ntokens, opt.batch_size, kld_weight=kld_weight)
    val_loss = val_loss.data[0]

    print('kld_annealing:')
    print(kld_annealing)
    print('Eavluating...')
    print(val_loss)
    example0 = create_example(inputs, sents[0], max_seq_len)
    example1 = create_example(inputs, sents[1], max_seq_len)
    print(model.generate(inputs, ntokens, example0, max_seq_len))
    print(model.generate(inputs, ntokens, example1, max_seq_len))

    start_time = time.time()



    # plot / dump check before proceeding with training
    kld_stats = { 'nll': nlls, 'kld_values': kld_values, 'kld_weights': kld_weights }
    with open('kld_stats.pkl', 'wb') as f:
        pickle.dump(kld_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    plot_losses([0, 1, 2, 3, 4], 'train', 'train_loss.eps')



    for epoch in range(epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        total_loss = 0
        train_loss = 0


        for batch_idx, batch in enumerate(train_iter):
            # Turn on training mode which enables dropout.
            model.train()
            model.encoder_prior.train()
            model.encoder_post.train()
            model.decoder.train()
            optimizer.zero_grad()

            #print(batch.text.data.shape) # 35 x 64
            #batch.text.data = batch.text.data.view(-1, max_seq_len) # -1 instead of opt.batch_size to avoid reshaping err at the end of the epoch
            batch.premise.data = batch.premise.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            batch.hypothesis.data = batch.hypothesis.data.transpose(1,0) # should be 64x35 [batch_size x seq_len]
            #nll, kld = model.batchNLLLoss(batch.premise, batch.hypothesis)
            nll, kld = model.batchNLLLoss(batch.premise, batch.hypothesis, train=True)

            # KLD Cost Annealing
            # ref: https://arxiv.org/pdf/1511.06349.pdf
            iteration += 1
            if kld_annealing:
                kld_weight = kld_coef(iteration, batch_size)
            else:
                kld_weight = 1.0
            loss = nll + kld_weight * kld

            nlls.append(nll.data)
            kld_values.append(kld.data)
            kld_weights.append(kld_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.encoder_prior.parameters(), clip)
            torch.nn.utils.clip_grad_norm(model.encoder_post.parameters(), clip)
            torch.nn.utils.clip_grad_norm(model.decoder.parameters(), clip)
            #torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()


            batch_loss = loss.data
            total_loss += batch_loss
            train_loss += batch_loss

            if batch_idx % log_interval == 0 and batch_idx > 0:
                print('iteration: %d' % iteration)
                print('kld_weight: %.16f' % kld_weight)
                print('nll: %.16f' % nll.data[0])
                print('kld_value: %.16f' % kld.data[0])
                cur_loss = total_loss[0] / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, len(train_iter) // max_seq_len, lr,
                    elapsed * 1000 / log_interval, cur_loss, 0))#math.exp(cur_loss)
                total_loss = 0
                start_time = time.time()

        print('Evalating...')
        val_loss = evaluate(val_iter, model, ntokens, opt.batch_size, kld_weight=kld_weight)
        print(val_loss.data[0])
        print(model.generate(inputs, ntokens, example0, max_seq_len))
        print(model.generate(inputs, ntokens, example1, max_seq_len))

        print(nlls[-1])
        print(kld_values[-1])
        print(kld_weights[-1])
        print('Epoch train loss:')
        print(train_loss[0])
        train_loss = train_loss / float(len(train_iter))
        print(train_loss[0])
        train_losses.append(train_loss[0])


        val_loss = evaluate(val_iter, model, ntokens, opt.batch_size, kld_weight=kld_weight)
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
        kld_stats = { 'nll': nlls, 'kld_values': kld_values, 'kld_weights': kld_weights }
        with open('kld_stats.pkl', 'wb') as f:
            pickle.dump(kld_stats, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    print('Training VRAE...')
    train(kld_annealing=False)
    #train(kld_annealing=True)



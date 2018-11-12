import os, sys, re, time, math, random, pickle
import nltk
import torch
import torch.optim as O
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets

import matplotlib, visdom, socket
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

from custom_snli_loader import CustomSNLI
from enc_dec import EncDec
from vnmt import VRAE_VNMT, AttnGRU_VNMT
from utils import get_args, makedirs, tokenize, load_dataset
from gte import create_example, reverse_input, show_plot, plot_losses, show_attention

vis = visdom.Visdom()
hostname = socket.gethostname()


##################################
# Load the entailment only snli
##################################
SOS_TOKEN = 2
EOS_TOKEN = 1
reverse = False
batch_size = 400
max_seq_len = 52#35
vocab_size = 10000
##word_vectors = 'glove.42B.300d'
word_vectors = 'glove.6B.300d'
vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
opt = get_args()
inputs, train_iter, val_iter, test_iter = load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache)

config = opt
d_embed = 300
n_hid = 250
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

##################################
#    Load model: no fine-tuning
##################################
model = AttnGRU_VNMT(rnn_type, d_embed, n_hid, config.n_embed, max_seq_len, n_layers=n_layers, dropout=dropout, word_dropout=0.5)
'''model.encoder.embeddings.weight.data = inputs.vocab.vectors
model.decoder.embeddings.weight.data = inputs.vocab.vectors
model.encoder.embeddings.weight.requires_grad = False
model.decoder.embeddings.weight.requires_grad = False'''
model.embeddings.weight.data = inputs.vocab.vectors
model.embeddings.weight.requires_grad = False

# setup optimizer
lr = 1e-3
epochs = 26
clip = 5.0
log_interval = 50
save_interval = 5
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999))

#cuda = and torch.cuda.is_available()
cuda = True
device = torch.device("cuda" if cuda else "cpu")
model.to(device)

def evaluate(val_iter, model, n_tokens, eval_batch_size, wv=None):
    """
    Eval acc, bleu, etc.
    """

    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss = 0
    for batch_idx, batch in enumerate(val_iter):
        #print(batch_idx)
        s, s_lengths = batch.premise
        t, t_lengths = batch.hypothesis
        s = s.to(device)
        t = t.to(device)
        _loss = model.batchNLLLoss(s, s_lengths, t, t_lengths, device, train=False)
        loss += _loss.item()
    return loss / float(len(val_iter))


def train(reverse=False, pretrain=False):
    print('Pretraining VNMT for GTE...')
    model.train()
    total_loss = 0
    total_acc = 0
    train_losses = [] # for plotting
    val_losses = []
    attn_weights = [[],[]]

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

    example0 = create_example(inputs, sents[0], max_seq_len)
    example1 = create_example(inputs, sents[1], max_seq_len)
    for i, sent in enumerate(sents):
        sent = '<sos> ' + sent + ' <pad>'
        example = create_example(inputs, sent, max_seq_len)
        output, attns = model.generate(inputs, ntokens, example, max_seq_len, device)
        ##show_attention('attn_vis%d'%i, sent, output, attns)
        attn_weights[i].append((output, attns))

    start_time = time.time()
    iteration = 0
    sys.stdout.flush()
    for epoch in range(epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        total_loss = 0
        train_loss = 0

        for batch_idx, batch in enumerate(train_iter):
            # Turn on training mode which enables dropout.
            model.train()
            optimizer.zero_grad()
            s, s_lengths = batch.premise
            t, t_lengths = batch.hypothesis
            s = s.to(device)
            t = t.to(device)

            if reverse:
                _loss = model.batchNLLLoss(t, t_lengths, s, s_lengths, device, train=False)
            else:
                _loss = model.batchNLLLoss(s, s_lengths, t, t_lengths, device, train=False)

            _loss.backward()
            optimizer.step()
            loss = _loss.item()
            total_loss += loss
            train_loss += loss
            iteration += 1

            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:03.3f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, len(train_iter), lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        print('Evalating...')
        val_loss = evaluate(val_iter, model, ntokens, opt.batch_size)
        val_losses.append(val_loss)
        for i, sent in enumerate(sents):
            sent = '<sos> ' + sent + ' <pad>'
            example = create_example(inputs, sent, max_seq_len)
            output, attns = model.generate(inputs, ntokens, example, max_seq_len, device)
            ##show_attention('attn_vis%d_%d'%(epoch,i), sent, output, attns)
            attn_weights[i].append((output, attns))

        train_loss = train_loss / float(len(train_iter))
        print('Epoch train loss:')
        print(train_loss)
        train_losses.append(train_loss)

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
        with open('attn_weights.pkl', 'wb') as f:
            pickle.dump(attn_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        ##plot_losses(train_losses, 'train', 'train_loss.pdf')
        ##plot_losses(val_losses, 'validation', 'val_loss.pdf')
        ##show_plot(train_losses, val_losses, 'train-val_loss.pdf')
        sys.stdout.flush()

    # Print the loss history just in case
    print(train_losses)
    print(val_losses)

    # save train/val loss lists
    with open('train_losses.pickle', 'wb') as f:
        pickle.dump(train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_losses.pickle', 'wb') as f:
        pickle.dump(val_losses, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('attn_weights.pkl', 'wb') as f:
            pickle.dump(attn_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    ##show_plot(train_losses, val_losses, 'train-val_loss.eps')


if __name__ == "__main__":
    print('Pre-training attentive GRU for VNMT...')
    train(reverse=False)
    #train(reverse=True)


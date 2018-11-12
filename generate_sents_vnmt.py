import os, sys, re, time, math, random, pickle, copy
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


##################################
# Load the entailment only snli
##################################
SOS_TOKEN = 2
EOS_TOKEN = 1
batch_size = 400
max_seq_len = 52#35
vocab_size = 10000
word_vectors = 'glove.6B.300d'
vector_cache = os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
opt = get_args()
inputs, train_iter, val_iter, test_iter = load_dataset(batch_size, max_seq_len, vocab_size, word_vectors, vector_cache)


config = opt
d_embed = 300
n_hid = 250
n_layers = 1 ## IMPORTANT
dropout = 0.5
model_name = 'vnmt'
#rnn_type = 'LSTM'
rnn_type = 'GRU'
#dec_type = 'attn'
dec_type = 'vanilla'
config.n_embed = len(inputs.vocab)
ntokens = len(inputs.vocab)
cuda = True
device = torch.device("cuda" if cuda else "cpu")
finetune = True


##################################
#    Load model
##################################
model = VRAE_VNMT(rnn_type, d_embed, n_hid, config.n_embed, max_seq_len, n_layers=n_layers, dropout=dropout)#, word_dropout=0.5)
model.to(device)
model.embeddings.weight.data = inputs.vocab.vectors
model.embeddings.weight.requires_grad = False


model = torch.load('vnmt_gru_gte_best.pkl')
##model = torch.load('vnmt_pretrain_gru_gte_best.pkl')
print(type(model))


sents = [
    'people are celebrating a victory on the square .',
    'two women who just had lunch hugging and saying goodbye .',
    'a man selling donuts to a customer during a world exhibition event .',
    'two men and a woman finishing a meal and drinks .',
    'people are running away from the bear .',
    'a boy is jumping on skateboard in the middle of a red bridge .',
    'a big brown dog swims towards the camera .',
    'a small group of church-goers watch a choir practice .',
]

i = 0
example0 = create_example(inputs, sents[0], max_seq_len)
example1 = create_example(inputs, sents[1], max_seq_len)
for i, sent in enumerate(sents):
    sent = sent + ' <pad>'
    example = create_example(inputs, sent, max_seq_len)
    print(example)
    output, attns = model.generate(inputs, ntokens, example, max_seq_len)
    show_attention('attn_vis%d.pdf'%i, sent, output, attns)
    show_attention('attn_vis%d.png'%i, sent, output, attns)
    ##output, attns = model.generate(inputs, ntokens, example, max_seq_len, device)
    ##show_attention('attn_pretrain_vis%d'%i, sent, output, attns)



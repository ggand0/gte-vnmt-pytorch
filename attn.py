# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import numpy as np
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

USE_CUDA = True


#https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq/blob/master/attentionRNN.py
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]


# org version in practical-pytorch
'''
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy
'''

'''
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(weight_init.xavier_uniform(torch.FloatTensor(1, self.hidden_size)))

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        return F.softmax(attn_energies).unsqueeze(1)

    def batch_score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            energy = torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1)

        elif self.method == 'general':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)

            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)).squeeze(1)

        elif self.method == 'concat':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)

            attn_input = torch.cat((hidden.repeat(length, 1, 1), encoder_outputs), dim=2)

            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(self.v.repeat(batch_size, 1, 1), energy.permute(1, 2, 0)).squeeze(1)
        return energy
'''
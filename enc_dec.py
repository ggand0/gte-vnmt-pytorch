# ref: https://github.com/suragnair/seqGAN/blob/master/generator.py

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from masked_cross_entropy import *
from attn import Attn
import pdb
import random
from custom_gru_cell import CustomGRU
SOS_TOKEN = 2
EOS_TOKEN = 1

# ref: https://github.com/suragnair/seqGAN/blob/725686d9b1a58dbf0b9215812374e75e1b9ca982/generator.py
# ref: https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        n_layers=1, dropout=0.1, word_dropout=0.5, bidirectional=False, gpu=True):
        super(EncoderRNN, self).__init__()

        self.hidden_dim = hidden_dim        # same hidden dim
        self.embedding_dim = embedding_dim  # same emb dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu
        #self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.word_dropout = word_dropout

    def forward(self, emb, s_lengths, hidden=None):
        seq_len, batch_size, emb_dim = emb.size()
        #emb = self.drop(emb) # apply a local dropout
        #emb = self.embeddings(s)                                          # batch_size x embedding_dim
        if self.dropout:
            emb = self.drop(emb)
        packed = pack_padded_sequence(emb, s_lengths.cpu().numpy())
        out, hidden = self.rnn(packed, hidden)             # hidden => should be the last output
        out, _ = pad_packed_sequence(out)
        return out, hidden


class DecoderRNN(nn.Module):
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_layers=2, dropout=0.1, word_dropout=None, gpu=True):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim        # same hidden dim
        self.embedding_dim = embedding_dim  # same emb dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        if self.rnn_type in ['LSTM', 'GRU', 'CustomGRU']:
            #self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
            self.rnn = getattr(nn, self.rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)

        self.rnn2out = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.rnn2out.weight.data.uniform_(-initrange, initrange)
        self.rnn2out.bias.data.fill_(0)

    def forward(self, inp, hidden):
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        if self.dropout:
            emb = self.drop(emb)
        output = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim

        out, hidden = self.rnn(output, hidden)            # 1 x batch_size x hidden_dim (out)

        out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out)
        return out, hidden

    def eval_forward(self, inp, hidden):
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        output = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.rnn(output, hidden)            # 1 x batch_size x hidden_dim (out)
        out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        #return F.softmax(out), hidden
        return out, hidden


    def init_hidden(self, batch_size=1):
        #h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.rnn_type == 'LSTM':
            return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())


        if self.gpu:
            return h.cuda()
        else:
            return h


# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, rnn_type, embedding_dim, hidden_dim, output_size, n_layers=1, dropout=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.rnn_type = rnn_type
        self.hidden_size = hidden_dim
        self.output_size = output_size
        self.vocab_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        # Define layers
        self.embeddings = nn.Embedding(output_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_dim)
        #self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        if self.rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
            self.rnn = getattr(nn, self.rnn_type)(embedding_dim+hidden_dim, hidden_dim, n_layers, dropout=dropout)
        #self.out = nn.Linear(hidden_dim, output_size)
        self.out = nn.Linear(hidden_dim*2, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # TODO: FIX BATCHING

        batch_size = word_input.size(0)

        # Get the embedding of the current input word (last output word)
        #word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        #word_embedded = self.dropout(word_embedded)
        embedded = self.embeddings(word_input)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_dim) # S=1 x B x N


        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N

        # Combine embedded input word and attended context, run through RNN
        #print(embedded.size()) # 1,1,300
        #print(context.size())  # 1,1,512
        rnn_input = torch.cat((embedded, context), 2)
        #print(rnn_input.size()) # 1,1,812
        output, hidden = self.rnn(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        context = context.squeeze(0) # added: ref: https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq/blob/master/attentionRNN.py
        #print(output.size()) # 1x512
        #print(context.size()) # 1x512
        #print( torch.cat((output, context), 1).size() ) # 1x1024
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, rnn_type, attn_model, embedding_dim, hidden_dim, output_size, n_layers=1, dropout=0.5):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.rnn_type = rnn_type
        self.attn_model = attn_model
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.vocab_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        # Define layers
        self.embeddings = nn.Embedding(output_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)

        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)

        # Choose attention model
        print(attn_model)
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_dim)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embeddings(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_dim) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn(embedded, last_hidden)


        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N

        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class EncDec(nn.Module): # really ok to use nn.Module?

    def __init__(self, rnn_type, dec_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        n_layers=1, dropout=0.5, word_dropout=None, gpu=True):
        super(EncDec, self).__init__()

        self.rnn_type = rnn_type
        self.dec_type = dec_type
        self.encoder = EncoderRNN(
            rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, bidirectional=False, gpu=True
        )
        #self.decoder = DecoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_layers=2, dropout=0.5, gpu=True)
        #self.decoder = LuongAttnDecoderRNN('general', small_hidden_size, output_lang.n_words, small_n_layers)

        if self.dec_type == 'vanilla':
            self.decoder = DecoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
                n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True)
        elif self.dec_type == 'attn':
            #self.decoder = LuongAttnDecoderRNN(rnn_type, 'general', embedding_dim, hidden_dim, vocab_size, n_layers, dropout=dropout)
            self.decoder = BahdanauAttnDecoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, n_layers, dropout=dropout)

        if gpu:
            self.encoder.cuda()
            self.decoder.cuda()


    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        #loss_fn = nn.CrossEntropyLoss()

        batch_size, seq_len = inp.size()
        loss = 0

        enc_h = self.encoder.init_hidden(batch_size) # (the very first hidden)
        if self.rnn_type == 'LSTM':
            enc_h[0].data.uniform_(-0.1, 0.1)
            enc_h[1].data.uniform_(-0.1, 0.1)
        else:
            enc_h.data.uniform_(-0.1, 0.1)

        inp_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in inp.data.cpu().numpy() ] ) # 1: <pad>

        # SORT YOUR TENSORS BY LENGTH!
        inp_lengths, perm_idx = inp_lengths.sort(0, descending=True)
        inp.data = inp.data[perm_idx]
        target.data = target.data[perm_idx]
        target_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in target.data.cpu().numpy() ] ) # 1: <pad>
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size



        ###################
        # forward encoder
        ###################
        if self.dec_type == 'attn':
            encoder_outputs, enc_h = self.encoder(inp, inp_lengths.tolist(), None)

        elif self.dec_type == 'vanilla':
            for i in range(seq_len):
                out, enc_h = self.encoder(inp[i], enc_h, inp_lengths)


        dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]*batch_size])).long().cuda()
        dec_inp = dec_inp.permute(1, 0) # 128x1

        if self.rnn_type == 'LSTM':
            dec_h = (enc_h[0][:self.decoder.n_layers].cuda(), enc_h[1][:self.decoder.n_layers].cuda())
        else:
            dec_h = enc_h[:self.decoder.n_layers].cuda()
        target_length = target.size()[0]
        all_decoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.decoder.vocab_size)).cuda()

        teacher_forcing_ratio=1.0

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        ###################
        # forward decoder
        ###################
        # attention
        if self.dec_type == 'attn':

            for i in range(target_length):
                out, dec_h, dec_attn = self.decoder.forward(dec_inp, dec_h, encoder_outputs)

                if use_teacher_forcing:
                    # Teacher forcing
                    dec_inp = target[i]         # shape: batch_size,
                else:
                    dec_inp = out.max(1)[1]     # shape: batch_size,

                all_decoder_outputs[i] = out

        elif self.dec_type == 'vanilla':
            # vanilla
            for i in range(target_length):
                out, dec_h = self.decoder.forward(dec_inp, dec_h)
                if use_teacher_forcing:
                    # Teacher forcing
                    dec_inp = target[i]         # shape: batch_size,
                else:
                    dec_inp = out.max(1)[1]     # shape: batch_size,

                all_decoder_outputs[i] = out

        loss = masked_cross_entropy( # bs x seq_len?
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target.transpose(0, 1).contiguous(),
            Variable(target_lengths)
        )


        return loss     # per batch



    def generate(self, inputs, ntokens, example, max_seq_len):
        """
        Generate example
        """
        self.encoder.eval()
        self.decoder.eval()
        SOS_TOKEN = 2
        EOS_TOKEN = 1
        out_seq = []

        ######################
        # encoder initial h
        #####################
        h = self.encoder.init_hidden(1) # (the very first hidden) batch_size = 1

        # create input tensor
        inp = Variable(torch.rand(1, max_seq_len).mul(ntokens).long().cuda(), volatile=True)
        for i in range(max_seq_len):
            inp.data[0][i] = EOS_TOKEN
        for i in range(len(example)):
            inp.data[0][i] = example[i]

        inp_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1) for x in inp.data.cpu().numpy() ] ) # 1: <pad>
        inp = inp.permute(1, 0) # seq_len x bs=1

        ######################
        # forward encoder
        ######################
        dec_type = self.dec_type
        if dec_type == 'vanilla':
            enc_outs, h = self.encoder(inp, inp_lengths.tolist(), None)

        elif dec_type == 'attn':
            enc_outs, h = self.encoder(inp, inp_lengths.tolist(), None)


        ######################
        #   decoder initial h
        ######################
        # create an input with the batch_size of 1
        dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]]), volatile=True).cuda()
        if self.rnn_type == 'LSTM':
            dec_h = (h[0][:self.decoder.n_layers].cuda(), h[1][:self.decoder.n_layers].cuda())
        else:
            dec_h = h[:self.decoder.n_layers].cuda()
        decoder_attentions = torch.zeros(max_seq_len+1, max_seq_len+1)


        ######################
        #   forward decoder
        ######################
        sample_type = 0
        for i in range(max_seq_len):
            if dec_type == 'vanilla':
                out, dec_h = self.decoder.eval_forward(dec_inp, dec_h)
            elif dec_type == 'attn':
                out, dec_h, dec_attn = self.decoder.forward(dec_inp, dec_h, enc_outs)
                decoder_attentions[i,:dec_attn.size(2)] += dec_attn.squeeze(0).squeeze(0).cpu().data

            # 0: argmax
            if sample_type == 0:
                dec_inp = out.max(1)[1]
                max_val, max_idx = out.data.squeeze().max(0)
                word_idx = max_idx[0]

            # 1: tempreture
            elif sample_type == 1:
                temperature = 1.0#1e-2
                word_weights = out.squeeze().data.div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                dec_inp.data.fill_(word_idx)

            #dec_inp = out.max(1)[1]
            output_word = inputs.vocab.itos[word_idx]
            out_seq.append(output_word)

            if word_idx == EOS_TOKEN:
                break

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        if dec_type == 'vanilla':
            return out_seq
        elif dec_type == 'attn':
            return out_seq, decoder_attentions[:i+1, :len(enc_outs)]#[:i+1, :len(example)]


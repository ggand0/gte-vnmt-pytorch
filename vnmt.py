import math, random, pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
from torch.autograd import Variable

from attn import Attn
from masked_cross_entropy import *
from enc_dec import EncoderRNN, DecoderRNN
from custom_gru_cell import CustomGRU, CustomGRUCell


SOS_TOKEN = 2
EOS_TOKEN = 1
UNK_TOKEN = 0


class CustomDecoderRNN(nn.Module):
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_layers=2, dropout=0.1, word_dropout=None, gpu=True):
        super(CustomDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim        # same hidden dim
        self.embedding_dim = embedding_dim  # same emb dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        if self.rnn_type in ['CustomLSTM', 'CustomGRU']:
            #self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
            if self.rnn_type == 'CustomGRU':
                #cell = CustomGRUCell(embedding_dim, hidden_dim)
                self.rnn = CustomGRU(CustomGRUCell, embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            raise

        self.rnn2out = nn.Linear(hidden_dim, vocab_size)

        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.word_dropout = word_dropout

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.rnn2out.weight.data.uniform_(-initrange, initrange)
        self.rnn2out.bias.data.fill_(0)

    def forward(self, inp, hidden, z):
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        if self.dropout:
            emb = self.drop(emb)
        output = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        #out, hidden = self.rnn(output, hidden, z)            # 1 x batch_size x hidden_dim (out)

        out, hidden = self.rnn(output, None, hidden, z.squeeze(dim=0))            # 1 x batch_size x hidden_dim (out)
        out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        #out = F.log_softmax(out)
        return out, hidden

    def eval_forward(self, inp, hidden):
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        output = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.rnn(output, hidden)            # 1 x batch_size x hidden_dim (out)
        out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        #return F.softmax(out), hidden
        return out, hidden


    def init_hidden(self, batch_size=1):
        if self.rnn_type == 'CustomLSTM':
            return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())

        if self.gpu:
            return h.cuda()
        else:
            return h

# ref: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
# ref: https://github.com/DeepLearnXMU/VNMT/blob/master/src/encdec.py
class CustomAttnDecoderRNN(nn.Module):
    """
    Use Bahdanau et al.'s implementation, but instead of concat, directly add the context vector in GRU cell.
    """
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_layers=1, dropout=0.1, word_dropout=None):
        super(CustomAttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim        # same hidden dim
        self.embedding_dim = embedding_dim  # same emb dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        if self.rnn_type in ['CustomLSTM', 'CustomGRU']:
            #self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
            if self.rnn_type == 'CustomGRU':
                #cell = CustomGRUCell(embedding_dim, hidden_dim)
                self.rnn = CustomGRU(CustomGRUCell, embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            raise

        self.attn = Attn('concat', hidden_dim)
        ##self.attn = Attn('concat', hidden_dim, max_seq_len)
        self.rnn2out = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.word_dropout = word_dropout


    def forward(self, emb, hidden, encoder_outputs, z):
        #emb = self.embeddings(inp)                              # batch_size x embedding_dim
        if self.dropout:
            emb = self.drop(emb)
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim

        attn_weights = self.attn(hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N


        out, hidden = self.rnn(emb, None, hidden, context.squeeze(dim=0), z)            # 1 x batch_size x hidden_dim (out)
        out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        ##out = F.log_softmax(out)
        return out, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        if self.rnn_type == 'CustomLSTM':
            return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())

        if self.gpu:
            return h.cuda()
        else:
            return h


class AttnGRU_VNMT(nn.Module):
    """
    Pretains attentive GRU for VNMT.
    """
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        n_layers=1, dropout=0.5, word_dropout=0.5, gpu=True):
        super(AttnGRU_VNMT, self).__init__()

        #self.word_dropout = 1.0#0.75
        self.word_dropout = word_dropout
        self.word_drop = nn.Dropout(word_dropout)
        self.rnn_type = rnn_type
        self.dec_type = 'attn'
        self.n_layers = n_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # encoder for x
        self.encoder = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=n_layers, dropout=dropout, word_dropout=word_dropout
        )
        # encoder for y
        #self.encoder_post = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        #   n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True
        #)


        ################################################
        #     Only supports 1-layer decoder for now
        ################################################
        self.decoder = CustomAttnDecoderRNN('CustomGRU', embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=1, dropout=dropout, word_dropout=word_dropout)


    def batchNLLLoss(self, s, s_lengths, t, t_lengths, device, train=False):
        loss = 0
        batch_size, seq_len = s.size()

        s_lengths, perm_idx = s_lengths.sort(0, descending=True) # SORT YOUR TENSORS BY LENGTH!
        s.data = s.data[perm_idx]
        t.data = t.data[perm_idx]
        s = s.permute(1, 0).to(device)              # seq_len x batch_size
        t = t.permute(1, 0).to(device)              # seq_len x batch_size

        emb_s = self.embeddings(s)
        emb_t = self.embeddings(t)
        emb_t_shift = torch.zeros_like(emb_t)           # 1 is the index for EOS_TOKEN
        emb_t_shift[1:, :, :] = emb_t[:-1, :, :]        # shift the input sentences
        emb_t_shift = self.word_drop(emb_t_shift)

        ############################
        #        Encode x          #
        ############################
        # encode x for both the prior model and the poterior model.
        # linear layers are independent but the encoder to create annotation vectors is shared.
        #enc_h_x = self.encoder.init_hidden(batch_size).to(device) # (the very first hidden)
        enc_h_x = None
        encoder_outputs, encoder_hidden = self.encoder(emb_s, s_lengths, enc_h_x) # torch.Size([12, 250, 256])
        enc_h_x_mean = encoder_outputs.mean(0)
        if self.rnn_type == 'LSTM':
            enc_h_x = encoder_hidden[0]

        enc_h = encoder_hidden
        if self.rnn_type == 'LSTM':
            dec_h = (enc_h[0][:self.decoder.n_layers].to(device), enc_h[1][:self.decoder.n_layers].to(device))
        else:
            dec_h = enc_h[:self.decoder.n_layers].to(device)


        #########################################################
        #  Decode using the last enc_h, context vectors, and z  #
        #########################################################
        #dec_s = Variable(torch.LongTensor([[SOS_TOKEN]*batch_size])).long().to(device)
        #dec_s = dec_s.permute(1, 0) # 128x1
        t_length = t.size()[0]
        all_decoder_outputs = torch.zeros(seq_len, batch_size, self.decoder.vocab_size).to(device)
        use_target = True
        #use_target = True if random.random() < self.word_dropout else False

        for i in range(t_length):
            if use_target:
                dec_s = emb_t_shift[i]         # shape: batch_size,
            else:
                dec_s =  Variable(torch.LongTensor([[UNK_TOKEN]*batch_size])).long().to(device)

            #out, dec_h = self.decoder.forward(dec_s, dec_h, z)
            ##out, dec_h, attn_weights = self.decoder.forward(dec_s, dec_h, encoder_outputs, None) # decode w/o z
            out, dec_h, attn_weights = self.decoder.forward(dec_s, dec_h, encoder_outputs, None) # decode w/o z
            all_decoder_outputs[i] = out


        # Compute masked cross entropy loss
        loss = masked_cross_entropy( # bs x seq_len?
            all_decoder_outputs.transpose(0, 1).contiguous(),
            t.transpose(0, 1).contiguous(),
            t_lengths.to(device)
        )
        return loss


    def generate(self, inputs, ntokens, example, max_seq_len, device, max_words=100):
        """
        Generate example
        """
        print('Generating...')
        self.encoder.eval()
        self.decoder.eval()
        dec_type = self.dec_type
        out_seq = []

        input = Variable(torch.rand(1, max_seq_len).mul(ntokens).long(), volatile=True).to(device)
        for i, wd_idx in enumerate(example):
            input.data[0][i] = wd_idx
        input_words = [inputs.vocab.itos[input.data[0][i]] for i in range(0,max_seq_len)]

        # encoder initial h
        #h = self.encoder.init_hidden(1) # (the very first hidden)
        inp = Variable(torch.rand(1, max_seq_len).mul(ntokens).long().cuda(), volatile=True)
        for i in range(max_seq_len):
            inp.data[0][i] = EOS_TOKEN
        for i in range(len(example)):
            inp.data[0][i] = example[i]

        seq_lengths = torch.LongTensor( [ len(x)-list(x).count(1) for x in inp.data.cpu().numpy() ] ).to(device) # 1: <pad>
        inp = inp.permute(1, 0)


        ############################
        #         Encode x         #
        ############################
        emb = self.embeddings(inp)
        emb_shift = torch.zeros_like(emb)           # 1 is the index for EOS_TOKEN
        emb_shift[1:, :, :] = emb[:-1, :, :]        # shift the input sentences
        emb_shift = self.word_drop(emb_shift)

        encoder_outputs, encoder_hidden = self.encoder(emb, seq_lengths, None)
        #enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # mean pool x: h_f


        #####################################
        # perform reparam trick and get z
        #####################################
        h = encoder_hidden
        if self.rnn_type == 'LSTM':
            h = (h[0].to(device), h[1].to(device))
        else:
            h = h.to(device)

        #####################################
        # perform reparam trick and get z
        #####################################
        # create an input with the batch_size of 1
        #dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]])).to(device)
        dec_emb = emb_shift[0]
        decoder_attentions = torch.zeros(max_seq_len, max_seq_len)
        sample_type = 0
        for i in range(max_seq_len):
            if dec_type == 'vanilla':
                out, h = self.decoder.forward(dec_emb, h, None)
            elif dec_type == 'attn':
                #out, h, dec_attn = self.decoder.forward(dec_inp, h, encoder_outputs, z)
                out, h, dec_attn = self.decoder.forward(dec_emb, h, encoder_outputs, None) # decode w/o z
                padded_attn = F.pad(dec_attn.squeeze(0).squeeze(0), pad=(0, max_seq_len-dec_attn.size(2)), mode='constant', value=EOS_TOKEN)

                ##decoder_attentions[i,:] += dec_attn.squeeze(0).squeeze(0).cpu().data
                decoder_attentions[i,:] += padded_attn.cpu().data

            # 0: argmax
            if sample_type == 0:
                dec_inp = out.max(1)[1]
                dec_emb = self.embeddings(dec_inp)
                max_val, max_idx = out.data.squeeze().max(0)
                word_idx = max_idx[0]
            # 1: tempreture
            elif sample_type == 1:
                temperature = 1.0#1e-2
                word_weights = out.squeeze().data.div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]

            output_word = inputs.vocab.itos[word_idx]
            out_seq.append(output_word)

            if word_idx == EOS_TOKEN:
                break

        return out_seq, decoder_attentions[:i+1, :len(example)-2]


class VRAE_VNMT(nn.Module):
    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        n_layers=1, dropout=0.5, word_dropout=0.5, gpu=True):
        super(VRAE_VNMT, self).__init__()

        self.word_dropout = word_dropout
        self.z_size = 1000 # concat size is absorbed by linear_mu_post etc, so z_size just needs to be equal with hidden_dim
        self.mode = 'vnmt'
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_drop = nn.Dropout(word_dropout)

        self.rnn_type = rnn_type
        self.dec_type = 'attn'
        self.n_layers = n_layers

        self.linear_mu_prior = nn.Linear(hidden_dim, self.z_size) # hidden_dim*1 because we only pass x
        self.linear_sigma_prior = nn.Linear(hidden_dim, self.z_size)
        self.linear_mu_post = nn.Linear(hidden_dim*2, self.z_size) # hidden_dim*2 because we pass x and y
        self.linear_sigma_post = nn.Linear(hidden_dim*2, self.z_size)

        self.encoder_prior = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True
        )
        self.encoder_post = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True
        )

        ################################################
        #     Only supports 1-layer decoder for now
        ################################################
        self.decoder = CustomAttnDecoderRNN('CustomGRU', embedding_dim, hidden_dim, vocab_size, max_seq_len,
            n_layers=1, dropout=dropout, word_dropout=word_dropout) # > We use a fixed word dropout rate of 75%

        # for projecting z into the hidden dim of the decoder so that it can be added inside the GRU cells
        self.linear_z = nn.Linear(self.z_size, self.decoder.hidden_dim) # W_z^(2) and b_z^(2)


    def reparam_trick(self, mu, log_sigma):
        # the reason of log_sigma: https://www.reddit.com/r/MachineLearning/comments/74dx67/d_why_use_exponential_term_rather_than_log_term/
        epsilon = torch.zeros(self.z_size).cuda()
        epsilon.normal_(0, 1) # 0 mean unit variance gaussian
        return Variable(epsilon*torch.exp(log_sigma.data*0.5)+mu.data)


    def vnmt_loss(self, recon_x, target_x, mu_prior, log_sigma_prior, mu_post, log_sigma_post):
        seq_len, batch_size = target_x.size()
        loss_fn = nn.CrossEntropyLoss()
        loss = 0
        for t in range(seq_len):
            loss += loss_fn(recon_x[t], target_x[t])

        total_KLD = 0
        sigma_prior = torch.exp(log_sigma_prior)
        sigma_post = torch.exp(log_sigma_post)

        KLD = ( log_sigma_prior - log_sigma_post + \
            (sigma_post*sigma_post + (mu_post - mu_prior)*(mu_post - mu_prior)) / (2.0*sigma_prior*sigma_prior) - 0.5
        )

        #########################################################
        #  Be careful with the dimension when taking the sum!!!
        #########################################################
        total_KLD += 1.0 * torch.sum(KLD, 1).mean().squeeze()
        return loss, total_KLD


    def batchNLLLoss(self, s, s_lengths, t, t_lengths, device, train=False):
        loss = 0
        batch_size, seq_len = s.size()

        tt = t.clone()
        s_lengths, perm_idx = s_lengths.sort(0, descending=True) # SORT YOUR TENSORS BY LENGTH!
        s.data = s.data[perm_idx]
        t.data = t.data[perm_idx]
        s = s.permute(1, 0).to(device)              # seq_len x batch_size
        t = t.permute(1, 0).to(device)              # seq_len x batch_size

        t_lengths, _perm_idx = t_lengths.sort(0, descending=True) # SORT YOUR TENSORS BY LENGTH!
        tt.data = tt.data[_perm_idx]
        tt = tt.permute(1, 0).to(device)              # seq_len x batch_size

        emb_s = self.embeddings(s)
        emb_t = self.embeddings(t)
        emb_tt = self.embeddings(tt) # for encoding target
        emb_t_shift = torch.zeros_like(emb_t)           # 1 is the index for EOS_TOKEN
        emb_t_shift[1:, :, :] = emb_t[:-1, :, :]        # shift the input sentences
        emb_t_shift = self.word_drop(emb_t_shift)

        ############################
        #     Encode x and y       #
        ############################
        # encode x for both the prior model and the poterior model.
        # linear layers are independent but the encoder to create annotation vectors is shared.
        enc_h_x = None
        encoder_outputs_x, encoder_hidden_x = self.encoder_prior(emb_s, s_lengths, enc_h_x) # torch.Size([12, 250, 256])
        enc_h_x_mean = encoder_outputs_x.mean(0)
        if self.rnn_type == 'LSTM':
            encoder_hidden = encoder_hidden[0]

        enc_h = encoder_hidden_x
        if self.rnn_type == 'LSTM':
            dec_h = (enc_h[0][:self.decoder.n_layers].to(device), enc_h[1][:self.decoder.n_layers].to(device))
        else:
            dec_h = enc_h[:self.decoder.n_layers].to(device)

        # encode y for both the poterior model.
        #enc_h_y = self.encoder_post.init_hidden(batch_size) # (the very first hidden)
        enc_h_y = None
        encoder_outputs_y, encoder_hidden_y = self.encoder_post(emb_tt, t_lengths, enc_h_y) # torch.Size([12, 250, 256])
        enc_h_y_mean = encoder_outputs_y.mean(0) # mean pool y

        ############################
        #      Compute Prior       #
        ############################
        #print(enc_h_x_mean.size()) # 250, 6
        mu_prior = self.linear_mu_prior(enc_h_x_mean)
        log_sigma_prior = self.linear_sigma_prior(enc_h_x_mean)

        ############################
        #     Compute Posterior    #
        ############################
        # define these for evaluation times
        mu_post = Variable(torch.zeros(batch_size, self.z_size)).to(device)
        log_sigma_post = Variable(torch.zeros(batch_size, self.z_size)).to(device)

        # concat h
        enc_h = torch.cat((enc_h_x_mean, enc_h_y_mean), 1) # h_z' => size:

        # get mu and sigma using the last hidden layer's output
        mu_post = self.linear_mu_post(enc_h)
        log_sigma_post = self.linear_sigma_post(enc_h)


        #####################################
        # perform reparam trick and get z
        #####################################
        # Obtain h_z
        z = self.reparam_trick(mu_post, log_sigma_post)

        ## project z into the decoder's hidden_dim so that it can be added in the GRU cells
        he = self.linear_z(z)

        # Take the last hidden state of the encoder and pass it to the decoder
        dec_h = encoder_hidden_x[:self.decoder.n_layers].to(device)


        ########################################################
        #  Decode using the last enc_h, context vectors, and z
        ########################################################
        #dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]*batch_size])).long().to(device)
        #dec_inp = dec_inp.permute(1, 0) # 128x1
        target_length = t.size()[0]
        all_decoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.decoder.vocab_size)).to(device)

        use_target = True#True if random.random() < self.word_dropout else False
        for i in range(target_length):
            dec_emb = emb_t_shift[i]
            #out, dec_h = self.decoder.forward(dec_inp, dec_h, z)
            #out, dec_h, dec_attn = self.decoder.forward(dec_inp, dec_h, encoder_outputs, he)

            out, dec_h, dec_attn = self.decoder.forward(dec_emb, dec_h, encoder_outputs_x, he.unsqueeze(0))
            if use_target:
                #dec_inp = target[i]         # shape: batch_size,
                dec_emb = emb_t_shift[i]
            else:
                dec_inp =  Variable(torch.LongTensor([[UNK_TOKEN]*batch_size])).long().to(device)

            all_decoder_outputs[i] = out


        # Compute the VNMT objective
        loss = self.vnmt_loss(all_decoder_outputs, t, mu_prior, log_sigma_prior, mu_post, log_sigma_post)
        return loss


    def sample(self, inp, max_seq_len):
        self.encoder_prior.eval()
        self.decoder.eval()
        pass

    def generate(self, inputs, ntokens, example, max_seq_len):
        """
        Generate example
        """
        batch_size = 1
        self.encoder_prior.eval()
        self.decoder.eval()
        out_seq = []
        dec_type = self.dec_type
        max_words = 100


        input = Variable(torch.rand(1, max_seq_len).mul(ntokens).long(), volatile=True)
        input.data = input.data.cuda()
        for i, wd_idx in enumerate(example):
            input.data[0][i] = wd_idx
        input_words = [inputs.vocab.itos[input.data[0][i]] for i in range(0,max_seq_len)]


        # encoder initial h
        #h = self.encoder_prior.init_hidden(1) # (the very first hidden)
        inp = Variable(torch.rand(1, max_seq_len).mul(ntokens).long().cuda(), volatile=True)
        for i in range(max_seq_len):
            inp.data[0][i] = EOS_TOKEN
        for i in range(len(example)):
            inp.data[0][i] = example[i]

        seq_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1) for x in inp.data.cpu().numpy() ] ) # 1: <pad>
        inp = inp.permute(1, 0)


        ############################
        #        Encode x             #
        ############################
        '''
        encoder_hiddens_x = Variable(torch.zeros(max_seq_len, batch_size, self.encoder_prior.hidden_dim)).cuda()
        if dec_type == 'vanilla':
            for i in range(max_seq_len):
                #enc_out, h = self.encoder_prior.forward(inp[i], h, seq_lengths)
                enc_out, h = self.encoder_prior.forward(inp[i], seq_lengths, h)
                encoder_hiddens_x[i] = h[0]
        elif dec_type == 'attn':
            enc_outs = Variable(torch.zeros(max_seq_len, 1, self.encoder_prior.hidden_dim)).cuda()
            for i in range(max_seq_len):
                #enc_out, h = self.encoder_prior.forward(inp[i], h, seq_lengths)
                enc_out, h = self.encoder_prior.forward(inp[i], seq_lengths, h)
                enc_outs[i] = enc_out
                encoder_hiddens_x[i] = h[0]
            ##encoder_outputs, enc_h = self.encoder(inp, inp_lengths.tolist(), None)
        '''
        emb = self.embeddings(inp)
        emb_shift = torch.zeros_like(emb)           # 1 is the index for EOS_TOKEN
        emb_shift[1:, :, :] = emb[:-1, :, :]        # shift the input sentences
        emb_shift = self.word_drop(emb_shift)

        encoder_outputs, encoder_hidden = self.encoder_prior(emb, seq_lengths, None)
        enc_h_x_mean = encoder_outputs.mean(dim=0) # mean pool x: h_f
        # mean pool x
        #enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # h_f



        #####################################
        # perform reparam trick and get z
        #####################################
        h = encoder_hidden
        if self.rnn_type == 'LSTM':
            h = (h[0].cuda(), h[1].cuda())
        else:
            h = h.cuda()
        mu_prior = self.linear_mu_prior(enc_h_x_mean)
        log_sigma_prior = self.linear_sigma_prior(enc_h_x_mean)

        # use the mean (the most representative one)
        z = mu_prior
        he = self.linear_z(z)
        h = h[:self.decoder.n_layers].cuda()

        #####################################
        #       Decode
        #####################################
        dec_emb = emb_shift[0]
        decoder_attentions = torch.zeros(max_seq_len, max_seq_len)
        sample_type = 0
        for i in range(max_seq_len):
            if dec_type == 'vanilla':
                out, h = self.decoder.forward(dec_emb, h, None)
            elif dec_type == 'attn':
                #out, h, dec_attn = self.decoder.forward(dec_inp, h, encoder_outputs, z)
                out, h, dec_attn = self.decoder.forward(dec_emb, h, encoder_outputs, None) # decode w/o z
                padded_attn = F.pad(dec_attn.squeeze(0).squeeze(0), pad=(0, max_seq_len-dec_attn.size(2)), mode='constant', value=EOS_TOKEN)

                ##decoder_attentions[i,:] += dec_attn.squeeze(0).squeeze(0).cpu().data
                decoder_attentions[i,:] += padded_attn.cpu().data

            # 0: argmax
            if sample_type == 0:
                dec_inp = out.max(1)[1]
                dec_emb = self.embeddings(dec_inp)
                max_val, max_idx = out.data.squeeze().max(0)
                word_idx = max_idx[0]
            # 1: tempreture
            elif sample_type == 1:
                temperature = 1.0#1e-2
                word_weights = out.squeeze().data.div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]

            output_word = inputs.vocab.itos[word_idx]
            out_seq.append(output_word)

            if word_idx == EOS_TOKEN:
                break
        '''
        # create an input with the batch_size of 1
        dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]])).cuda()
        sample_type = 0
        for i in range(max_seq_len):
            if dec_type == 'vanilla':
                out, h = self.decoder.forward(dec_inp, h, z)
            elif dec_type == 'attn':
                out, h, dec_attn = self.decoder.forward(dec_inp, h, enc_outs, he.unsqueeze(0))

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



            output_word = inputs.vocab.itos[word_idx]
            out_seq.append(output_word)

            if word_idx == EOS_TOKEN:
                break
        '''
        #decoder_attentions[:i+1, :len(example)]
        return out_seq, decoder_attentions[:i+1, :len(example)-2]


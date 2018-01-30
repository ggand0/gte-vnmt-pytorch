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
	def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_layers=2, dropout=0.1, word_dropout=None, gpu=True):
		super(CustomAttnDecoderRNN, self).__init__()
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

		self.attn = Attn('concat', hidden_dim)

		self.rnn2out = nn.Linear(hidden_dim, vocab_size)
		self.drop = nn.Dropout(dropout)
		self.dropout = dropout
		self.word_dropout = word_dropout
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.rnn2out.weight.data.uniform_(-initrange, initrange)
		self.rnn2out.bias.data.fill_(0)

	def forward(self, inp, hidden, encoder_outputs, z):
		emb = self.embeddings(inp)                              # batch_size x embedding_dim
		if self.dropout:
			emb = self.drop(emb)
		output = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim

		attn_weights = self.attn(hidden[-1], encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
		context = context.transpose(0, 1) # 1 x B x N

		#out, hidden = self.rnn(output, None, hidden, context.squeeze(dim=0), z.squeeze(dim=0))            # 1 x batch_size x hidden_dim (out)
		out, hidden = self.rnn(output, None, hidden, context.squeeze(dim=0), z)            # 1 x batch_size x hidden_dim (out)
		out = self.rnn2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
		out = F.log_softmax(out)
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
		n_layers=1, dropout=0.5, word_dropout=None, gpu=True):
		super(AttnGRU_VNMT, self).__init__()

		self.word_dropout = 1.0#0.75
		self.rnn_type = rnn_type
		self.dec_type = 'attn'
		self.n_layers = n_layers

		# encoder for x
		self.encoder = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
			n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True
		)
		# encoder for y
		#self.encoder_post = EncoderRNN(rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
		#	n_layers=n_layers, dropout=dropout, word_dropout=word_dropout, gpu=True
		#)


		################################################
		#     Only supports 1-layer decoder for now
		################################################
		self.decoder = CustomAttnDecoderRNN('CustomGRU', embedding_dim, hidden_dim, vocab_size, max_seq_len,
			n_layers=1, dropout=dropout, word_dropout=word_dropout, gpu=True)


	def batchNLLLoss(self, inp, target, train=False):
		loss = 0
		batch_size, seq_len = inp.size()

		inp_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in inp.data.cpu().numpy() ] ) # 1: <pad>
		inp_lengths, perm_idx = inp_lengths.sort(0, descending=True) # SORT YOUR TENSORS BY LENGTH!
		# make sure to align the target data along with the sorted input
		inp.data = inp.data[perm_idx]
		target.data = target.data[perm_idx]
		target_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in target.data.cpu().numpy() ] ) # 1: <pad>
		inp = inp.permute(1, 0)           # seq_len x batch_size
		target = target.permute(1, 0)     # seq_len x batch_size


		############################
		#        Encode x          #
		############################
		# encode x for both the prior model and the poterior model.
		# linear layers are independent but the encoder to create annotation vectors is shared.
		enc_h_x = self.encoder.init_hidden(batch_size) # (the very first hidden)
		encoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.encoder.hidden_dim)).cuda() ## max_len x batch_size x hidden_size
		encoder_hiddens_x = Variable(torch.zeros(seq_len, self.n_layers, batch_size, self.encoder.hidden_dim)).cuda()
		for i in range(seq_len):
			#out, enc_h_x = self.encoder(inp[i], enc_h_x, inp_lengths) # enc_h_x: n_layers, batch_size, hidden_dim
			out, enc_h_x = self.encoder(inp[i], inp_lengths, enc_h_x) # enc_h_x: n_layers, batch_size, hidden_dim
			encoder_outputs[i] = out
			encoder_hiddens_x[i] = enc_h_x
		if self.rnn_type == 'LSTM':
			enc_h_x = enc_h_x[0]
		# mean pool x
		enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # h_f

		enc_h = enc_h_x
		if self.rnn_type == 'LSTM':
			dec_h = (enc_h[0][:self.decoder.n_layers].cuda(), enc_h[1][:self.decoder.n_layers].cuda())
		else:
			dec_h = enc_h[:self.decoder.n_layers].cuda()


		#########################################################
		#  Decode using the last enc_h, context vectors, and z  #
		#########################################################
		dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]*batch_size])).long().cuda()
		dec_inp = dec_inp.permute(1, 0) # 128x1
		target_length = target.size()[0]
		all_decoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.decoder.vocab_size)).cuda()

		use_target = True#True if random.random() < self.word_dropout else False
		for i in range(target_length):
			#out, dec_h = self.decoder.forward(dec_inp, dec_h, z)
			out, dec_h, attn_weights = self.decoder.forward(dec_inp, dec_h, encoder_outputs, None) # decode w/o z
			if use_target:
				dec_inp = target[i]         # shape: batch_size,
			else:
				dec_inp =  Variable(torch.LongTensor([[UNK_TOKEN]*batch_size])).long().cuda()

			all_decoder_outputs[i] = out


		# apply the objective
		loss = masked_cross_entropy( # bs x seq_len?
			all_decoder_outputs.transpose(0, 1).contiguous(),
			target.transpose(0, 1).contiguous(),
			Variable(target_lengths)
		)

		return loss


	def generate(self, inputs, ntokens, example, max_seq_len):
		"""
		Generate example
		"""

		print('Generating...')
		self.encoder.eval()
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
		h = self.encoder.init_hidden(1) # (the very first hidden)
		inp = Variable(torch.rand(1, max_seq_len).mul(ntokens).long().cuda(), volatile=True)
		for i in range(max_seq_len):
			inp.data[0][i] = EOS_TOKEN
		for i in range(len(example)):
			inp.data[0][i] = example[i]

		seq_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1) for x in inp.data.cpu().numpy() ] ) # 1: <pad>
		inp = inp.permute(1, 0)


		############################
		#         Encode x         #
		############################
		encoder_hiddens_x = Variable(torch.zeros(max_seq_len, self.n_layers, 1, self.encoder.hidden_dim)).cuda()
		if dec_type == 'vanilla':
			for i in range(max_seq_len):
				enc_out, h = self.encoder.forward(inp[i], seq_lengths, h)
				encoder_hiddens_x[i] = h
		elif dec_type == 'attn':
			enc_outs = Variable(torch.zeros(max_seq_len, 1, self.encoder.hidden_dim)).cuda()
			for i in range(max_seq_len):
				enc_out, h = self.encoder.forward(inp[i], seq_lengths, h)
				enc_outs[i] = enc_out
				encoder_hiddens_x[i] = h

		# mean pool x
		#enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # h_f



		#####################################
		# perform reparam trick and get z
		#####################################
		if self.rnn_type == 'LSTM':
			h = (h[0].cuda(), h[1].cuda())
		else:
			h = h.cuda()

		#####################################
		# perform reparam trick and get z
		#####################################
		# create an input with the batch_size of 1
		dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]])).cuda()
		decoder_attentions = torch.zeros(max_seq_len, max_seq_len)
		sample_type = 0
		for i in range(max_seq_len):
			if dec_type == 'vanilla':
				out, h = self.decoder.forward(dec_inp, h, None)
			elif dec_type == 'attn':
				#out, h, dec_attn = self.decoder.forward(dec_inp, h, enc_outs, z)
				out, h, dec_attn = self.decoder.forward(dec_inp, h, enc_outs, None) # decode w/o z
				decoder_attentions[i,:] += dec_attn.squeeze(0).squeeze(0).cpu().data

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
				#print(EOS_TOKEN)
				break

		#print(out_seq)
		#print('testtest')
		return out_seq, decoder_attentions[:i+1, :len(example)]



class VRAE_VNMT(nn.Module):
	def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, max_seq_len,
		n_layers=1, dropout=0.5, word_dropout=None, gpu=True):
		super(VRAE_VNMT, self).__init__()

		self.word_dropout = 1.0#0.75
		self.z_size = 1000 # concat size is absorbed by linear_mu_post etc, so z_size just needs to be equal with hidden_dim
		self.mode = 'vnmt'
		self.hidden_dim = hidden_dim

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
			n_layers=1, dropout=dropout, word_dropout=word_dropout, gpu=True) # > We use a fixed word dropout rate of 75%

		# for projecting z into the hidden dim of the decoder so that it can be added inside the GRU cells
		self.linear_z = nn.Linear(self.z_size, self.decoder.hidden_dim) # W_z^(2) and b_z^(2)

		self.init_weights(None)


	def init_weights(self, initrange):
		if initrange is not None:
			self.linear_mu_prior.weight.data.uniform_(-initrange, initrange)
			self.linear_mu_prior.bias.data.fill_(0)
			self.linear_sigma_prior.weight.data.uniform_(-initrange, initrange)
			self.linear_sigma_prior.bias.data.fill_(0)
			self.linear_mu_post.weight.data.uniform_(-initrange, initrange)
			self.linear_mu_post.bias.data.fill_(0)
			self.linear_sigma_post.weight.data.uniform_(-initrange, initrange)
			self.linear_sigma_post.bias.data.fill_(0)
			self.linear_z.weight.data.uniform_(-initrange, initrange)
			self.linear_z.bias.data.fill_(0)


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

		######
		# Be careful with the dimension when taking the sum!!!
		#####
		total_KLD += 1.0 * torch.sum(KLD, 1).mean().squeeze()

		return loss, total_KLD


	def batchNLLLoss(self, inp, target, train=False):
		loss = 0
		batch_size, seq_len = inp.size()

		# pack sentence data
		inp_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in inp.data.cpu().numpy() ] ) # 1: <pad>
		inp_lengths, perm_idx = inp_lengths.sort(0, descending=True) # SORT YOUR TENSORS BY LENGTH!
		# make sure to align the target data along with the sorted input
		inp.data = inp.data[perm_idx]
		target.data = target.data[perm_idx]
		target_lengths = torch.cuda.LongTensor( [ len(x)-list(x).count(1)+1 for x in target.data.cpu().numpy() ] ) # 1: <pad>
		inp = inp.permute(1, 0)           # seq_len x batch_size
		target = target.permute(1, 0)     # seq_len x batch_size


		############################
		#     Encode x and y       #
		############################
		# encode x for both the prior model and the poterior model.
		# linear layers are independent but the encoder to create annotation vectors is shared.
		enc_h_x = self.encoder_prior.init_hidden(batch_size) # (the very first hidden)
		encoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.encoder_prior.hidden_dim)).cuda() ## max_len x batch_size x hidden_size
		encoder_hiddens_x = Variable(torch.zeros(seq_len, batch_size, self.encoder_prior.hidden_dim)).cuda()
		for i in range(seq_len):
			out, enc_h_x = self.encoder_prior(inp[i], inp_lengths, enc_h_x) # enc_h_x: n_layers, batch_size, hidden_dim
			encoder_outputs[i] = out

			#encoder_hiddens_x[i] = enc_h_x[0] # 1,bs,hd => bs,hd
			encoder_hiddens_x[i] = enc_h_x[-1] # 1,bs,hd => bs,hd
		# mean pool x
		enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # h_f


		# encode y for both the poterior model.
		enc_h_y = self.encoder_post.init_hidden(batch_size) # (the very first hidden)
		encoder_hiddens_y = Variable(torch.zeros(seq_len, batch_size, self.encoder_post.hidden_dim)).cuda()
		for i in range(seq_len):
			 out, enc_h_y = self.encoder_post(target[i], target_lengths, enc_h_y)
			 #encoder_hiddens_y[i] = enc_h_y[0]
			 encoder_hiddens_y[i] = enc_h_y[-1]
		# mean pool y
		enc_h_y_mean = encoder_hiddens_y.mean(dim=0) # h_e

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
		mu_post = Variable(torch.zeros(batch_size, self.z_size)).cuda()
		log_sigma_post = Variable(torch.zeros(batch_size, self.z_size)).cuda()


		#if train:
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
		dec_h = enc_h_x[:self.decoder.n_layers].cuda()


		########################################################
		#  Decode using the last enc_h, context vectors, and z
		########################################################
		dec_inp = Variable(torch.LongTensor([[SOS_TOKEN]*batch_size])).long().cuda()
		dec_inp = dec_inp.permute(1, 0) # 128x1
		target_length = target.size()[0]
		all_decoder_outputs = Variable(torch.zeros(seq_len, batch_size, self.decoder.vocab_size)).cuda()

		use_target = True#True if random.random() < self.word_dropout else False
		for i in range(target_length):
			#out, dec_h = self.decoder.forward(dec_inp, dec_h, z)
			#out, dec_h, dec_attn = self.decoder.forward(dec_inp, dec_h, encoder_outputs, he)
			out, dec_h, dec_attn = self.decoder.forward(dec_inp, dec_h, encoder_outputs, he.unsqueeze(0))
			if use_target:
				dec_inp = target[i]         # shape: batch_size,
			else:
				dec_inp =  Variable(torch.LongTensor([[UNK_TOKEN]*batch_size])).long().cuda()

			all_decoder_outputs[i] = out


		# apply the objective
		loss = self.vnmt_loss(all_decoder_outputs, target, mu_prior, log_sigma_prior, mu_post, log_sigma_post)

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
		h = self.encoder_prior.init_hidden(1) # (the very first hidden)
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

		# mean pool x
		enc_h_x_mean = encoder_hiddens_x.mean(dim=0) # h_f



		#####################################
		# perform reparam trick and get z
		#####################################
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


		return out_seq


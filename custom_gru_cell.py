# ref: https://github.com/pytorch/pytorch/blob/6d2e39559aa3e9aff9f81d42da5739b165e73c3e/torch/nn/_functions/rnn.py

import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

# ref: https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
class CustomGRU(nn.Module):

	"""A module that runs multiple steps of CustomGRU."""

	def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
				 use_bias=True, batch_first=False, dropout=0, **kwargs):
		super(CustomGRU, self).__init__()

		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_bias = use_bias
		self.batch_first = batch_first
		self.dropout = dropout

		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size
			cell = cell_class(input_size=layer_input_size,
							  hidden_size=hidden_size,
							  **kwargs)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx, cj, he):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			if isinstance(cell, CustomGRUCell):
				h_next = cell(input_=input_[time], hx=hx, cj=cj, he=he)
			else:
				# vanilla GRU?
				#h_next, c_next = cell(input_=input_[time], hx=hx)
				h_next = cell(input_=input_[time], hx=hx)
			mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hx[0]*(1 - mask)
			output.append(h_next)
			hx = h_next
		output = torch.stack(output, 0)
		return output, hx

	def forward(self, input_, length=None, hx=None, cj=None, he=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)

		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				device = input_.get_device()
				length = length.cuda(device)

		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
			hx = (hx, hx)
		h_n = []
		#c_n = []
		layer_output = None
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			if he is not None:
				layer_output, layer_h_n = CustomGRU._forward_rnn(cell=cell, input_=input_, length=length, hx=hx[layer], cj=cj, he=he[layer])
				#layer_output, layer_h_n = CustomGRU._forward_rnn(cell=cell, input_=input_, length=length, hx=hx[layer], cj=cj, he=he)
			else:
				layer_output, layer_h_n = CustomGRU._forward_rnn(cell=cell, input_=input_, length=length, hx=hx[layer], cj=cj, he=None)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
			#c_n.append(layer_c_n)
		output = layer_output
		h_n = torch.stack(h_n, 0)
		#c_n = torch.stack(c_n, 0)
		return output, h_n



class CustomGRUCell(nn.Module):

	"""A custom GRU cell for VNMT-type architecture."""

	def __init__(self, input_size, hidden_size, bias=True):
		super(CustomGRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bias = bias
		self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
		self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
		# added
		self.w_ch = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
		self.w_zh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
		if bias:
			self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
			self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
			# added
			self.b_ch = nn.Parameter(torch.Tensor(3 * hidden_size))
			self.b_zh = nn.Parameter(torch.Tensor(3 * hidden_size))
		else:
			self.register_parameter('bias_ih', None)
			self.register_parameter('bias_hh', None)
			# added
			self.register_parameter('bias_ch', None)
			self.register_parameter('bias_zh', None)
		self.reset_parameters()

	def reset_parameters(self):
		"""
		Initialize parameters following the way proposed in the paper.
		"""

		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)



	def forward(self, input_, hx, cj, he):
		"""
		Args:
			input_: A (batch, input_size) tensor containing input
				features.
			hx: A tuple (h_0, c_0), which contains the initial hidden
				and cell state, where the size of both states is
				(batch, hidden_size).
		Returns:
			h_1, c_1: Tensors containing the next hidden and cell state.
		"""

		hidden = hx

		#if input_.is_cuda:
		#    gi = F.linear(input_, self.w_ih)
		#    gh = F.linear(hidden, self.w_hh)
		#    state = fusedBackend.GRUFused.apply
		#    return state(gi, gh, hidden) if self.b_ih is None else state(gi, gh, hidden, self.b_ih, self.b_hh)

		try:
			gi = F.linear(input_, self.w_ih, self.b_ih)
			gh = F.linear(hidden, self.w_hh, self.b_hh)
			# added
			gc = F.linear(cj, self.w_ch, self.b_ch)
			
			i_r, i_i, i_n = gi.chunk(3, 1)
			h_r, h_i, h_n = gh.chunk(3, 1)
			# added
			c_r, c_i, c_n = gc.chunk(3, 1)

			if he is not None:
				gz = F.linear(he, self.w_zh, self.b_zh)
				z_r, z_i, z_n = gz.chunk(3, 1)

			if he is not None:
				resetgate = F.sigmoid(i_r + h_r + c_r + z_r)
				inputgate = F.sigmoid(i_i + h_i + c_i + z_i)
				newgate = F.tanh(i_n + resetgate * h_n + c_n + z_n)
			else:
				resetgate = F.sigmoid(i_r + h_r + c_r)
				inputgate = F.sigmoid(i_i + h_i + c_i)
				newgate = F.tanh(i_n + resetgate * h_n + c_n)

			hy = newgate + inputgate * (hidden - newgate)
		except Exception as e:
			print(e)
			print('='*50)
			print(hidden.size()) # 1,512: bs x hidden_size(H)
			print(self.w_hh.size()) # 3*H x H
			print(self.b_hh.size()) # 3*H
			print('='*50)
			print(he.size()) # 512
			print(gz.size())
			print(i_r.size())
			print(h_r.size())
			print(z_r.size())

		return hy

	def __repr__(self):
		s = '{name}({input_size}, {hidden_size})'
		return s.format(name=self.__class__.__name__, **self.__dict__)



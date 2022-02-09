import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils import weight_norm
#from sru import SRU, SRUCell

from .utils import PositionalEncoding

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

#define the model
class SequenceModule(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, init_word_features, params, paramsExt):
        super(SequenceModule, self).__init__()

        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features
        self.init_word_features = init_word_features
        """hyper part"""
        GNN_depth, inner_CNN_depth, transformer_depth, DMA_depth, k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, transformer_hidden = params
        self.GNN_depth = GNN_depth
        self.inner_CNN_depth = inner_CNN_depth
        self.transformer_depth = transformer_depth
        self.DMA_depth = DMA_depth
        self.k_head = k_head
        self.transformer_head = transformer_head
        self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.transformer_hidden = transformer_hidden

        self.paramsExt = paramsExt
        self.stack = 'cnn'

        """CNN-RNN Module"""
        #CNN parameters
        self.embed_seq = nn.Embedding(len(self.init_word_features), 20, padding_idx=0)
        self.embed_seq.weight = nn.Parameter(self.init_word_features)
        self.embed_seq.weight.requires_grad = False

        self.embed_drop = torch.nn.Dropout(0.10)

        if self.inner_CNN_depth >= 0:
            #self.cnn_dropout = torch.nn.Dropout(0.10)
            if self.stack == 'transformers':
                self.conv_first = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size,
                                            padding=(self.kernel_size - 1) // 2)
            else:
                self.conv_first = nn.Conv1d(20, self.hidden_size1, kernel_size=self.kernel_size,
                                            padding=(self.kernel_size - 1) // 2)

            self.conv_last = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size, padding=(self.kernel_size-1) //2)

            self.plain_CNN = nn.ModuleList([])
            for i in range(self.inner_CNN_depth):
                self.plain_CNN.append(nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size,  dilation=paramsExt.cnn_dilation, padding='same'))

        self.transformer_dropout = self.paramsExt.transformer_dropout
        if self.stack != '':
            assert self.transformer_depth > 0 and self.inner_CNN_depth > 0


        if self.transformer_depth > 0:
            if self.stack == 'cnn':
                self.trans_hid = nn.Linear(self.hidden_size1, self.transformer_hidden)
            else:
                self.trans_hid = nn.Linear(20, self.transformer_hidden)

            self.pos_encoder = PositionalEncoding(self.transformer_hidden, self.transformer_dropout)
            encoder_layers = TransformerEncoderLayer(self.transformer_hidden, self.transformer_head,
                                                     self.transformer_hidden * 4, self.transformer_dropout, batch_first=True)
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.transformer_depth)


            self.transformer_encode = nn.Sequential(
              self.pos_encoder,
              self.transformer_encoder
            )


            if self.paramsExt._lambda == 'transformers' or self.paramsExt._lambda == 'cnn':
                if self.paramsExt.lambdaValue is None:
                    self._lambda = nn.Parameter(torch.randn(1))
                else:
                    self._lambda = self.paramsExt.lambdaValue
            elif self.paramsExt._lambda == 'average':
                self._lambda = nn.Parameter(torch.randn(2))



    def mask_softmax(self,a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax

    def transformer_block(self, seq_mask, seq):
        src = F.gelu(self.trans_hid(seq))
        # src = src * math.sqrt(self.transformer_hidden)
        # src = self.pos_encoder(src)

        params = [src]

        output = self.transformer_encode(*params)
        return output

    def cnn_block(self, seq_mask, seq):
        seq = seq.transpose(1, 2)
        x = F.leaky_relu(self.conv_first(seq), 0.1)
        if self.paramsExt.identity == 'cnn':
            y = x
        for i in range(self.inner_CNN_depth):
            x = self.plain_CNN[i](x)
            x = F.leaky_relu(x, 0.1)

        # x = self.cnn_dropout(x)
        x = F.leaky_relu(self.conv_last(x), 0.1)
        if self.paramsExt.identity == 'cnn':
            x = x + y
        H = x.transpose(1, 2)
        return H


    def forward(self, batch_size, seq_mask, sequence):

        ebd = self.embed_seq(sequence)
        #ebd = self.embed_drop(ebd)


        blocks = {
            'transformers': [self.transformer_block, self.transformer_depth, None],
            'cnn': [self.cnn_block, self.inner_CNN_depth, None]
        }

        if self.stack == 'cnn':
            block_keys = ['cnn', 'transformers']
        else:
            block_keys = ['transformers', 'cnn']

        for block_name in block_keys:
            block_func, depth, _ = blocks[block_name]
            if depth > 0:
                block_out = block_func(seq_mask, ebd)
                if self.paramsExt._lambda == block_name:
                    block_out *= self._lambda

                if self.stack == block_name:
                    ebd = block_out
                blocks[block_name][2] = block_out

        cnn_out = blocks['cnn'][2]
        transformer_out = blocks['transformers'][2]
        output = blocks[block_keys[-1]][2]


        if self.transformer_depth > 0 and self.inner_CNN_depth > 0:
            if self.paramsExt._lambda == 'average':
                output = (cnn_out * self._lambda[0] + transformer_out * self._lambda[1]) / (self._lambda.sum())
            elif self.stack == '':
                output = cnn_out + transformer_out
        elif self.transformer_depth > 0:
            output = transformer_out
        else:
            output = cnn_out

        return output


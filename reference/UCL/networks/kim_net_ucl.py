import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv1D
from bayes_layer import BayesianLinear

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio,voc_size,weights_matrix):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[100, 100, 100]
        self.WORD_DIM = 300
        self.MAX_SENT_LEN = 240
        self.DROPOUT_PROB = 0.5
        self.CLASS_SIZE = 10

        self.embedding = torch.nn.Embedding(voc_size, self.WORD_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.embedding.weight.requires_grad = False # non trainable

        for i in range(len(self.FILTERS)):
            conv = BayesianConv1D(1, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.dropout=torch.nn.Dropout(self.DROPOUT_PROB)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(sum(self.FILTER_NUM),n))


    def forward(self, x, sample=False):
        h = self.embedding(x).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        conv_results = []
        for i in range(len(self.FILTERS)):
            conv = F.max_pool1d(F.relu(self.get_conv(i)(h)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])
            conv_results.append(conv)

        h = torch.cat(conv_results, 1)
        h = self.dropout(h)
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, relu
import numpy as np
from torch.utils import data
import utils
from torch.utils.data import DataLoader
import argparse

MAX_LEN = 11

class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention = None

    def forward(self, q, k , v, mask, training):
        residual = q
        dim_per_head = self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        #linear projection
        key = self.wk(k) #[n, step, num_head * head_dim]
        value = self.wv(v)
        query = self.wq(q)
        context = self.scaled_dot_product_attention(query, key, value, mask)
        #split by head
        query = self.split_head(query)


    def split_head(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0,2,1,3) #permute:交换维度

    def scaled_dot_product_attention(self, q, k, v, mask = None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q, k.permute(0,1,3,2)) / (torch.sqrt(dk) + 1e-8) #[n, n_head, step, step]
        if mask is not None:
            score = score.masked_fill_(mask, -np.inf)
        self.attention = softmax(score, dim=-1)
        context = torch.matmul(self.attention, v)
        context = context.permute(0,2,1,3)
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context #[n, step, model_dim]
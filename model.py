"""
@author: awaash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KAFEModel(nn.Module):

    def __init__(self, emb_size,root_emb_size, emb_dimension):
        super(KAFEModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.r_embeddings = nn.Embedding(root_emb_size, emb_dimension, sparse=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0.5] * (self.emb_size)), requires_grad=True)
        initrange = 0.1
        self.v_embeddings.weight.data.uniform_(-0, 0)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.r_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.requires_grad = True
        self.v_embeddings.weight.requires_grad = True
        self.r_embeddings.weight.requires_grad = True

    def forward(self, pos_u, pos_v, neg_v, pos_r):
        batch_size = len(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        self.alpha.data = torch.clamp(self.alpha.data, max=0.99, min=0.01)
        emb_u = self.u_embeddings(pos_u) * self.alpha[pos_u].view(batch_size, 1) \
                + self.r_embeddings(pos_r) * (1 - self.alpha[pos_u].view(batch_size, 1))
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

# -----------------------------------------------------------------------------------------------------------------

class AdvModel(nn.Module):

    def __init__(self, emb_dimension, output_dim):
        super(AdvModel, self).__init__()
        self.adv_hidden0 = nn.Linear(emb_dimension, 1)
        #self.adv_hidden1 = nn.Linear(64, output_dim)
        self.adv_hidden0.weight.data.uniform_(-0.1, 0.1)
        #self.adv_hidden1.weight.data.uniform_(-0.1, 0.1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        #return self.adv_hidden1(self.relu(self.adv_hidden0(x)))
       return self.adv_hidden0(x)

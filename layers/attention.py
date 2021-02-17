import math
import torch
import torch.nn as nn


class NoQueryAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, hidden_dim, num_heads, q_len):
        super(NoQueryAttention, self).__init__(embed_dim + hidden_dim, num_heads)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim + hidden_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        q = q.permute([1, 0, 2])
        k = k.permute([1, 0, 2])
        return super(NoQueryAttention, self).forward(q, k, k)

from layers.dynamic_lstm import DynamicLSTM
from layers.attention import NoQueryAttention
from layers.batch_alignment import BatchAlignment
import torch
import torch.nn as nn


class NoQueryAttnLstm(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(NoQueryAttnLstm, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.align_batch = BatchAlignment()
        self.lstm = DynamicLSTM(opt['embed_dim'] * 2, opt['hidden_dim'], num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt['embed_dim'], opt['hidden_dim'], 5, 1)
        self.dense = nn.Linear(opt['hidden_dim'], opt['polarities_dim'])

    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.align_batch(x, x_len)

        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        return out

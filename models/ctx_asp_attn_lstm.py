from layers.dynamic_lstm import DynamicLSTM
import torch
import torch.nn as nn


def prepare_for_attention(q, k):
    q = torch.unsqueeze(q, dim=1)
    q = q.permute([1, 0, 2])
    k = k.permute([1, 0, 2])
    return q, k, k


class CtxAspAttnLstm(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CtxAspAttnLstm, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt['embed_dim'], opt['hidden_dim'], num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt['embed_dim'], opt['hidden_dim'], num_layers=1, batch_first=True)
        self.attention_aspect = nn.MultiheadAttention(embed_dim=opt['hidden_dim'], num_heads=5)
        self.attention_context = nn.MultiheadAttention(embed_dim=opt['hidden_dim'], num_heads=5)
        self.dense = nn.Linear(opt['hidden_dim']*2, opt['polarities_dim'])

    def forward(self, inputs):
        context_indexes, aspect_indexes = inputs[0], inputs[1]
        context_len = torch.sum(context_indexes != 0, dim=-1)
        aspect_len = torch.sum(aspect_indexes != 0, dim=-1)

        context = self.embed(context_indexes)
        aspect = self.embed(aspect_indexes)

        context, (_, _) = self.lstm_context(context, context_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        context_len = torch.tensor(context_len, dtype=torch.float)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, context_len.view(context_len.size(0), 1))

        query, key, value = prepare_for_attention(context_pool, aspect)
        aspect_final, _ = self.attention_aspect(query, key, value)
        aspect_final = aspect_final.squeeze(dim=0)

        query, key, value = prepare_for_attention(aspect_pool, context)
        context_final, _ = self.attention_context(query, key, value)
        context_final = context_final.squeeze(dim=0)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out


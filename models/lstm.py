from layers.dynamic_lstm import DynamicLSTM
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt['embed_dim'], opt['hidden_dim'], num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt['hidden_dim'], opt['polarities_dim'])

    def forward(self, inputs):
        text_raw_indexes = inputs[0]
        x = self.embed(text_raw_indexes)
        x_len = torch.sum(text_raw_indexes != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out

import torch
import torch.nn as nn


class BertBased(nn.Module):
    def __init__(self, bert, opt):
        super(BertBased, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt['dropout'])
        self.dense = nn.Linear(opt['bert_dim'], opt['polarities_dim'])

    def forward(self, inputs):
        text_bert_indexes, bert_segments_ids = inputs[0], inputs[1]
        bert_output = self.bert(text_bert_indexes, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(bert_output.pooler_output)
        out = self.dense(pooled_output)
        return out

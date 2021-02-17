import logging
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformers import BertModel

from models import LSTM, CtxAspAttnLstm, NoQueryAttnLstm, BertBased

from torch.optim import Adam
from utils.data_utils import build_tokenizer, build_embedding_matrix, SADataset, Tokenizer4Bert
from sklearn import metrics

from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter

train_file = './datasets/train.raw'
test_file = './datasets/test.raw'


logging.basicConfig(level=logging.INFO)


class TrainTask:
    def __init__(self, opt):
        self.opt = opt
        self.summary_writer = SummaryWriter(comment=opt['model_name'])

        if 'bert' == opt['model_name']:
            tokenizer = Tokenizer4Bert(opt['max_seq_len'], opt['pretrained_bert_name'])
            bert = BertModel.from_pretrained(opt['pretrained_bert_name'])
            self.model = opt['model_class'](bert, opt)
        else:
            tokenizer = build_tokenizer(fnames=[train_file, test_file],
                                        max_seq_len=85,
                                        dat_fname='tokenizer.dat')
            embedding_matrix = build_embedding_matrix(word2idx=tokenizer.word2idx,
                                                      embed_dim=opt['embed_dim'],
                                                      dat_fname='{0}_embedding_matrix.dat'.format(
                                                          str(opt['embed_dim'])))
            self.model = opt['model_class'](embedding_matrix, opt)

        self.trainset = SADataset(train_file, tokenizer)
        self.testset = SADataset(test_file, tokenizer)
        self.valset = self.testset

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt['num_epoch']):
            logging.info('>' * 100)
            logging.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col] for col in self.opt['input_columns']]
                outputs = self.model(inputs)
                targets = batch['polarity']

                # if i_epoch == 0 and i_batch == 0:
                    # self.summary_writer.add_graph(self.model, input_to_model=inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt['log_step'] == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logging.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            train_acc, train_f1 = self._evaluate_acc_f1(train_data_loader)
            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)

            self.summary_writer.add_scalar("Loss/train", train_acc, i_epoch + 1)
            self.summary_writer.add_scalar("F1/train", train_f1, i_epoch + 1)

            self.summary_writer.add_scalar("Loss/test", val_acc, i_epoch + 1)
            self.summary_writer.add_scalar("F1/test", val_f1, i_epoch + 1)

            logging.info('Accuracy: {}. F1: {}'.format(val_acc, val_f1))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc_{1}'.format(self.opt['model_name'], round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logging.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col] for col in self.opt['input_columns']]
                t_targets = t_batch['polarity']
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.00005)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt['batch_size'], shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt['batch_size'], shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt['batch_size'], shuffle=False)

        self._reset_params()
        best_model_path = self.train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logging.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    models = {
        'lstm': LSTM,
        'no_query_attn_lstm': NoQueryAttnLstm,
        'ctx_asp_attn_lstm': CtxAspAttnLstm,
        'bert': BertBased
    }

    input_columns = {
        'lstm': ['text_indexes'],
        'no_query_attn_lstm': ['text_indexes', 'context_indexes'],
        'ctx_asp_attn_lstm': ['text_indexes', 'context_indexes'],
        'bert': ['concat_bert_indexes', 'concat_segments_indexes']
    }

    models_to_evaluate = ['lstm', 'no_query_attn_lstm', 'ctx_asp_attn_lstm', 'bert']

    for model in models_to_evaluate:
        opt = {
            'model_name': model,
            'model_class': models[model],
            'input_columns': input_columns[model],
            'hidden_dim': 300,
            'embed_dim': 100,
            'num_epoch': 20,
            'batch_size': 16,
            'log_step': 10,
            'polarities_dim': 3,
            'max_seq_len': 85,
            'pretrained_bert_name': 'bert-base-uncased',
            'dropout': 0.1,
            'bert_dim': 768
        }
        train_task = TrainTask(opt)
        train_task.run()


if __name__ == '__main__':
    main()

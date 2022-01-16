from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from Models.Linear import Linear


class BertLSTM(BertPreTrainedModel):

    def __init__(self, config, num_labels, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertLSTM, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,bidirectional=bidirectional, batch_first=True, dropout=dropout)
        # self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_dp',nn.Dropout(p=0.5))
        self.classifier.add_module('c_rnn', nn.Linear(rnn_hidden_size * 2, num_labels))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(rnn_hidden_size * 2, 64))
        # self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(0.2, inplace=True))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(64, 2))

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alpha=0):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)
        reversed_hidden = ReverseLayerF.apply(hidden, alpha)
        domain_logits = self.domain_classifier(reversed_hidden)

        return logits,domain_logits

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn

class BertCNN(BertPreTrainedModel):

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        super(BertCNN, self).__init__(config)
        self.num_labels = num_labels
        self.n_filter = n_filters
        self.filter_sizes = filter_sizes

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, out_channels=100, kernel_size=3)
        self.maxP1 = nn.MaxPool1d(kernel_size=510)

        self.fc = nn.Linear(in_features=100, out_features=nums_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]

        encoded_layers = self.dropout(encoded_layers)

        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim=768, seq_len]

        x = self.conv1(encoded_layers)
        x = self.maxP1(x)
        x = x.view(-1, x.size(1))
        logits = self.fc(x)

        return logits

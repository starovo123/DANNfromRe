
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

from torch import nn
from torch.nn import CrossEntropyLoss

class BertOrigin(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertOrigin, self).__init__(config)    # 继承了BertPretrainedModel里面，self.config=config
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        '''

        :param input_ids: 词对应的id
        :param token_type_ids: 句子类型id，0为第一句，1为第二句
        :param attention_mask: 区分padding与token，0是padding，1是token
        :param labels: 类别标签
        :return:
                返回？
        '''

        # pooled_output：bert输出的向量
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # 将bert输出的向量传入隐藏层中，通过dropout机制防止过拟合；Linear层是全连接，在深度学习里用logits表示全连接层的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


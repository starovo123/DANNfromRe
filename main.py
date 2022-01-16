import random
import numpy as np
import os

import torch
import torch.nn as nn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from Utils.utils import get_device, set_seed
from Utils.load_datasets import load_data

from train import train, evaluate

def main(config, model_times, label_list):

    if not os.path.exists(config.output_dir + model_times):
        os.makedirs(config.output_dir + model_times)

    if not os.path.exists(config.cache_dir + model_times):
        os.makedirs(config.cache_dir + model_times)

    # bert模型输出文件
    output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)
    output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

    ''' 设备检测与准备 '''
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    ''' 设定随机种子 '''
    config.n_gpu = n_gpu
    set_seed(config)

    ''' 导入分词器 '''
    tokenizer = BertTokenizer.from_pretrained(config.bert_vocab_file, do_lower_case=config.do_lower_case)
    num_labels = len(label_list)


    if config.do_train:

        ''' 数据导入 '''
        train_dataloader, train_examples_len = load_data(config.data_dir, tokenizer, config.max_seq_length, config.train_batch_size, "train", label_list)
        dev_dataloader, _ = load_data(config.data_dir, tokenizer, config.max_seq_length, config.dev_batch_size, "dev", label_list)
        print("length of dev_dataloader:",len(dev_dataloader))
        num_train_optimization_steps = int(train_examples_len/config.train_batch_size/config.gradient_accumulation_steps) * config.num_train_epochs


        ''' 模型准备 '''
        print("model name is {}".format(config.model_name))
        if config.model_name == "BertOrigin":
            from Models.BertOrigin.model import BertOrigin
            model = BertOrigin.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        elif config.model_name == "BertCNN":
            from Models.BertCNN.model import BertCNN
            model = BertCNN.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels,                                   n_filters=config.filter_num, filter_sizes=config.filter_sizes)
        elif config.model_name == 'BertLSTM':
            from Models.BertLSTM.model import BertLSTM
            model = BertLSTM.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels,
            rnn_hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=config.bidirectional,
            dropout=config.dropout)


        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)


        ''' 优化器准备 '''
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
            {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion, t_total=num_train_optimization_steps)


        ''' 损失函数准备 '''
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)


        ''' 开始训练 '''
        log_time = train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer, criterion, device, label_list, output_model_file, output_config_file, config.log_dir, config.print_step, config.early_stop, config.gradient_accumulation_steps)

        config.log_time = log_time

        ''' 开始测试 '''
        # test_dataloader, _ = load_data(config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test", label_list)
        #
        # bert_config = BertConfig(output_config_file)
        #
        # model = BertOrigin(bert_config, num_labels=num_labels)
        #
        # model.load_state_dict(torch.load(output_model_file))
        # model.to(device)
        #
        # criterion = nn.CrossEntropyLoss()
        # criterion = criterion.to(device)
        #
        # test_loss, test_acc, test_report, test_auc = evaluate(model, test_dataloader, criterion, device, label_list)
        # print('--------Test--------')
        # print(f'\t Loss:{test_loss: .3f} | Acc:{test_acc*100: .3f} %')

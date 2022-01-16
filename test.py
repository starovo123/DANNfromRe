
import os
import time

import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig,WEIGHTS_NAME,CONFIG_NAME


from Utils.load_datasets import load_data
from Utils.utils import get_device, write_log

from train import evaluate


def test(config, model_times, label_list):

    output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)
    output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

    if config.log_time != "":
        r_time = config.log_time
    else:
        r_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if not os.path.exists(config.log_dir + r_time):
        os.makedirs(config.log_dir + r_time)
    log_file = config.log_dir + r_time + '/result.txt'

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    tokenizer = BertTokenizer.from_pretrained(config.bert_vocab_file, do_lower_case=config.do_lower_case)
    num_labels = len(label_list)

    test_dataloader, _ = load_data(config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test",
                                   label_list)

    bert_config = BertConfig(output_config_file)

    if config.model_name == "BertOrigin":
        from Models.BertOrigin.model import BertOrigin
        model = BertOrigin(bert_config, num_labels=num_labels)
    elif config.model_name == "BertCNN":
        from Models.BertCNN.model import BertCNN
        model = BertCNN.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels,n_filters=config.filter_num, filter_sizes=filter_sizes)
    elif config.model_name == 'BertLSTM':
        from Models.BertLSTM.model import BertLSTM
        model = BertLSTM.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels,rnn_hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=config.bidirectional,dropout=config.dropout)

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    test_loss, test_acc, test_report, test_auc = evaluate(model, test_dataloader, criterion, device, label_list)

    write_log(log_file, "--------Test--------")
    write_log(log_file, f'\t Loss:{test_loss: .3f} | Acc:{test_acc * 100: .3f} %')
    print('--------Test--------')
    print(f'\t Loss:{test_loss: .3f} | Acc:{test_acc*100: .3f} %')

    write_log(log_file, '\n')
    write_log(log_file, "---------Config-------")
    write_log(log_file, str(config))

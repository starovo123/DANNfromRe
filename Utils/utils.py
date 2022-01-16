
from sklearn import metrics
import random
import numpy as np

import torch

def get_device(gpu_id):
    device = torch.device("cuda:"+ str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins)*60)
    return elapsed_mins, elapsed_secs

def classification_metric(preds, labels, label_list):
    ''' 分类任务的评价指标，传入的数据需要是numpy类型 '''
    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)

    if len(label_list) > 2:
        auc = 0.5
    else:
        auc = metrics.roc_auc_score(labels, preds)
    return acc, report, auc

def write_log(log_file, s):
    with open(log_file, 'a') as f:
        f.write(s+'\n')

def splitItem(item):
    tran_item = item.transpose(0, 1)
    s_item = tran_item[:512]
    t_item = tran_item[512:]
    s_item = s_item.transpose(0, 1)
    t_item = t_item.transpose(0, 1)
    return s_item, t_item
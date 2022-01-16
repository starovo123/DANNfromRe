import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from Utils.utils import classification_metric, write_log, splitItem

def train(epoch_num, n_gpu, model, train_dataloader, dev_dataloader, optimizer, criterion, device, label_list, output_model_file, output_config_file, log_dir, print_step, early_stop, gradient_accumulation_steps):
    '''
    :param epoch_num:
    :param n_gpu:
    :param model:
    :param train_dataloader:
    :param dev_dataloader:
    :param optimizer:
    :param criterion: 损失函数定义
    :param device:
    :param label_list:
    :param output_model_file: 保存bert模型
    :param output_config_file:
    :param log_dir: tensorboard读取的日志目录，用于后续分析
    :param print_step: 多少步保存一次模型
    :param early_stop:
    :return:
            不用返回任何东西，保存模型的参数即可
    '''

    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_file = log_dir + log_time + '/log.txt'

    if not os.path.exists(log_dir + log_time):
        os.makedirs(log_dir + log_time)

    early_stop_times = 0
    best_acc = 0
    global_step = 0

    for epoch in range(int(epoch_num)):

        if early_stop_times >= early_stop:
            break

        epoch_str = f'-------------------Epoch: {epoch + 1:02} ---------------------'
        write_log(log_file, epoch_str)
        print(epoch_str)

        epoch_loss = 0
        train_steps = 0

        all_preds_s = np.array([],dtype=int)
        all_labels_s = np.array([], dtype=int)
        all_preds_t = np.array([], dtype=int)
        all_labels_t = np.array([], dtype=int)

        for step, batch in enumerate(train_dataloader):

            model.train()

            p = float(step + epoch * len_dataloader) / epoch_num / len_dataloader
            alpha = 2. / (1 + np.exp(-10 * p)) - 1

            batch = tuple(t.to(device) for t in batch)  # 1.将batch变成能导入gpu的形式；2.整个batch转成tuple
            _, input_ids, input_mask, segment_ids, label_ids = batch

            s_input_ids, t_input_ids = splitItem(input_ids)
            s_input_mask, t_input_mask = splitItem(input_mask)
            s_segment_ids, t_segment_ids = splitItem(segment_ids)

            batch_size = len(label_ids)
            domain_label_s = torch.full([batch_size], 0).long().to(device)
            domain_label_t = torch.full([batch_size], 1).long().to(device)

            logits_class, domain_logits_s = model(s_input_ids, s_segment_ids, s_input_mask, labels=None,alpha=alpha)     # 全连接输出
            logits_t, domain_logits_t = model(t_input_ids, t_segment_ids, t_input_mask, labels=None,alpha=alpha)

            loss_class = F.cross_entropy(logits_class.view(-1, len(label_list)), label_ids.view(-1))
            loss_domain_s = F.cross_entropy(domain_logits_s.view(-1, 2), domain_label_s.view(-1))
            loss_domain_t = F.cross_entropy(domain_logits_t.view(-1, 2), domain_label_t.view(-1))


            ''' 修正loss '''
            if n_gpu > 1:
                loss_class = loss_class.mean()
                loss_domain_s = loss_domain_s.mean()
                loss_domain_t = loss_domain_t.mean()
            loss = loss_class.item() + loss_domain_s.item()+ loss_domain_t.item()
            if gradient_accumulation_steps > 1:
                loss = loss/gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss   # epoch的loss为所有batch_loss之和


            preds_s = logits_class.detach().cpu().numpy()
            outputs_s = np.argmax(preds_s, axis=1)
            all_preds_s = np.append(all_preds_s, outputs_s)   # 将outputs的值append到all_preds中

            preds_t = logits_t.detach().cpu().numpy()
            outputs_t = np.argmax(preds_t, axis=1)
            all_preds_t = np.append(all_preds_t, outputs_t)

            label_ids = label_ids.detach().cpu().numpy()
            all_labels_s = np.append(all_labels_s, label_ids)
            all_labels_t = np.append(all_labels_t, label_ids)

            train_steps += 1
            global_step += 1

            if global_step % gradient_accumulation_steps == 0:
                ''' 参数更新 '''
                optimizer.step()    # （根据梯度）更新参数
                optimizer.zero_grad()   # 将梯度初始化为0（因为一个batch的 [loss关于weight的导数] 是所有sample的[loss关于weight的导数]的累加和）

                if global_step % print_step == 0:

                    ''' 输出loss和acc '''
                    train_loss = epoch_loss / train_steps
                    train_acc, train_report, train_auc = classification_metric(all_preds, all_labels, label_list)
                    dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, device, label_list)

                    loss_result = f'Iter:{train_steps}/{len(train_dataloader)} || train Loss:{train_loss: .3f}   Acc:{train_acc*100: .3f}%  |  dev Loss:{dev_loss: .3f}   Acc:{dev_acc*100: .3f}%'
                    write_log(log_file, loss_result)
                    print(loss_result)

                    # for label in label_list:
                    #     label_f1 = label+":" + "f1/train", str(train_report[label]['f1-score']) +' '+ "f1/dev", str(dev_report[label]['f1-score'])
                    #     write_log(log_file, label_f1)
                    #
                    # print_list = ['macro avg', 'weighted avg']
                    # for label in print_list:
                    #     print_f1 = label+":"+"f1/train", str(train_report[label]['f1-score']) +'  '+"f1/dev", str(dev_report[label]['f1-score'])
                    #     write_log(log_file, print_f1)


                    if dev_acc > best_acc:

                        best_acc = dev_acc

                        model_to_save = model.module if hasattr(model,'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        early_stop_times = 0

                    else:
                        early_stop_times += 1

    return log_time


def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([],dtype=int)
    all_labels = np.array([],dtype=int)

    epoch_loss = 0

    for batch in dataloader:

        batch = tuple(t.to(device) for t in batch)
        _, input_ids, input_mask, segment_ids, label_ids = batch

        _, t_input_ids = splitItem(input_ids)
        _, t_input_mask = splitItem(input_mask)
        _, t_segment_ids = splitItem(segment_ids)

        with torch.no_grad():
            logits = model(t_input_ids, t_segment_ids, t_input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.detach().cpu().numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report, auc = classification_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc


# def evaluate_save(model, dataloader, criterion, device, label_list):
#
#     model.eval()
#
#     all_preds = np.array([], dtype=int)
#     all_labels = np.array([], dtype=int)
#     all_idx = np.array([], dtype=int)
#
#     epoch_loss = 0
#
#     for batch in dataloader:
#         batch = [d.to(device) for d in batch]
#         idx, input_ids, input_mask, segment_ids, label_ids = batch
#
#         with torch.no_grad():
#             logits = model(input_ids, segment_ids, input_mask, labels=None)
#         loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))
#
#         preds = logits.detach().cpu().numpy()
#         outputs = np.argmax(preds, axis=1)
#         all_preds = np.append(all_preds, outputs)
#
#         label_ids = label_ids.to("cpu").numpy()
#         all_labels = np.append(all_labels, label_ids)
#
#         idxs = idxs.detach().cpu().numpy()
#         all_idx = np.append(all_idx, idxs)
#
#         epoch_loss += loss.mean().item()
#
#     acc, report, auc = classification_metric(all_preds, all_labels, label_list)
#     return epoch_loss/len(dataloader), acc, report, auc, all_idx, all_labels, all_preds
#
#

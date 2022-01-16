from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        '''

        :param guid: 例子的唯一编号
        :param text_a: string；
        :param text_b: (Optional string)；
        :param label: (Optional string)
        '''

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeature(object):

    def __init__(self, idx, input_ids, input_mask, segment_ids, label_id):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):     # 截断句子a、b的tokens，小于max_length

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(examples, label_list, max_seq_length, tokenizer):
    '''
    :param examples: 样本集合
    :param label_list:
    :param max_seq_length:
    :param tokenizer: 分词器
    :return:
            features: 表示样本转化后的特征
    '''

    label_map = {label: i for i,label in enumerate(label_list)}

    feature = []

    for ex_index, example in enumerate(tqdm(examples,desc='writing example')):

        tokens_a = tokenizer.tokenize(example.text_a)   #分词
        tokens_b = None
        if example.text_b:
            tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-4)    # 减去[CLS],[SEP]x2
        else:
            if len(tokens_a) > max_seq_length-2:
                tokens_a = tokens_a[:(max_seq_length-2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)     # 句子标识，0表示第一个句子，1表示第二个句子

        if tokens_b:
            tokens += ["[CLS]"] + tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b)+1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # input_mask：1表示真正的tokens，0表示padding tokens
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        try:
            label_id = label_map[example.label]
        except:
            print(example.label)

        idx = int(example.guid)

        feature.append(InputFeature(idx=idx,
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_id=label_id))
    return feature


def convert_feature_to_tensor(features, batch_size, data_type):
    ''' 将features转化为tensor，并塞入迭代器
    :param features:
    :param batch_size:
    :param data_type:
    :return:
            dataloader:以batch_size为基础的迭代器
    '''

    all_idx_ids = torch.tensor([f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_idx_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_id)

    sampler = RandomSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader
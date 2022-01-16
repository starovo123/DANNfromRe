import os
import csv

from Utils.Classifier_utils import InputExample, convert_example_to_feature, convert_feature_to_tensor

def read_tsv(filename):
    csv.field_size_limit(500 * 1024 * 1024)
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")      # delimiter 用于指定分割字段的字符，此处为换行符
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def load_tsv_datasets(filename, set_type):
    examples = []
    lines = read_tsv(filename)

    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = line[0]
        text_b = line[1]
        label = line[2]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

def load_data(data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
    if format_type == 0:
        load_func = load_tsv_datasets

    if data_type == "train":
        train_file = os.path.join(data_dir,'train.tsv')
        examples = load_func(train_file, data_type)
    elif data_type == "dev":
        dev_file = os.path.join(data_dir,'dev.tsv')
        examples = load_func(dev_file, data_type)
    elif data_type == "test":
        test_file = os.path.join(data_dir, 'test.tsv')
        examples = load_func(test_file, data_type)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_example_to_feature(examples, label_list, max_length, tokenizer)
    dataloader = convert_feature_to_tensor(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len










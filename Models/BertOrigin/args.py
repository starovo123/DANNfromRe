
import argparse

def get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir, gpu_ids, train_bs, epoch_nums):

    parser = argparse.ArgumentParser(description='BERT_Baseline')

    parser.add_argument("--model_name", default="BertOrigin", type=str, help="the name of model")

    parser.add_argument("--save_name", default="BertOrigin", type=str, help="the name file of model")

    parser.add_argument("--log_time", default="", type=str)

    # 文件路径
    parser.add_argument("--data_dir",
                        default=data_dir,
                        type=str,
                        help="data directory; should contain .tsv file")

    parser.add_argument("--output_dir",
                        default=output_dir + "BertOrigin/",
                        type=str,
                        help="model predictions and checkpoints will be writtten.")

    parser.add_argument("--cache_dir",
                        default=cache_dir + "BertOrigin/",
                        type=str,
                        help="模型缓存目录")

    parser.add_argument("--log_dir",
                        default=log_dir + "BertOrigin/",
                        type=str,
                        help="日志目录，用于tensorboard分析")

    parser.add_argument("--bert_vocab_file",
                        default=bert_vocab_file,
                        type=str)

    parser.add_argument("--bert_model_dir",
                        default=bert_model_dir,
                        type=str)

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="随机种子")


    # 文件预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="set this flag if you are using uncased model")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="分词后的最大字符数，大于此将被截断，小于此则被填充")


    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=train_bs,
                        type=int)

    parser.add_argument("--dev_batch_size",
                        default=16,
                        type=int)

    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int)

    parser.add_argument("--do_train",
                        action='store_true')

    parser.add_argument("--num_train_epochs",
                        default=epoch_nums,
                        type=float)

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)


    # optimizer参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float)


    # 梯度累积
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--print_step",
                        type=int,
                        default=10)

    parser.add_argument("--early_stop",
                        type=int,
                        default=10)

    parser.add_argument("--gpu_ids",
                        type=str,
                        default=gpu_ids)


    config = parser.parse_args()

    return config
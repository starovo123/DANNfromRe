
from main import main
from test import test

import warnings


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    model_name = 'BertLSTM'
    # label_list = [u'ด้านเศรษฐกิจ', u'ด้านสังคม', u'ด้านความมั่นคง', u'ข่าวทำเนียบรัฐบาล', u'ด้านวัฒนธรรมท่องเที่ยวฯ']
    label_list = [u'นิติศาสตร์', u'วิศวกรรมไฟฟ้า', u'วิศวกรรมอุตสาหการ', u'วิจัยการศึกษา', u'Chemical Engineering', u'การบริหารการพยาบาล', u'นิเทศศาสตรพัฒนาการ', u'มัธยมศึกษา', u'บริหารการศึกษา', u'โสตทัศนศึกษา']
    # label_list = ['GNM', 'SNT', 'TTL', 'BLD', 'BDS', 'CLT', 'RLW', 'LTT', 'ROD', 'SAT', 'EPR', 'FML', 'SCL', 'PNM',
    #             'HST']

    data_dir = 'D:/data/en-th-mix'
    output_dir = 'D:/Code/DANNfromRe/.output/'
    cache_dir = 'D:/Code/DANNfromRe/.cache/'
    log_dir = 'D:/Code/DANNfromRe/.log/'

    bert_vocab_file = 'D:/Code/bert-base-multilingual-cased/vocab.txt'
    bert_model_dir = 'D:/Code/bert-base-multilingual-cased'

    gpu_ids = '4,5,6,7'
    train_batch_size = 32
    epoch_nums = 30

    if model_name == "BertOrigin":
        from Models.BertOrigin import args
    elif model_name == "BertCNN":
        from Models.BertCNN import args
    elif model_name == "BertLSTM":
        from Models.BertLSTM import args
        
    config = args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir, gpu_ids, train_batch_size, epoch_nums)
    main(config, config.save_name, label_list)
    test(config, config.save_name, label_list)


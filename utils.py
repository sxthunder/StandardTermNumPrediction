import os
import json
import torch
import transformers
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertForNextSentencePrediction
from model import BertAttentionFfnAdapterForSequenceClassification, BertForMaskLMAndNSP
from sklearn.model_selection import train_test_split
from functools import reduce

from prefix_ner_bert import BertForMaskedLM

def print_args(args, logger):
    logger.info('#'*20 + 'Arguments' + '#'*20)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        logger.info('{}:{}'.format(k, v))

def get_optimizer_and_scheduler(model, t_total, lr, warmup_steps, eps=1e-6, optimizer_class=AdamW, scheduler='WarmupLinear'):
    def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01, 'lr':lr},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0, 'lr':lr},
    ]

    local_rank = -1
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer_params = {'lr': lr, 'eps': eps}
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)
    # scheduler_obj = None
    return optimizer, scheduler_obj

# 从训练数据的分布来开，个数大于11的训练数量加起来占比已经小于0.1%，故最大大于等于11的算作一个类别
def read_classification_data(data_path, logger, args):
    logger.info('label num is set to {}'.format(args.num_labels))
    def read_data(data_path):
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                text, _, num = line.strip().split('\t')
                num = min(int(num), 11) 
                data_list.append((text, num))
        return data_list
    logger.info('Reading classification data from {}'.format(data_path))
    train_data = read_data(os.path.join(data_path, 'train_num_data.txt'))
    dev_data = read_data(os.path.join(data_path, 'dev_num_data.txt'))
    test_data = read_data(os.path.join(data_path, 'test_num_data.txt'))

    logger.info('Train data size: {}'.format(len(train_data)))
    logger.info('Dev data size: {}'.format(len(dev_data)))
    logger.info('Test data size: {}'.format(len(test_data)))

    return train_data, dev_data, test_data

# 读partition的data
def read_parition_data(data_path, logger, args):
    def read_data(path):
        logger.info('Reading partition data from {}'.format(path))
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_list = line.strip().split('\t')
                line_list[0] = line_list[0].replace('（', '(').replace('）', ')')
                data_list.append(line_list)
            f.close()
        return data_list

    train_list = read_data(os.path.join(data_path, 'train', 'multi_split.txt')) + read_data(os.path.join(data_path, 'train', 'one_not_split.txt'))

    dev_list = read_data(os.path.join(data_path, 'dev', 'split_data.txt'))
    test_list = read_data(os.path.join(data_path, 'test', 'split_data.txt'))

    # num_list = [min(int(x[2]), 10) for x in data_list]

    # train_list, dev_list, _, dev_y = train_test_split(data_list, num_list, test_size=0.3, random_state=42, stratify=num_list)
    # dev_list, test_list, _, _ = train_test_split(dev_list, dev_y, test_size=0.66, random_state=42, stratify=dev_y)

    # test_list = data_list[-100:]
    # dev_list = data_list[-200:-100]
    # train_list = data_list[:-200]

    logger.info('Train data size: {}'.format(len(train_list)))
    logger.info('Dev data size: {}'.format(len(dev_list)))
    logger.info('Test data size: {}'.format(len(test_list)))
    
    return train_list, dev_list, test_list

def read_partition_cla_rank_data(data_path, args, logger):
    def read_data(path):
        logger.info('Reading partition data from {}'.format(path))
        data_list = []
        with open(path, 'r')  as f:
            for line in f:
                line_arr = line.strip().split('\t')
                data_list.append(line_arr)
            f.close()
        return data_list

    train_data_list = read_data(os.path.join(data_path, 'beam_search_train_result.txt'))
    dev_data_list = read_data(os.path.join(data_path, 'beam_search_dev_result.txt'))
    test_data_list = read_data(os.path.join(data_path, 'beam_search_test_result.txt'))

    logger.info('Train data size: {}'.format(len(train_data_list)))
    logger.info('Dev data size: {}'.format(len(dev_data_list)))
    logger.info('Test data size: {}'.format(len(test_data_list)))

    return train_data_list, dev_data_list, test_data_list


def read_partition_cla_data(data_path, args, logger):
    def read_data(path):
        logger.info('Reading partition data from {}'.format(path))
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_list = line.strip().split('\t')
                data_list.append(line_list)
            f.close()
        return data_list

    # train_data_list = read_data(os.path.join(data_path, 'best_strict_acc_predict_train_split_data.txt'))
    # dev_data_list = read_data(os.path.join(data_path, 'best_strict_acc_predict_dev_split_data.txt'))
    # test_data_list = read_data(os.path.join(data_path, 'best_strict_acc_predict_test_split_data.txt'))

    train_data_list = read_data(os.path.join(data_path, 'train.txt'))
    dev_data_list = read_data(os.path.join(data_path, 'dev.txt'))
    test_data_list = read_data(os.path.join(data_path, 'test.txt'))

    logger.info('Train data size: {}'.format(len(train_data_list)))
    logger.info('Dev data size: {}'.format(len(dev_data_list)))
    logger.info('Test data size: {}'.format(len(test_data_list)))

    logger.info('load code from path {}'.format(args.code_path))
    standard_to_code = {}
    # with open(args.code_path, 'r') as f:
    #     for line in f:
    #         line_arr = line.strip().split('\t')
    #         code, standard = line_arr
    #         code = code[0]
    #         standard_to_code[standard] = code
    #     f.close()

    with open(args.code_path, 'r') as f:
        for line in f:
            line_arr = line.strip().split('\t')
            code, standard = line_arr
            if len(code) == 1:
                code += '00'
            code = code[:3]
            standard_to_code[standard] = code
        f.close()

    if args.code_cla_metric == 'alphabet':
        for k, v in standard_to_code.items():
            standard_to_code[k] = v[0]

        code_list = sorted(set(standard_to_code.values()))
        code_to_idx = {code: i + 1 for i, code in enumerate(code_list)}
        idx_to_code = {v:k for k, v in code_to_idx.items()}
        logger.info('code to idx {}'.format(code_to_idx))

    elif args.code_cla_metric == 'icd':
        order_dict = {'I':('A00', 'B99'), 'II':('C00', 'D48'), 'III':('D50', 'D89'), 'IV':('E00', 'E90'), 'V':('F00', 'F99'), 'VI':('G00', 'G99'), 'VII':('H00', 'H59'), 'VIII':('H60', 'H95'), 'IX':('I00', 'I99'), 'X':('J00', 'J99'), 'XI':('K00', 'K93'), 'XII':('L00', 'L99'), 'XIII':('M00', 'M99'), 'XIV':('N00', 'N99'), 'XV':('O00', 'O99'), 'XVI':('P00', 'P96'), 'XVII':('Q00', 'Q99'), 'XVIII':('R00', 'R99'), 'XIX':('S00', 'T98'), 'XX':('V01', 'Y98'), 'XXI':('Z00', 'Z99'), 'XXII':('U00', 'U99')}

        code_list = sorted(set(standard_to_code.values()))
        code_to_roma = {}
        for code in code_list:
            for k, v in order_dict.items():
                if code >= v[0] and code <= v[1]:
                    code_to_roma[code] = k
                    break
        
        standard_to_code = {k: code_to_roma[v] for k, v in standard_to_code.items()}
        code_list = sorted(set(standard_to_code.values()))
        code_to_idx = {code: i + 1 for i, code in enumerate(code_list)}
        idx_to_code = {v:k for k, v in code_to_idx.items()}
        logger.info('code to idx {}'.format(code_to_idx)) 
    elif args.code_cla_metric == 'one':
        for k, v in standard_to_code.items():
            standard_to_code[k] = v[0]
        code_to_idx = {chr(i+ord('A')):1 for i in range(26)}
        idx_to_code = {v:k for k, v in code_to_idx.items()}

    else:
        raise Exception('code_cla_metric is not supported')

    return train_data_list, dev_data_list, test_data_list, standard_to_code, code_to_idx, idx_to_code





def read_fina_sim_data(data_path, logger, args):
    def read_data(data_path):
        x_list = []
        y_list = []
        with open(data_path, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                x_list.append((line['sentence1'], line['sentence2']))
                y_list.append(line['label'])
            f.close()

        return x_list, y_list

    train_x, train_y = read_data(os.path.join(data_path, 'train.json'))
    test_x, test_y = read_data(os.path.join(data_path, 'dev.json'))

    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42, stratify=train_y)

    logger.info('Train data size : {}'.format(len(train_x)))
    logger.info('Train data size : {}'.format(len(dev_x)))
    logger.info('Train data size : {}'.format(len(test_x)))
    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)

    
# 读取Bert
def load_bert(model_path, logger, args):
    logger.info('Loading BERT model from {}'.format(model_path))
    bert_config = BertConfig.from_pretrained(model_path)
    bert_config.num_labels = args.num_labels
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForSequenceClassification.from_pretrained(model_path, config=bert_config)
    return bert_model, bert_tokenizer, bert_config

# 读取nsp bert
def load_nsp_bert(model_path, logger, args):
    logger.info('Loading BERT model from {}'.format(model_path))
    bert_config = BertConfig.from_pretrained(model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForNextSentencePrediction.from_pretrained(model_path, config=bert_config)
    return bert_model, bert_tokenizer, bert_config

def load_masklm_nsp_bert(model_path, logger, args):
    logger.info('Loading BERT model from {}'.format(model_path))
    bert_config = BertConfig.from_pretrained(model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForMaskLMAndNSP.from_pretrained(model_path)
    checkpoints = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')

    # update nsp cls 
    logger.info('Loading nsp cls params')
    bert_model.nsp_cls.seq_relationship.weight.data = checkpoints.pop('cls.seq_relationship.weight')
    bert_model.nsp_cls.seq_relationship.bias.data = checkpoints.pop('cls.seq_relationship.bias')
    # bert_model.load_state_dict(checkpoints)
    return bert_model, bert_tokenizer, bert_config
    

def load_ffn_adapter_bert(model_path, logger, args, model_class=BertAttentionFfnAdapterForSequenceClassification):
    logger.info('Loading BERT model from {}'.format(model_path))
    bert_config = BertConfig.from_pretrained(model_path)
    bert_config.num_labels = args.num_labels
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)

    if args.type == 'train':
        bert_model = model_class.from_pretrained(model_path, config=bert_config, ffn_adapter_size=args.ffn_adapter_size, prefix_len=args.prefix_len)

        # 如果二者都没有，则不锁参数
        if args.ffn_adapter_size + args.prefix_len > 0:
            update_params = 0
            fixed_params = 0
            for k, v in bert_model.named_parameters():
                if 'ffn_adapter' not in k and 'prefix_embedding' not in k:
                    v.requires_grad = False
                    fixed_params += reduce(lambda x, y: x * y, v.size())
                else:
                    update_params += reduce(lambda x, y: x * y, v.size())
                logger.info('{}:{}'.format(k, v.requires_grad))
            logger.info('total params:{}, updated:{}, fixed:{}, update percentage {}%'.format(update_params + fixed_params, update_params, fixed_params, update_params / (update_params + fixed_params) * 100))
    else:
        bert_model = model_class(bert_config, ffn_adapter_size=args.ffn_adapter_size, prefix_len=args.prefix_len)

    return bert_model, bert_tokenizer, bert_config

def circle_loss(y_pred, y_true):
    # origin ner circle loss
    # _, _, seq_len = y_pred.size()
    # y_pred = y_pred.reshape(-1, seq_len)
    # y_true = y_true.reshape(-1, seq_len)

    last_dim = y_pred.size()[-1]
    y_pred = y_pred.reshape(-1, last_dim)
    y_true = y_true.reshape(-1, last_dim)
    
    zeros = torch.zeros_like(y_pred[..., :1])
    
    y_true_p = (y_true == 1).to(torch.float32)
    y_true_n = (y_true == -1).to(torch.float32)
    
    y_pred_p = -y_pred + (1 - y_true_p) * -1e12
    y_pred_n = y_pred + (1 - y_true_n) * -1e12

    y_pred_p = torch.cat([y_pred_p, zeros], dim=-1)
    y_pred_n = torch.cat([y_pred_n, zeros], dim=-1)

    p_loss = torch.logsumexp(y_pred_p, dim=-1)
    n_loss = torch.logsumexp(y_pred_n, dim=-1)
    
    loss = p_loss + n_loss

    return loss.mean(dim=-1)

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none').sum(dim=-1)
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none').sum(dim=-1)
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
# 基于Partition v2的结果，进行个数分类的计算

from functools import wraps
import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from transformers.utils.dummy_pt_objects import FlaubertForQuestionAnswering
from dataset import PartitionClaDataset, PartitionPredictDataset
from log import Logger
from argparse import ArgumentParser
from utils import print_args, get_optimizer_and_scheduler, read_partition_cla_data, load_ffn_adapter_bert, circle_loss, compute_kl_loss
from torch.utils.data import DataLoader
from model import BertAttentionFfnAdapterForMaskedLM 
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-device', default=1, type=int)
    parser.add_argument('-output_name', default='test', type=str)
    parser.add_argument('-train_batch_size', default=128, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-eval_batch_size', default=1, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch_num', default=20, type=int)
    parser.add_argument('-num_labels', default=3, type=int) # 个数在11及其以上的均视作同一类
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-ffn_adapter_size', default=0, type=int) 
    parser.add_argument('-steps_per_epoch', default=200, type=int) 
    parser.add_argument('-prefix_len', default=0, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)
    parser.add_argument('-r_drop', default='no', type=str)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-use_which_partition', default=2, type=int) # 0表示不使用parition， 1表示使用rule， 2表示使用模型的partition
    parser.add_argument('-data_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/split_cla_v3_data/vanilla_bert_0.5p_rdrop/ha', type=str) # 是否使用规则
    parser.add_argument('-code_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/CHIP-CDN/code.txt', type=str) # 是否使用规则
    parser.add_argument('-code_cla_metric', default='alphabet', type=str) # 如何对icd的code做分类：alphabet：按照第一个字母分类； icd: 按照icd分类, one：表示只有一种类别（相当于不考虑类别任务）
    parser.add_argument('-p', default=0, type=float) # 使用组合数据的概率
    parser.add_argument('-grad_acc_step', default=4, type=int) # 梯度累计步数
    parser.add_argument('-beam_size', default=5, type=int) # 梯度累计步数
    parser.add_argument('-gen_max_len', default=15, type=int) # 梯度累计步数
    
    
    args = parser.parse_args()
    args.r_drop = args.r_drop == 'yes'

    output_path = os.path.join('./output1/Bert_partition_cla/v1', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data, standard_to_code, code_to_idx, idx_to_code  = read_partition_cla_data(args.data_path, args, logger)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_ffn_adapter_bert(pretrained_model_path, logger=logger, args=args, model_class=BertAttentionFfnAdapterForMaskedLM)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        if args.saved_model_path is not None:
            logger.info('load saved model from {}'.format(args.saved_model_path))
            checkpoint = torch.load(args.saved_model_path, map_location='cpu')
            bert_model.load_state_dict(checkpoint)
            bert_model = bert_model.to(args.device)

        # #准备数据
        train_dataset = PartitionClaDataset(train_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, shuffle=True, while_true=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

        dev_dataset = PartitionClaDataset(dev_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset = PartitionClaDataset(test_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * args.epoch_num
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        evaluate(dev_dataloader, bert_model, args, code_to_idx, bert_config, bert_tokenizer) 
        train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer)
    
    elif args.type == 'evaluate':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        # test_dataset = PartitionClaDataset(train_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        # test_dataset = PartitionClaDataset(test_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        test_dataset = PartitionClaDataset(dev_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        # 传统方式，贪心预测
        # logger.info('top1 pred num acc')
        # num_acc, y_pred, y_true = evaluate(test_dataloader, bert_model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=True)  
        # logger.info('num_acc: {:.2f}'.format(num_acc))
        # pred_saved_path = args.saved_model_path.replace('best_num_acc_model.pth', 'top1_pred_result.txt')
        # write_top1_pred_result(test_dataloader.dataset.data_list, y_pred, y_true, pred_saved_path)

        # beam search 预测
        logger.info('beam search pred num acc')
        num_acc, y_pred, y_pred_score, y_true = beam_search(test_dataloader, bert_model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=True)
        logger.info('num_acc: {:.2f}'.format(num_acc))
        pred_saved_path = args.saved_model_path.replace('best_num_acc_model.pth', 'beam_search_result.txt')
        write_beam_search_results(test_dataloader.dataset.data_list, y_pred, y_pred_score, y_true, pred_saved_path)
        
    elif args.type == 'predict':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        for data_list, name in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
            predict_dataset = PartitionClaDataset(data_list, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
            predict_dataloader = DataLoader(predict_dataset, batch_size=args.eval_batch_size, collate_fn=predict_dataset.collate_fn)

            logger.info('beam search pred num acc')
            num_acc, y_pred, y_pred_score, y_true = beam_search(predict_dataloader, bert_model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=True)
            logger.info('num_acc: {:.2f}'.format(num_acc))
            pred_saved_path = args.saved_model_path.replace('best_num_acc_model.pth', 'beam_search_{}_result.txt'.format(name))
            write_beam_search_results(data_list, y_pred, y_pred_score, y_true, pred_saved_path)

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer):
    model.train()
    loss_list = []
    token_acc_list = []

    best_num_acc = 0
    step = 0

    model_saved_path = os.path.join(output_path, 'saved_model')
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    batch_iter = iter(train_dataloader)
    for epoch in range(args.epoch_num):
        logger.info('#'*20 + 'Epoch{}'.format(epoch + 1) + '#'*20)
        iteration = tqdm(range(args.steps_per_epoch), desc='Training')
        model.zero_grad()
        for _ in iteration:
            loss = 0
            batch = next(batch_iter)
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, labels = batch
            output = model.forward(input_ids, attention_mask, labels=labels)

            loss += output.loss

            if args.r_drop:
                output1 = model.forward(input_ids, attention_mask, labels=labels)
                loss += compute_kl_loss(output.logits, output1.logits, pad_mask=(labels == -100))

            loss_list.append(loss.item())

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            logits = output.logits

            token_acc = get_token_acc(logits, labels, code_to_idx)
            token_acc_list.append(token_acc)

            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'total loss:{},token acc : {}%,lr:{}'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(token_acc_list) / len(token_acc_list), 2),
                    round(lr, 7)))
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % args.grad_acc_step == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        num_acc = evaluate(dev_dataloader, model, args, code_to_idx, bert_config, bert_tokenizer) 

        model.train()

        if num_acc > best_num_acc:
            best_num_acc = num_acc
            logger.info('save model at f1 {}'.format(best_num_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_num_acc_model.pth')) 

    logger.info('#'*20 + 'Evaluate' + '#'*20)
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_num_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_num_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    num_acc, y_pred, y_true = evaluate(test_dataloader, model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=True)  
    logger.info('num_acc: {:.2f}'.format(num_acc))

def write_top1_pred_result(data_list, y_pred, y_true, output_path):
    with open(output_path, 'w') as f:
        for true, pred, data in zip(y_true, y_pred, data_list):
            # print(true, pred)
            raw_sen, standard_sen, label_num, rule_split_sen, model_split_sen = data
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(raw_sen, standard_sen, rule_split_sen, model_split_sen, true, pred, len(true), len(pred)))
        f.close()

# 用于后续做分类训练
def write_beam_search_results(data_list, y_pred, y_pred_score, y_true, output_path):
    with open(output_path, 'w') as f:
        for true, pred, score, data in zip(y_true, y_pred, y_pred_score, data_list):
            # print(true, pred)
            raw_sen, standard_sen, label_num, rule_split_sen, model_split_sen = data
            pred_zip = '+++'.join([str(x) for x in list(zip(pred, score))])

            f.write('{}\t{}\t{}\t{}\t{}\n'.format(raw_sen, rule_split_sen, model_split_sen, pred_zip, true))
        f.close()

# top1 eval
def evaluate(dataloader, model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=False):
    # 生成出来的token id
    id_list = list(code_to_idx.values()) + [bert_tokenizer.sep_token_id] 
    id_mask = torch.tensor([x not in id_list for x in range(bert_config.vocab_size)], dtype=torch.bool).to(model.device)

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, total=len(dataloader.dataset.data_list)):
            item_pred_list = []
            
            input_ids, attention_mask, labels = [x.to(args.device) for x in item]
            input_len = len(input_ids[0]) 

            for _ in range(15):
                output = model.forward(input_ids, attention_mask)
                logits = output.logits[0, -1]
                logits = logits.masked_fill(id_mask, -1e10)
                pred_token = logits.argmax(dim=-1)
                if pred_token == bert_tokenizer.sep_token_id:
                    break
                else:
                    item_pred_list.append(pred_token.item())
                    input_ids = torch.cat([input_ids, pred_token.unsqueeze(dim=0).unsqueeze(dim=0)], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(args.device)], dim=-1)
                    input_len += 1
                
            y_pred.append(item_pred_list)
            y_true.append(labels[0].tolist())

    num_acc = 0

    for pred, true in zip(y_pred, y_true):
        if len(pred) == len(true):
            num_acc += 1

    num_acc = num_acc / len(y_true) * 100
    if is_eval:
        return num_acc, y_pred, y_true
    else:
        return num_acc

# 得到token的acc
def get_token_acc(logits, labels, code_to_idx):
    pred_tokens = logits.argmax(dim=-1)[labels != -100]
    labels = labels[labels != -100]
    acc = (pred_tokens == labels).to(torch.int32).sum().item() / len(labels) * 100
    return acc
    

def predict(model, dataloader, data_list, logger, args):
    model.eval()

    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to(args.device) for x in batch] 
            output = model.forward(*batch)
            # batch_pred = (output.logits > 0).to(int).squeeze(dim=-1).cpu().tolist()
            batch_pred = output.logits.argmax(dim=-1).cpu().tolist()
            y_pred += batch_pred
    
    assert len(y_pred) == len(data_list)
    return decode_label(y_pred, data_list)

def decode_label(y_pred, data_list):
    res_list = []
    for pred_list, data in zip(y_pred, data_list):
        s = data[0]
        res = ''
        for i in range(1, len(pred_list)):
            if i < len(s) + 1:
                if pred_list[i] == 2:
                    continue
                elif pred_list[i] == 0:
                    res += s[i - 1]
                elif pred_list[i] == 1:
                    res += s[i - 1] + '###'
                    # if i == len(s): res += '###'
        # res_list.append(data + [','.join(res)])
        res_list.append(data + [res])

    return res_list

# 返回topk个最大的句子
# beam search eval
def beam_search(dataloader, model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=False):
    # 生成出来的token id, 这里并不mask sep token，避免第一个生成sep
    id_list = list(code_to_idx.values())
    id_mask = torch.tensor([x not in id_list for x in range(bert_config.vocab_size)], dtype=torch.bool).to(model.device)

    y_pred_id_list = []
    y_pred_score_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, total=len(dataloader.dataset.data_list)):
            input_ids, attention_mask, labels = [x.to(args.device) for x in item]
            origin_input_ids = input_ids

            finished_output_ids, finished_output_scores = [], []
            on_process_output_ids, on_process_output_scores = None, torch.zeros(1, 1).to(args.device) #记录目前已经生成且尚未结束的的output_id 以及对应的 scores

            for step in range(args.gen_max_len):
                output = model.forward(input_ids, attention_mask)

                logits = output.logits[:, -1] #取最后一个token的logits, bs, 1
                logits = logits.masked_fill(id_mask, -1e10) # 加入对特定字符的限制
                batch_scores = torch.log_softmax(logits, dim=-1) # 转化为score
                batch_scores += on_process_output_scores # 和当前batch已有的output scores求和
                
                topk_score, topk_index  = torch.topk(batch_scores.view(-1, 1), k=args.beam_size, dim=0) # 取topk

                # 得到topk对应的row和vocab id
                vocab_size = batch_scores.size(-1)
                r_ids, vocab_ids = [], []
                for idx in topk_index:
                    idx = idx[0]
                    r_idx = idx // vocab_size
                    vocab_idx = idx % vocab_size
                    r_ids.append(r_idx)
                    vocab_ids.append(vocab_idx)
                
                # 当下topk对应现有输入的行号，以及其对应当下的vocab
                r_ids = torch.tensor(r_ids).to(args.device)
                vocab_ids = torch.tensor(vocab_ids).to(args.device)

                # 此时，修改on_process_ouput_ids, 更新为最新
                if step != 0:
                    on_process_output_ids = on_process_output_ids[r_ids]
                else:
                    on_process_output_ids = vocab_ids.unsqueeze(dim=-1)
                    id_mask[bert_tokenizer.sep_token_id] = False # 加入sep mask，允许结束
                on_process_output_scores = topk_score

                # 根据vocab id筛选出已经结束的
                end_flag = vocab_ids == bert_tokenizer.sep_token_id
                # 如果有已经结束的(step等于0时，不会进入该判断)
                if sum(end_flag) > 0:
                    end_scores = on_process_output_scores[end_flag]
                    end_ids = on_process_output_ids[end_flag]
                    end_scores /= end_ids.size()[-1] # avg log sum
                    finished_output_ids += end_ids.cpu().tolist()
                    finished_output_scores += end_scores.cpu().tolist()
                    if len(finished_output_ids) >= args.beam_size:
                        break

                # 把没有结束的过滤出来
                on_process_output_scores = on_process_output_scores[~end_flag]
                if step != 0:
                    on_process_output_ids = on_process_output_ids[~end_flag]
                    not_end_vocab_ids = vocab_ids[~end_flag].unsqueeze(dim=-1)
                    on_process_output_ids = torch.cat([on_process_output_ids, not_end_vocab_ids], dim=-1)
                
                input_ids = torch.cat([origin_input_ids.repeat(len(on_process_output_ids), 1), on_process_output_ids], dim=-1)
                attention_mask = torch.ones_like(input_ids)

            if len(finished_output_ids) < args.beam_size:
                finished_output_ids += on_process_output_ids.cpu().tolist()
                finished_output_scores += on_process_output_scores.cpu().tolist()

            finished_output_ids = finished_output_ids[:args.beam_size]
            finished_output_scores = finished_output_scores[:args.beam_size]
            
            y_pred_id_list.append(finished_output_ids)
            y_pred_score_list.append(finished_output_scores)
            y_true_list.append(labels[0].tolist())

    y_pred_score = torch.tensor(y_pred_score_list).squeeze(dim=-1)
    y_pred_max_idx = torch.argmax(y_pred_score, dim=-1)
    # y_pred_list = []
    num_acc = 0
    for pred_list, pred_max_idx, true_label in zip(y_pred_id_list, y_pred_max_idx, y_true_list):
        # y_pred_list.append(pred_list[pred_max_idx])
        if len(pred_list[pred_max_idx]) == len(true_label):
            num_acc += 1
        # for pred in pred_list:
        #     if len(pred) == len(true_label):
        #         num_acc += 1
        #         break
    
    num_acc = num_acc / len(y_pred_max_idx) * 100

    if is_eval:
        return num_acc, y_pred_id_list, y_pred_score_list, y_true_list
    else:
        return num_acc


if __name__ == '__main__':
    main()
    
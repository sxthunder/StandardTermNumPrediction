# Rank和Generate Joint训练，共享参数
# 先训练M轮Generate，再进行M轮联合训练

import math
import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from transformers.utils.dummy_pt_objects import FlaubertForQuestionAnswering
from dataset import PartitionClaDataset, PartitionPredictDataset, PartitionClaRankDataset
from log import Logger
from argparse import ArgumentParser
from utils import print_args, get_optimizer_and_scheduler, load_masklm_nsp_bert, compute_kl_loss, read_partition_cla_rank_data, read_partition_cla_data
from torch.utils.data import DataLoader
from model import BertAttentionFfnAdapterForMaskedLM 
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-output_name', default='test', type=str)
    parser.add_argument('-gen_train_batch_size', default=128, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-joint_train_batch_size', default=20, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-gen_eval_batch_size', default=1, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-rank_eval_batch_size', default=256, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-gen_epoch_num', default=10, type=int)
    parser.add_argument('-joint_epoch_num', default=20, type=int)
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-steps_per_epoch', default=250, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)
    parser.add_argument('-r_drop', default='no', type=str)
    parser.add_argument('-use_which_partition', default=2, type=int) # 0表示不使用parition， 1表示使用rule， 2表示使用模型的partition
    parser.add_argument('-rank_use_which_partition', default=0, type=int) # 0表示不使用parition， 1表示使用rule， 2表示使用模型的partition
    parser.add_argument('-data_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/split_cla_v3_data/vanilla_bert_0.5p_rdrop/ha', type=str) # 是否使用规则
    parser.add_argument('-code_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/CHIP-CDN/code.txt', type=str) # 是否使用规则
    parser.add_argument('-grad_acc_step', default=4, type=int) # 梯度累计步数
    parser.add_argument('-code_metric', default='alphabet', type=str) # icd使用的类型：alphabet or icd
    parser.add_argument('-beam_size', default=5, type=int) # 梯度累计步数
    parser.add_argument('-gen_max_len', default=15, type=int) # 梯度累计步数
    parser.add_argument('-code_cla_metric', default='alphabet', type=str) # 梯度累计步数
    parser.add_argument('-p', default=0, type=float) # 梯度累计步数
    
    args = parser.parse_args()
    args.r_drop = args.r_drop == 'yes'

    output_path = os.path.join('./output/Bert_partition_cla_rank_merge', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_gen_data, dev_gen_data, test_gen_data, standard_to_code, code_to_idx, idx_to_code  = read_partition_cla_data(args.data_path, args, logger)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_masklm_nsp_bert(pretrained_model_path, logger, args)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        if args.saved_model_path is not None:
            logger.info('load saved model from {}'.format(args.saved_model_path))
            checkpoint = torch.load(args.saved_model_path, map_location='cpu')
            bert_model.load_state_dict(checkpoint)
            bert_model = bert_model.to(args.device)

        # 准备数据, 先加在partition generate的参数
        train_gen_dataset = PartitionClaDataset(train_gen_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, shuffle=True, while_true=True)
        train_gen_dataloader = DataLoader(train_gen_dataset, batch_size=args.gen_train_batch_size, collate_fn=train_gen_dataset.collate_fn)

        dev_gen_dataset = PartitionClaDataset(dev_gen_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        dev_gen_dataloader = DataLoader(dev_gen_dataset, batch_size=args.gen_eval_batch_size, collate_fn=dev_gen_dataset.collate_fn)

        test_gen_dataset = PartitionClaDataset(test_gen_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        test_gen_dataloader = DataLoader(test_gen_dataset, batch_size=args.gen_eval_batch_size, collate_fn=test_gen_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * (args.gen_epoch_num + args.joint_epoch_num) 
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        # 先训练generate model
        if args.gen_epoch_num > 0:
            logger.info("#"*20 + "Generation Train" + "#"*20)
            bert_model = generate_train(bert_model, train_gen_dataloader, dev_gen_dataloader, test_gen_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer)

        logger.info("#"*20 + "Joint Train" + "#"*20)
        joint_train(bert_model, train_gen_dataloader, dev_gen_dataloader, test_gen_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer, standard_to_code)
    
    elif args.type == 'evaluate':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        test_gen_dataset = PartitionClaDataset(test_gen_data, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False)
        test_gen_dataloader = DataLoader(test_gen_dataset, batch_size=args.gen_eval_batch_size, collate_fn=test_gen_dataset.collate_fn)
        test_rank_dataloader = generate_rank_dataloader(bert_model, test_gen_dataloader, args, code_to_idx, bert_config, bert_tokenizer, logger, 'test')

        only_gen_num_acc, only_rank_num_acc, merge_num_acc = joint_evaluate(test_rank_dataloader, bert_model, args, logger)

    elif args.type == 'predict':
        pass

# 输入生成模型 + 生成的dataloader， 返回rank的dataloader
def generate_rank_dataloader(model, gen_dataloader, args, code_to_idx, bert_config, bert_tokenizer, logger, name):
    _, y_pred, y_pred_score, y_true = gen_beam_search_eval(gen_dataloader, model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=True)
    rank_data_list = []
    for true, pred, score, data in zip(y_true, y_pred, y_pred_score, gen_dataloader.dataset.data_list):
        raw_sen, standard_sen, label_num, rule_split_sen, model_split_sen = data
        pred_zip = '+++'.join([str(x) for x in list(zip(pred, score))])
        rank_data_list.append([raw_sen, rule_split_sen, model_split_sen, pred_zip, true])

    
    # 如果是训练，rank的batch size乘上beam_size倍
    batch_size = args.rank_eval_batch_size if name != 'train' else args.joint_train_batch_size * args.beam_size
    # shuffle = name == 'train'
    shuffle = False
    while_true = name == 'train'

    logger.info('#'*20 + '{} dataloader params'.format(name) + '#'*20)
    logger.info('data length : {}, batch size :{}, shuffle :{}, while_true :{}'.format(len(rank_data_list), batch_size, shuffle, while_true))

    # #准备数据
    rank_dataset = PartitionClaRankDataset(rank_data_list, bert_tokenizer, args, shuffle=shuffle, while_true=while_true)
    rank_dataloader = DataLoader(rank_dataset, batch_size=batch_size, collate_fn=rank_dataset.collate_fn)

    return rank_dataloader

# 生成模型训练
def generate_train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer):
    model.train()
    loss_list = []
    token_acc_list = []

    best_num_acc = 0
    step = 0

    model_saved_path = os.path.join(output_path, 'saved_model')
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    batch_iter = iter(train_dataloader)
    for epoch in range(args.gen_epoch_num):
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

        logger.info('#'*20 + 'Evaluate by beam search' + '#'*20)
        num_acc = gen_beam_search_eval(dev_dataloader, model, args, code_to_idx, bert_config, bert_tokenizer) 

        model.train()

        if num_acc > best_num_acc:
            best_num_acc = num_acc
            logger.info('save model at f1 {}'.format(best_num_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_generate_num_acc_model.pth')) 

    logger.info('#'*20 + 'Test Evaluate' + '#'*20)
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_generate_num_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_generate_num_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    num_acc = gen_beam_search_eval(test_dataloader, model, args, code_to_idx, bert_config, bert_tokenizer) 
    logger.info('num_acc: {:.2f}'.format(num_acc))

    return model

# 得到token的acc
def get_token_acc(logits, labels, code_to_idx):
    pred_tokens = logits.argmax(dim=-1)[labels != -100]
    labels = labels[labels != -100]
    acc = (pred_tokens == labels).to(torch.int32).sum().item() / len(labels) * 100
    return acc

# 生成式beam search的eval
def gen_beam_search_eval(dataloader, model, args, code_to_idx, bert_config, bert_tokenizer, is_eval=False):
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
            id_mask[bert_tokenizer.sep_token_id] = True # 第一个step不允许生成sep

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

def joint_train(model, train_gen_dataloader, dev_gen_dataloader, test_gen_dataloader, optimizer, scheduler, args, output_path, logger, code_to_idx, bert_config, bert_tokenizer, standard_to_code):
    model.train()
    loss_list = []
    gen_loss_list = []
    rank_loss_list = []
    rank_acc_list = []
    gen_token_acc_list = []

    best_only_gen_num_acc = 0
    best_only_rank_num_acc = 0
    best_merge_acc = 0
    step = 0

    model_saved_path = os.path.join(output_path, 'saved_model')
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    # 联合训练时，将generate的train的batch_size做修改, 联合训练时shuffle=False, 且rank的batch_size=gen_batch_size * 5
    train_gen_dataset = PartitionClaDataset(train_gen_dataloader.dataset.data_list, standard_to_code, code_to_idx, bert_tokenizer, logger, args, shuffle=False, while_true=True)
    train_gen_dataloader = DataLoader(train_gen_dataset, batch_size=args.joint_train_batch_size, collate_fn=train_gen_dataset.collate_fn)
    gen_batch_iter = iter(train_gen_dataloader)

    for epoch in range(args.joint_epoch_num):
        logger.info('#'*20 + 'Epoch{}'.format(epoch + 1) + '#'*20)
        logger.info('#'*20 + 'Generate rank dataloader' + '#'*20)

        # 为了生成rank train, 新建一个batch_size=1的gen_traindataloader, 为了保证rank和generate能够对齐, 生成rank的shuffle=False
        train_gen_dataset_for_pred = PartitionClaDataset(train_gen_dataloader.dataset.data_list, standard_to_code, code_to_idx, bert_tokenizer, logger, args, while_true=False, shuffle=False)
        train_gen_dataloader_for_pred = DataLoader(train_gen_dataset_for_pred, batch_size=args.gen_eval_batch_size, collate_fn=train_gen_dataset_for_pred.collate_fn)
        train_rank_dataloader = generate_rank_dataloader(model, train_gen_dataloader_for_pred, args, code_to_idx, bert_config, bert_tokenizer, logger, 'train')
        dev_rank_dataloader = generate_rank_dataloader(model, dev_gen_dataloader, args, code_to_idx, bert_config, bert_tokenizer, logger, 'dev')
        test_rank_dataloader = generate_rank_dataloader(model, test_gen_dataloader, args, code_to_idx, bert_config, bert_tokenizer, logger, 'test')
        only_gen_num_acc, only_rank_num_acc, merge_num_acc = joint_evaluate(dev_rank_dataloader, model, args, logger)
        

        rank_batch_iter = iter(train_rank_dataloader)
        iteration = tqdm(range(args.steps_per_epoch), desc='Training')
        model.zero_grad()
        for _ in iteration:
            loss = 0
            rank_loss = 0
            gen_loss = 0

            # 训练rank
            rank_batch = next(rank_batch_iter)
            input_ids, attention_mask, gen_pred_scores, cla_labels = [x.to(args.device) for x in rank_batch]
            rank_output = model.forward(input_ids, attention_mask, labels=cla_labels, is_mlm=False)
            rank_loss += rank_output.loss

            # if args.r_drop:
            #     output1 = model.forward(input_ids, attention_mask)
            #     loss += compute_kl_loss(output.logits, output1.logits)

            rank_loss_list.append(rank_loss.item())

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            rank_logits = rank_output.logits

            rank_acc = (rank_logits.argmax(dim=-1) == cla_labels).sum() / len(cla_labels) * 100
            rank_acc_list.append(rank_acc.item())

            # 训练generate
            gen_batch = next(gen_batch_iter)
            gen_batch = [x.to(args.device) for x in gen_batch]
            input_ids, attention_mask, labels = gen_batch
            gen_output = model.forward(input_ids, attention_mask, labels=labels)

            gen_loss += gen_output.loss

            # if args.r_drop:
            #     output1 = model.forward(input_ids, attention_mask, labels=labels)
                # loss += compute_kl_loss(output.logits, output1.logits, pad_mask=(labels == -100))

            gen_loss_list.append(gen_loss.item())

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            gen_logits = gen_output.logits

            gen_token_acc = get_token_acc(gen_logits, labels, code_to_idx)
            gen_token_acc_list.append(gen_token_acc)

            loss = gen_loss + rank_loss
            loss_list.append(loss.item())

            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                    "total loss:{:.4f}, rank loss:{:.4f}, rank acc:{:.2f}, gen loss:{:.4f}, gen token acc:{:.2f}".format(
                        loss, rank_loss, rank_acc, gen_loss, gen_token_acc
                    )
                )
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % args.grad_acc_step == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        only_gen_num_acc, only_rank_num_acc, merge_num_acc = joint_evaluate(dev_rank_dataloader, model, args, logger)

        model.train()

        if only_gen_num_acc > best_only_gen_num_acc:
            best_only_gen_num_acc = only_gen_num_acc
            logger.info('save model at best_only_gen_num_acc {}'.format(best_only_gen_num_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_joint_only_gen_num_acc_model.pth')) 

        if only_rank_num_acc > best_only_rank_num_acc:
            best_only_rank_num_acc = only_rank_num_acc
            logger.info('save model at best_only_rank_num_acc {}'.format(best_only_rank_num_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_joint_only_rank_num_acc_model.pth')) 
        
        if merge_num_acc > best_merge_acc:
            best_merge_acc = merge_num_acc
            logger.info('save model at best_merge_acc {}'.format(best_merge_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_joint_merge_acc_model.pth')) 


    logger.info('#'*20 + 'Evaluate' + '#'*20)
    logger.info('Evaluate best only rank model')
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_joint_only_rank_num_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_joint_only_rank_num_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    joint_evaluate(test_rank_dataloader, model, args, logger)

    logger.info('Evaluate best only gen model')
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_joint_only_gen_num_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_joint_only_gen_num_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    joint_evaluate(test_rank_dataloader, model, args, logger)

    logger.info('Evaluate best merge model')
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_joint_merge_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_joint_merge_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    joint_evaluate(test_rank_dataloader, model, args, logger)

def joint_evaluate(dataloader, model, args, logger):
    y_pred = []
    gen_pred_scores = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, total=math.floor(len(dataloader.dataset.data_list) * 5 / dataloader.batch_size)):
            input_ids, attention_mask, batch_gen_pred_scores, cla_labels = [x.to(args.device) for x in item]
            output = model.forward(input_ids, attention_mask, is_mlm=False)
            logits = output.logits

            y_pred.append(torch.log_softmax(logits, dim=-1)[:, 1].cpu())
            gen_pred_scores.append(batch_gen_pred_scores.cpu())
            y_true.append(cla_labels.cpu())

        y_pred = torch.cat(y_pred, dim=0).reshape(-1, 5)
        gen_pred_scores = torch.cat(gen_pred_scores, dim=0).reshape(-1, 5)
        y_true = torch.cat(y_true, dim=0).reshape(-1, 5)

        assert len(y_pred) == len(gen_pred_scores), print(len(y_pred), len(gen_pred_scores))
        assert len(y_pred) == len(y_true)

    only_gen_num_acc = 0
    only_rank_num_acc = 0
    merge_num_acc = 0

    pred_gen_count = 0

    for pred, gen_pred, true in zip(y_pred, gen_pred_scores, y_true):
        if pred.argmax().item() == gen_pred.argmax().item():
            pred_gen_count += 1
        only_rank_num_acc += true[pred.argmax().item()].item()
        only_gen_num_acc += true[gen_pred.argmax().item()].item()
        merge_num_acc += true[(gen_pred + pred).argmax().item()].item()

    print(pred_gen_count, only_rank_num_acc, only_gen_num_acc)
    
    only_gen_num_acc = only_gen_num_acc / len(y_pred) * 100
    only_rank_num_acc = only_rank_num_acc / len(y_pred) * 100
    merge_num_acc = merge_num_acc / len(y_pred) * 100
    
    logger.info('only gen num acc : {:.2f}'.format(only_gen_num_acc))
    logger.info('only rank num acc : {:.2f}'.format(only_rank_num_acc))
    logger.info('merge num acc : {:.2f}'.format(merge_num_acc))

    return only_gen_num_acc, only_rank_num_acc, merge_num_acc

if __name__ == '__main__':
    main()
    
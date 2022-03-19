# 基于Partition v2的结果，进行个数分类的计算

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
from utils import print_args, get_optimizer_and_scheduler, load_nsp_bert, compute_kl_loss, read_partition_cla_rank_data
from torch.utils.data import DataLoader
from model import BertAttentionFfnAdapterForMaskedLM 
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-output_name', default='test', type=str)
    parser.add_argument('-train_batch_size', default=128, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-eval_batch_size', default=256, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch_num', default=20, type=int)
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-steps_per_epoch', default=200, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)
    parser.add_argument('-r_drop', default='no', type=str)
    parser.add_argument('-rank_use_which_partition', default=0, type=int) # 0表示不使用parition， 1表示使用rule， 2表示使用模型的partition 就是之前的use_which_parititon
    parser.add_argument('-data_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/split_rank_data', type=str) # 是否使用规则
    parser.add_argument('-grad_acc_step', default=4, type=int) # 梯度累计步数
    parser.add_argument('-code_metric', default='alphabet', type=str) # icd使用的类型：alphabet or icd
    
    args = parser.parse_args()
    args.r_drop = args.r_drop == 'yes'

    output_path = os.path.join('./output1/Bert_partition_cla_rank', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data  = read_partition_cla_rank_data(args.data_path, args, logger)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_nsp_bert(pretrained_model_path, logger, args)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        if args.saved_model_path is not None:
            logger.info('load saved model from {}'.format(args.saved_model_path))
            checkpoint = torch.load(args.saved_model_path, map_location='cpu')
            bert_model.load_state_dict(checkpoint)
            bert_model = bert_model.to(args.device)

        # #准备数据
        train_dataset = PartitionClaRankDataset(train_data, bert_tokenizer, args, shuffle=True, while_true=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

        dev_dataset = PartitionClaRankDataset(dev_data, bert_tokenizer, args, shuffle=True, while_true=False)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset = PartitionClaRankDataset(test_data, bert_tokenizer, args, shuffle=True, while_true=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * args.epoch_num
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger)
    
    elif args.type == 'evaluate':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        test_dataset = PartitionClaRankDataset(test_data, bert_tokenizer, args, shuffle=False, while_true=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        evaluate(test_dataloader, bert_model, args, logger)

    elif args.type == 'predict':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        test_dataset = PartitionClaRankDataset(test_data, bert_tokenizer, args, shuffle=False, while_true=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        pred_ids = predict(test_dataloader, bert_model, args, logger)
        import pickle
        pickle.dump(pred_ids, open(os.path.join(args.saved_model_path.replace('best_merge_acc_model.pth', 'test_pred_num_ids')), 'wb'))

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger):
    model.train()
    loss_list = []
    acc_list = []

    best_only_rank_num_acc = 0
    best_merge_acc = 0
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
            input_ids, attention_mask, gen_pred_scores, cla_labels = [x.to(args.device) for x in batch]
            output = model.forward(input_ids, attention_mask, labels=cla_labels)
            loss += output.loss

            if args.r_drop:
                output1 = model.forward(input_ids, attention_mask)
                loss += compute_kl_loss(output.logits, output1.logits)

            loss_list.append(loss.item())

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            logits = output.logits

            acc = (logits.argmax(dim=-1) == cla_labels).sum() / len(cla_labels) * 100
            acc_list.append(acc.item())

            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'total loss:{},token acc : {}%,lr:{}'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(acc_list) / len(acc_list), 2),
                    round(lr, 7)))
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % args.grad_acc_step == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        only_gen_num_acc, only_rank_num_acc, merge_num_acc = evaluate(dev_dataloader, model, args, logger)

        model.train()

        if only_rank_num_acc > best_only_rank_num_acc:
            best_only_rank_num_acc = only_rank_num_acc
            logger.info('save model at best_only_rank_num_acc {}'.format(best_only_rank_num_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_only_rank_num_acc_model.pth')) 
        
        if merge_num_acc > best_merge_acc:
            best_merge_acc = merge_num_acc
            logger.info('save model at best_merge_acc {}'.format(best_merge_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_merge_acc_model.pth')) 

    logger.info('#'*20 + 'Evaluate' + '#'*20)
    logger.info('Evaluate best only rank model')
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_only_rank_num_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_only_rank_num_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    evaluate(test_dataloader, model, args, logger)

    logger.info('Evaluate best merge model')
    logger.info('load model from {}'.format(os.path.join(model_saved_path, 'best_merge_acc_model.pth')))
    checkpoint = torch.load(os.path.join(model_saved_path, 'best_merge_acc_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    evaluate(test_dataloader, model, args, logger)

def evaluate(dataloader, model, args, logger):
    y_pred = []
    gen_pred_scores = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, total=math.floor(len(dataloader.dataset.data_list) * 5 / dataloader.batch_size)):
            input_ids, attention_mask, batch_gen_pred_scores, cla_labels = [x.to(args.device) for x in item]
            output = model.forward(input_ids, attention_mask)
            logits = output.logits

            y_pred.append(torch.log_softmax(logits, dim=-1)[:, 1])
            gen_pred_scores.append(batch_gen_pred_scores)
            y_true.append(cla_labels)

        y_pred = torch.cat(y_pred, dim=0).reshape(-1, 5)
        gen_pred_scores = torch.cat(gen_pred_scores, dim=0).reshape(-1, 5)
        y_true = torch.cat(y_true, dim=0).reshape(-1, 5)

        assert len(y_pred) == len(gen_pred_scores)
        assert len(y_pred) == len(y_true)

    only_gen_num_acc = 0
    only_rank_num_acc = 0
    merge_num_acc = 0

    pred_ids = []
    

    for pred, gen_pred, true, data in zip(y_pred, gen_pred_scores, y_true, dataloader.dataset.data_list):
        only_rank_num_acc += true[pred.argmax().item()].item()
        only_gen_num_acc += true[gen_pred.argmax().item()].item()
        merge_num_acc += true[(gen_pred + pred).argmax().item()].item()
        pred_ids.append((true[(gen_pred + pred).argmax().item()].item(), (gen_pred + pred).argmax().item()))

    import pickle
    pickle.dump(pred_ids, open('./rank_pred_ids', 'wb'))

    only_gen_num_acc = only_gen_num_acc / len(y_pred) * 100
    only_rank_num_acc = only_rank_num_acc / len(y_pred) * 100
    merge_num_acc = merge_num_acc / len(y_pred) * 100
    
    logger.info('only gen num acc : {:.2f}'.format(only_gen_num_acc))
    logger.info('only rank num acc : {:.2f}'.format(only_rank_num_acc))
    logger.info('merge num acc : {:.2f}'.format(merge_num_acc))

    return only_gen_num_acc, only_rank_num_acc, merge_num_acc

def predict(dataloader, model, args, logger):
    y_pred = []
    gen_pred_scores = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, total=math.floor(len(dataloader.dataset.data_list) * 5 / dataloader.batch_size)):
            input_ids, attention_mask, batch_gen_pred_scores, cla_labels = [x.to(args.device) for x in item]
            output = model.forward(input_ids, attention_mask)
            logits = output.logits

            y_pred.append(torch.log_softmax(logits, dim=-1)[:, 1])
            gen_pred_scores.append(batch_gen_pred_scores)
            y_true.append(cla_labels)

        y_pred = torch.cat(y_pred, dim=0).reshape(-1, 5)
        gen_pred_scores = torch.cat(gen_pred_scores, dim=0).reshape(-1, 5)
        y_true = torch.cat(y_true, dim=0).reshape(-1, 5)

        assert len(y_pred) == len(gen_pred_scores)
        assert len(y_pred) == len(y_true)

    merge_num_acc = 0

    pred_ids = []
    

    for pred, gen_pred, true, data in zip(y_pred, gen_pred_scores, y_true, dataloader.dataset.data_list):
        merge_num_acc += true[(gen_pred + pred).argmax().item()].item()
        # pred_ids.append((true[(gen_pred + pred).argmax().item()].item(), (gen_pred + pred).argmax().item()))
        # pred_ids


    merge_num_acc = merge_num_acc / len(y_pred) * 100
    
    logger.info('merge num acc : {:.2f}'.format(merge_num_acc))

    return pred_ids

if __name__ == '__main__':
    main()
    
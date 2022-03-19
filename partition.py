import os
import torch
from dataset import PartitionDataset, PartitionPredictDataset
from log import Logger
from argparse import ArgumentParser
from utils import print_args, get_optimizer_and_scheduler, read_parition_data, load_ffn_adapter_bert, circle_loss
from torch.utils.data import DataLoader
from model import BertAttentionFfnAdapterForTokenClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-device', default=1, type=int)
    parser.add_argument('-output_name', default='test', type=str)
    parser.add_argument('-train_batch_size', default=128, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-eval_batch_size', default=256, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-max_len', default=128, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch_num', default=20, type=int)
    parser.add_argument('-num_labels', default=1, type=int) # 个数在11及其以上的均视作同一类
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-ffn_adapter_size', default=64, type=int) 
    parser.add_argument('-steps_per_epoch', default=200, type=int) 
    parser.add_argument('-prefix_len', default=68, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)

    args = parser.parse_args()
    output_path = os.path.join('./output/Bert_partition', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    data_path = '/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data/multi_split.txt'
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data = read_parition_data(data_path, logger, args)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_ffn_adapter_bert(pretrained_model_path, logger=logger, args=args, model_class=BertAttentionFfnAdapterForTokenClassification)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        # #准备数据
        train_dataset = PartitionDataset(train_data, bert_tokenizer, args, shuffle=True, while_true=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

        dev_dataset = PartitionDataset(dev_data, bert_tokenizer, args)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset = PartitionDataset(test_data, bert_tokenizer, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * args.epoch_num
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger)
    
    elif args.type == 'evaluate':
        pass
    
    elif args.type == 'predict':
        data_path = '/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data/multi_not_split.txt'
        data_list = []
        with open(data_path, 'r') as f:
            for line in f:
                data_list.append(line.strip().split('\t'))
            f.close()
        str_list = [x[0] for x in data_list]

        predict_dataset = PartitionPredictDataset(str_list, bert_tokenizer, args)
        predict_dataloader = DataLoader(predict_dataset, batch_size=args.eval_batch_size, collate_fn=predict_dataset.collate_fn)

        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        res_list = predict(bert_model, predict_dataloader, data_list, logger, args)
        with open('./res_list', 'w') as f:
            # for res,data in zip(res_list, data_list):
            for res in res_list:
                f.write('\t'.join(res) + '\n')
            f.close()

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger):
    model.train()
    loss_list = []
    acc_list = []
    best_acc = 0
    best_f1 = 0
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
            batch = next(batch_iter)
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, labels = batch
            output = model.forward(input_ids, attention_mask)
            logits = output.logits

            loss = circle_loss(logits, labels)

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_list.append(loss.item())
            batch_y_pred = (logits > 0).to(int)[labels != 0].squeeze(dim=-1)
            labels = ((labels[labels != 0] + 1) / 2).to(int)

            acc = torch.sum((batch_y_pred == labels)) / len(labels) * 100
            acc_list.append(acc.item())
            
            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'total loss:{}, acc : {}%, lr:{}'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(acc_list) / len(acc_list), 2),
                    round(lr, 7)))
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % 4 == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        acc, f1 = evaluate(dev_dataloader, model, logger, args)
        model.train()
        if acc > best_acc:
            best_acc = acc 
            logger.info('save model at acc {}'.format(best_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_acc_model.pth')) 

        if f1 > best_f1:
            best_f1 = f1
            logger.info('save model at f1 {}'.format(best_f1))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_f1_model.pth')) 

    logger.info('#'*20 + 'Evaluate' + '#'*20)
    acc, f1 = evaluate(test_dataloader, model, logger, args)


def evaluate(dataloader, model, logger, args):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, labels = batch
            output = model.forward(input_ids, attention_mask)
            batch_pred = (output.logits > 0).to(int)
            batch_pred = batch_pred[labels != 0].cpu().tolist()
            labels = ((labels[labels != 0] + 1) / 2).to(int).cpu().tolist()

            y_pred += batch_pred
            y_true += labels

    report = classification_report(y_true, y_pred, zero_division=0, digits=4)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    logger.info(report)
    return acc, f1

def predict(model, dataloader, data_list, logger, args):
    model.eval()

    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to(args.device) for x in batch] 
            output = model.forward(*batch)
            batch_pred = (output.logits > 0).to(int).squeeze(dim=-1).cpu().tolist()
            y_pred += batch_pred
    
    assert len(y_pred) == len(data_list)
    return decode_label(y_pred, data_list)

def decode_label(y_pred, data_list):
    res_list = []
    for pred_list, data in zip(y_pred, data_list):
        s = data[0]
        res = []
        idx = 0
        for i in range(1, len(pred_list)):
            if i < len(s):
                if pred_list[i] == 1:
                    res.append(s[idx:i+1])
                    idx = i + 1
        res.append(s[idx:])
        res_list.append(data + [','.join(res)])

    return res_list
    

if __name__ == '__main__':
    main()
    
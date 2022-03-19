import os
import torch
import torch.nn.functional as F
from dataset import PartitionV2Dataset, PartitionPredictDataset
from log import Logger
from argparse import ArgumentParser
from utils import print_args, get_optimizer_and_scheduler, read_parition_data, load_ffn_adapter_bert, circle_loss, compute_kl_loss
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
    parser.add_argument('-num_labels', default=3, type=int) # 个数在11及其以上的均视作同一类
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-ffn_adapter_size', default=0, type=int) 
    parser.add_argument('-steps_per_epoch', default=200, type=int) 
    parser.add_argument('-prefix_len', default=0, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)
    parser.add_argument('-r_drop', default='no', type=str)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-predict_data_path', default=None, type=str)
    parser.add_argument('-p', default=0.3, type=float) # 训练时，从标准词中采样的概率
    

    args = parser.parse_args()
    args.r_drop = args.r_drop == 'yes'


    output_path = os.path.join('./output1/Bert_partition_v2', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    data_path = '/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data'
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data = read_parition_data(data_path, logger, args)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_ffn_adapter_bert(pretrained_model_path, logger=logger, args=args, model_class=BertAttentionFfnAdapterForTokenClassification)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        # #准备数据
        train_dataset = PartitionV2Dataset(train_data, bert_tokenizer, args, shuffle=True, while_true=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

        dev_dataset = PartitionV2Dataset(dev_data, bert_tokenizer, args)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset = PartitionV2Dataset(test_data, bert_tokenizer, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * args.epoch_num
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        evaluate(dev_dataloader, bert_model, logger, args) 
        train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger)
    
    elif args.type == 'evaluate':
        pass
    
    elif args.type == 'predict':
        # data_path = '/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data/train/multi_not_split.txt'
        logger.info('load predict data from {}'.format(args.predict_data_path))
        data_list = []
        with open(args.predict_data_path, 'r') as f:
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

        parent_dir = os.path.abspath(os.path.join(args.saved_model_path, os.pardir))
        model_base_name = os.path.basename(args.saved_model_path).replace('_model.pth', '')
        res_base_name = os.path.basename(args.predict_data_path)
        res_save_path = os.path.join(parent_dir, '{}_predict_{}'.format(model_base_name, res_base_name))
        logger.info('save predict result to {}'.format(res_save_path))
        with open(res_save_path, 'w') as f:
            for res in res_list:
                f.write('\t'.join(res) + '\n')
            f.close()

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger):
    model.train()
    loss_list = []
    token_acc_list = []
    partition_acc_list = []
    strict_acc_list = []

    best_partition_acc = 0
    best_strict_acc = 0
    best_f1 = 0
    best_acc = 0
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

            loss = 0
            if args.r_drop:
                output_1 = model.forward(input_ids, attention_mask)
                pad_mask = (labels.sum(dim=-1) != 0)
                loss += compute_kl_loss(output.logits, output_1.logits, pad_mask=pad_mask)

            logits = output.logits

            loss += circle_loss(logits, labels)

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_list.append(loss.item())

            token_acc, partition_acc, strict_acc = get_acc(logits, labels)
            token_acc_list.append(token_acc)
            partition_acc_list.append(partition_acc)
            strict_acc_list.append(strict_acc)

            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'total loss:{},token acc : {}%,partition acc : {}%,strict acc : {}%, lr:{}'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(token_acc_list) / len(token_acc_list), 2),
                    round(sum(partition_acc_list) / len(partition_acc_list), 2),
                    round(sum(strict_acc_list) / len(strict_acc_list), 2),
                    round(lr, 7)))
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % 4 == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        token_acc, partition_acc, strict_acc, f1, acc = evaluate(dev_dataloader, model, logger, args)

        model.train()
        if acc > best_acc:
            best_acc = acc
            logger.info('save model at acc {}'.format(best_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_acc_model.pth')) 
            
        # if partition_acc > best_partition_acc:
        #     best_partition_acc = partition_acc 
        #     logger.info('save model at partition_acc {}'.format(best_partition_acc))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_partition_acc_model.pth')) 

        # if strict_acc > best_strict_acc:
        #     best_strict_acc = strict_acc 
        #     logger.info('save model at strict_acc {}'.format(best_strict_acc))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_strict_acc_model.pth')) 

        # if f1 > best_f1:
        #     best_f1 = f1
        #     logger.info('save model at f1 {}'.format(best_f1))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_f1_model.pth')) 

    logger.info('#'*20 + 'Evaluate' + '#'*20)
    evaluate(test_dataloader, model, logger, args)


def evaluate(dataloader, model, logger, args):
    model.eval()
    all_logits = []
    all_y_true = []
    all_standard_count = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, labels, standard_count = batch
            output = model.forward(input_ids, attention_mask)

            all_logits.append(output.logits)
            all_y_true.append(labels)
            all_standard_count.append(standard_count)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_y_true, dim=0)
    standard_count = torch.cat(all_standard_count, dim=0)

    token_acc, partition_acc, strict_acc, report, f1, acc = get_acc(logits, labels, is_eval=True, standard_count=standard_count)

    logger.info(report)
    logger.info('token acc:{}%, partition acc:{}%, strict acc:{}%, f1:{}'.format('{:.2f}'.format(token_acc), '{:.2f}'.format(partition_acc), '{:.2f}'.format(strict_acc), '{:.2f}'.format(f1)))
    logger.info('acc : {}'.format(acc))
    return token_acc, partition_acc, strict_acc, f1, acc

# 得到012的acc，以及分段的acc
def get_acc(logits, labels, is_eval=False, standard_count=None):
    acc = None
    label_mask = (labels.sum(dim=-1) != 0)

    logits = logits.argmax(dim=-1)
    labels = labels.argmax(dim=-1)
    token_pred = logits[label_mask].squeeze(dim=-1)
    token_labels = labels[label_mask].squeeze(dim=-1)
    token_acc = torch.sum(token_pred == token_labels) / len(token_labels) * 100

    if is_eval:
        y_true = token_labels.cpu().tolist()
        y_pred = token_pred.cpu().tolist()

        report = classification_report(y_true, y_pred, zero_division=0, digits=4)
        f1 = f1_score(y_true, y_pred, average='weighted') * 100

    partition_num_pred = ((logits == 1) * label_mask).sum(dim=-1)
    partition_num_label = (labels == 1).sum(dim=-1)
    partition_num_acc = torch.sum(partition_num_pred == partition_num_label) / len(partition_num_label) * 100
    if standard_count is not None:
        acc = torch.sum(torch.sum(torch.masked_fill(logits, ~label_mask, 0), dim=-1)  == standard_count)  / len(standard_count) * 100
        # acc = torch.sum(partition_num_pred == standard_count) / len(standard_count) * 100

    strict_num_pred = logits * label_mask.to(torch.int32)
    strict_acc = torch.sum(torch.all(strict_num_pred == labels, dim=-1)) / len(labels) * 100

    if not is_eval:
        return token_acc.item(), partition_num_acc.item(), strict_acc.item()
    else:
        return token_acc.item(), partition_num_acc.item(), strict_acc.item(), report, f1, acc

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
        s = s.replace(' ', '').replace('\x04', '')
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

if __name__ == '__main__':
    main()
    
import os
import torch
import torch.nn.functional as F
from dataset import PartitionV3Dataset, PartitionV3PredictDataset
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
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-output_name', default='test', type=str)
    parser.add_argument('-train_batch_size', default=64, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-eval_batch_size', default=256, type=int) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-max_len', default=256, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch_num', default=20, type=int)
    parser.add_argument('-num_labels', default=5, type=int) # 个数在11及其以上的均视作同一类
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-ffn_adapter_size', default=0, type=int) 
    parser.add_argument('-steps_per_epoch', default=200, type=int) 
    parser.add_argument('-prefix_len', default=0, type=int) 
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-saved_model_path', default=None, type=str)
    parser.add_argument('-r_drop', default='no', type=str)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-predict_data_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data/train/multi_not_split.txt', type=str)
    parser.add_argument('-p', default=0.3, type=float) # 训练时，从标准词中采样的概率
    parser.add_argument('-data_path', default='/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data', type=str) # 数据path
    

    args = parser.parse_args()
    args.r_drop = args.r_drop == 'yes'


    output_path = os.path.join('./output1/Bert_partition_v3', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data = read_parition_data(args.data_path, logger, args)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_ffn_adapter_bert(pretrained_model_path, logger=logger, args=args, model_class=BertAttentionFfnAdapterForTokenClassification)

    bert_model = bert_model.to(args.device)
    
    if args.type == 'train':
        # #准备数据
        train_dataset = PartitionV3Dataset(train_data, bert_tokenizer, args, shuffle=True, while_true=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

        dev_dataset = PartitionV3Dataset(dev_data, bert_tokenizer, args)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset = PartitionV3Dataset(test_data, bert_tokenizer, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)

        #配置optimizer和scheduler
        t_total = args.steps_per_epoch * args.epoch_num
        optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

        evaluate(dev_dataloader, bert_model, logger, args) 
        train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger)
    
    elif args.type == 'evaluate':
        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)
        test_dataset = PartitionV3Dataset(test_data, bert_tokenizer, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)
        evaluate(test_dataloader, bert_model, logger, args) 
    
    elif args.type == 'predict':
        # data_path = '/home/liangming/nas/ml_project/Biye/ThirdChapter/split_data/train/multi_not_split.txt'
        logger.info('load predict data from {}'.format(args.predict_data_path))
        data_list = []
        with open(args.predict_data_path, 'r') as f:
            for line in f:
                data_list.append(line.strip().split('\t'))
            f.close()
        str_list = [x[0] for x in data_list]

        predict_dataset = PartitionV3PredictDataset(str_list, bert_tokenizer, args)
        predict_dataloader = DataLoader(predict_dataset, batch_size=args.eval_batch_size, collate_fn=predict_dataset.collate_fn)

        logger.info('load model from {}'.format(args.saved_model_path))
        checkpoint = torch.load(args.saved_model_path, map_location='cpu')
        bert_model.load_state_dict(checkpoint)
        bert_model = bert_model.to(args.device)

        hard_and_res_list, hard_or_res_list, soft_res_list = predict(bert_model, predict_dataloader, data_list, logger, args)
        res_dict = {
            'ha': hard_and_res_list, 'ho':hard_or_res_list, 'sr':soft_res_list
        }

        parent_dir = os.path.abspath(os.path.join(args.saved_model_path, os.pardir))
        model_base_name = os.path.basename(args.saved_model_path).replace('_model.pth', '')
        res_base_name = os.path.basename(args.predict_data_path)

        for k, v in res_dict.items():
            res_save_path = os.path.join(parent_dir, '{}_{}_{}'.format(model_base_name, k, res_base_name))
            logger.info('save predict result to {}'.format(res_save_path))
            with open(res_save_path, 'w') as f:
                for res in v:
                    f.write('\t'.join(res) + '\n')
                f.close()

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger):
    model.train()
    loss_list = []
    token_acc_list = []
    mask_acc_list = []
    hap_acc_list = []
    hop_acc_list = []
    has_acc_list = []
    hos_acc_list = []
    ss_acc_list = []
    sp_acc_list = []

    best_ha_acc = 0
    best_ho_acc = 0
    best_sr_acc = 0
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
            input_ids, attention_mask, position_ids, labels = batch
            output = model.forward(input_ids, attention_mask, position_ids=position_ids)

            loss = 0
            if args.r_drop:
                output_1 = model.forward(input_ids, attention_mask, position_ids=position_ids)

                # pad_mask = 1则mask掉
                pad_mask = torch.all(labels == 0, dim=-1) 
                loss += compute_kl_loss(output.logits, output_1.logits, pad_mask=pad_mask)

            logits = output.logits

            loss += circle_loss(logits, labels)

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_list.append(loss.item())

            token_acc, mask_acc, hap_acc, has_acc, hop_acc, hos_acc, sp_acc, ss_acc = get_acc(logits, labels)
            token_acc_list.append(token_acc)
            mask_acc_list.append(mask_acc)
            hap_acc_list.append(hap_acc)
            has_acc_list.append(has_acc)
            hop_acc_list.append(hop_acc)
            hos_acc_list.append(hos_acc)
            sp_acc_list.append(sp_acc) 
            ss_acc_list.append(ss_acc) 


            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'loss:{},token acc:{}%,mask acc:{}%, hap_acc:{}%, hop_acc:{}%, sp_acc:{}%'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(token_acc_list) / len(token_acc_list), 2),
                    round(sum(mask_acc_list) / len(mask_acc_list), 2),
                    round(sum(hap_acc_list) / len(hap_acc_list), 2),
                    round(sum(hop_acc_list) / len(hop_acc_list), 2),
                    round(sum(sp_acc_list) / len(sp_acc_list), 2),))
            loss.backward()
            step += 1

            # 每4步累积梯度
            if step % 4 == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        ha_acc, ho_acc, sr_acc = evaluate(dev_dataloader, model, logger, args)
        model.train()

        if ha_acc > best_ha_acc:
            best_ha_acc = ha_acc
            logger.info('save model at ha acc {}'.format(ha_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_hard_and_acc_model.pth')) 

        if ho_acc > best_ho_acc:
            best_ho_acc = ho_acc
            logger.info('save model at ho acc {}'.format(ho_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_hard_or_acc_model.pth')) 

        if sr_acc > best_sr_acc:
            best_sr_acc = sr_acc
            logger.info('save model at sr acc {}'.format(sr_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_soft_acc_model.pth')) 
    

        # token_acc, mask_acc, hap_acc, has_acc, hop_acc, hos_acc, sp_acc, ss_acc = evaluate(dev_dataloader, model, logger, args)

        # model.train()
        # if hap_acc > best_hap_acc:
        #     best_hap_acc = hap_acc 
        #     logger.info('save model at hard_and_parition_acc {}'.format(hap_acc))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_hard_and_parition_acc_model.pth')) 

        # if hop_acc > best_hop_acc:
        #     best_hop_acc = hop_acc 
        #     logger.info('save model at hard_or_parition_acc {}'.format(hop_acc))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_hard_or_parition_acc_model.pth')) 

        # if sp_acc > best_sp_acc:
        #     best_sp_acc = sp_acc 
        #     logger.info('save model at soft_partition_acc {}'.format(sp_acc))
        #     torch.save(model.state_dict(), os.path.join(model_saved_path, 'best_soft_parition_acc_model.pth')) 

    logger.info('#'*20 + 'Evaluate' + '#'*20)
    evaluate(test_dataloader, model, logger, args)


def evaluate(dataloader, model, logger, args):
    model.eval()
    all_logits = []
    all_y_true = []
    all_standard_count =[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, position_ids, labels, standard_count = batch
            output = model.forward(input_ids, attention_mask, position_ids=position_ids)

            all_logits.append(output.logits)
            all_y_true.append(labels)
            all_standard_count.append(standard_count)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_y_true, dim=0)
    standard_counts = torch.cat(all_standard_count, dim=0)

    token_acc, mask_acc, hap_acc, has_acc, hop_acc, hos_acc, sp_acc, ss_acc, ha_acc, ho_acc, sr_acc = get_acc(logits, labels, standard_counts)

    logger.info('token acc:{}%'.format(token_acc))
    logger.info('mask acc:{}%'.format(mask_acc))
    # logger.info('hard and partition acc:{}%'.format(hap_acc))
    # logger.info('hard and strict acc:{}%'.format(has_acc))
    logger.info('hard and acc:{}%'.format(ha_acc))

    # logger.info('hard or partition acc:{}%'.format(hop_acc))
    # logger.info('hard or strict acc:{}%'.format(hos_acc))
    logger.info('hard or acc:{}%'.format(ho_acc))

    # logger.info('soft partition acc:{}%'.format(sp_acc))
    # logger.info('soft strict acc:{}%'.format(ss_acc))
    logger.info('soft acc:{}%'.format(sr_acc))


    return ha_acc, ho_acc, sr_acc

# 得到012的acc，以及分段的acc
def get_acc(logits, labels, standard_counts=None):
    def get_partition_acc(token_logits, token_labels, token_mask, mask_logits, mask_labels, mask_mask, standard_counts=None):
        # 直接使用partition num作为label的准确率
        ha_acc = None
        ho_acc = None
        sr_acc = None

        # token扔掉最后一个字符 
        token_logits = token_logits[:, :-1, :]
        token_mask = token_mask[:, :-1]
        token_labels = token_labels[:, :-1]

        # mask 不要第一个（cls）
        mask_logits = mask_logits[:, 1:, :]
        mask_mask = mask_mask[:, 1:]
        mask_labels = mask_labels[:, 1:]

        partition_num = torch.sum(token_labels == 1, dim=-1)

        # batch_size, s / 2
        token_pred = token_logits.argmax(dim=-1) * token_mask
        mask_pred = mask_logits.argmax(dim=-1) * mask_mask

        # 预测为2的地方为mask为1
        noise_token_mask = (token_pred == 2) 

        # 严格硬投票：token和mask认为是边界时取1
        hard_and_pred = ((token_pred == 1) & (mask_pred == 1)).to(int)
        hard_and_parition_acc = torch.sum(torch.sum(hard_and_pred, dim=-1) == partition_num).item() / len(partition_num) * 100
        hard_and_strict_acc = torch.sum(torch.all(torch.masked_fill(hard_and_pred, noise_token_mask, 2) == token_labels, dim=-1)).item()  / len(token_labels) * 100
        if standard_counts is not None:
            ha_acc = torch.sum(torch.sum(hard_and_pred, dim=-1) == standard_counts).item() / len(standard_counts) * 100
        
        
        # 非严格硬投票：token和mask只要有一个人为是边界则取1
        hard_or_pred = ((token_pred == 1) | (mask_pred == 1)).to(int)
        hard_or_parititon_acc = torch.sum(torch.sum(hard_or_pred, dim=-1) == partition_num).item() / len(partition_num) * 100
        hard_or_strict_acc = torch.sum(torch.all(torch.masked_fill(hard_and_pred, noise_token_mask, 2) == token_labels, dim=-1)).item()  / len(token_labels) * 100
        if standard_counts is not None:
            ho_acc = torch.sum(torch.sum(hard_or_pred, dim=-1) == standard_counts).item() / len(standard_counts) * 100

        # 软投票：token预测非2的时候，将token和mask的logits进行融合
        # 把对应位置的mask logits变为0
        mask_logits = torch.masked_fill(mask_logits, noise_token_mask.unsqueeze(dim=-1), 0)

        soft_logits = mask_logits + token_logits[:, :, :2]
        soft_pred = soft_logits.argmax(dim=-1) * token_mask
        # 把原本是2的地方给补上
        soft_pred = torch.masked_fill(soft_pred, noise_token_mask, 2)
        soft_parition_acc = torch.sum(torch.sum(torch.masked_fill(soft_pred, noise_token_mask, 0), dim=-1) == partition_num).item() / len(partition_num) * 100
        soft_strict_acc = torch.sum(torch.all(torch.masked_fill(soft_pred, noise_token_mask, 2) == token_labels, dim=-1)).item()  / len(token_labels) * 100
        if standard_counts is not None:
            sr_acc = torch.sum(torch.sum(torch.masked_fill(soft_pred, noise_token_mask, 0), dim=-1) == standard_counts).item() / len(standard_counts) * 100

        if standard_counts is None:
            return hard_and_parition_acc, hard_and_strict_acc, hard_or_parititon_acc, hard_or_strict_acc, soft_parition_acc, soft_strict_acc
        else:
            return hard_and_parition_acc, hard_and_strict_acc, hard_or_parititon_acc, hard_or_strict_acc, soft_parition_acc, soft_strict_acc, ha_acc, ho_acc, sr_acc

    def get_token_and_mask_acc(logits, mask, labels):
        pred = logits.argmax(dim=-1)
        
        pred = pred[mask].squeeze(dim=-1)
        labels = labels[mask].squeeze(dim=-1)
        # acc = torch.sum(pred == labels) / len(labels) * 100
        acc = torch.sum(pred == labels).item() / len(labels) * 100

        return acc
    

    # 为1表示有效，0表示无效，包括了token和mask
    # label_mask = (labels.sum(dim=-1) != 0)
    # label_mask = torch.any(labels != 0)

    label_mask = torch.any(labels != 0, dim=-1)


    batch_size, seq_len, num_labels = logits.size()
    # 偶数为是mask，奇数为token, 为1表示为token或mask
    mask_mask, token_mask = label_mask.reshape(batch_size, -1, 2).permute(2, 0, 1)
    # mask_logits: cls,m,m,m,m....
    # token_logits: A,B,C,D,E....
    mask_logits, token_logits = logits.reshape(batch_size, -1, 2, num_labels).permute(2, 0, 1, 3)
    mask_labels, token_labels = labels.reshape(batch_size, -1, 2, num_labels).permute(2, 0, 1, 3)

    # mask: batch_size, seq_len / 2, 2
    # token: batch_size, seq_len / 2, 3
    mask_logits = mask_logits[..., -2:]
    token_logits = token_logits[..., :-2]
    mask_labels = mask_labels[..., -2:].argmax(dim=-1)
    token_labels = token_labels[..., :-2].argmax(dim=-1)

    # token
    token_acc = get_token_and_mask_acc(token_logits, token_mask, token_labels)

    # mask
    mask_acc = get_token_and_mask_acc(mask_logits, mask_mask, mask_labels)

        
    out = (token_acc, mask_acc, ) + get_partition_acc(token_logits, token_labels, token_mask, mask_logits, mask_labels, mask_mask, standard_counts)
    return out

def predict(model, dataloader, data_list, logger, args):
    model.eval()

    # y_pred = []
    pred_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, position_ids = [x.to(args.device) for x in batch] 
            output = model.forward(input_ids, attention_mask, position_ids=position_ids)
            pred_logits.append(output.logits)
    
    return get_res_list(pred_logits, data_list)

def get_res_list(pred_logits, data_list):
    # 根据pred来得到最终
    logits = torch.cat(pred_logits, dim=0)

    batch_size, seq_len, num_labels = logits.size()
    # 偶数为是mask，奇数为token, 为1表示为token或mask
    # mask_logits: cls,m,m,m,m....
    # token_logits: A,B,C,D,E....
    mask_logits, token_logits = logits.reshape(batch_size, -1, 2, num_labels).permute(2, 0, 1, 3)

    # mask: batch_size, seq_len / 2, 2
    # token: batch_size, seq_len / 2, 3
    mask_logits = mask_logits[..., -2:]
    token_logits = token_logits[..., :-2]

    token_logits = token_logits[:, :-1, :]
    mask_logits = mask_logits[:, 1:, :]

    # batch_size, s / 2
    token_pred = token_logits.argmax(dim=-1) 
    mask_pred = mask_logits.argmax(dim=-1)

    # 预测为2的地方为mask为1
    noise_token_mask = (token_pred == 2) 

    # 严格硬投票：token和mask认为是边界时取1
    hard_and_pred = ((token_pred == 1) & (mask_pred == 1)).to(int)
    hard_and_pred = torch.masked_fill(hard_and_pred, noise_token_mask, 2)

    # 非严格硬投票：token和mask只要有一个人为是边界则取1
    hard_or_pred = ((token_pred == 1) | (mask_pred == 1)).to(int)
    hard_or_pred = torch.masked_fill(hard_or_pred, noise_token_mask, 2)

    # 软投票：token预测非2的时候，将token和mask的logits进行融合
    # 把对应位置的mask logits变为0
    mask_logits = torch.masked_fill(mask_logits, noise_token_mask.unsqueeze(dim=-1), 0)

    soft_logits = mask_logits + token_logits[:, :, :2]
    soft_pred = soft_logits.argmax(dim=-1) 
    soft_pred = torch.masked_fill(soft_pred, noise_token_mask, 2)

    hard_and_res_list = decode_label(hard_and_pred, data_list)
    hard_or_res_list = decode_label(hard_or_pred, data_list)
    soft_res_list = decode_label(soft_pred, data_list)
    return hard_and_res_list, hard_or_res_list, soft_res_list

def decode_label(y_pred, data_list):
    assert len(y_pred) == len(data_list)
    res_list = []
    for pred_list, data in zip(y_pred, data_list):
        s = data[0]
        s = s.replace(' ', '').replace('\x04', '')
        res = ''

        # 直接从0开始，pred的第0位已经是s的第一个字符
        for i in range(len(s)):
            if pred_list[i] == 2:
                continue
            elif pred_list[i] == 0:
                res += s[i]
            elif pred_list[i] == 1:
                res += s[i] + '###'
        res_list.append(data + [res])

    return res_list 


if __name__ == '__main__':
    main()
    
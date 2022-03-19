import os
import torch
from dataset import FinaDataset
from log import Logger
from argparse import ArgumentParser
from utils import print_args, get_optimizer_and_scheduler, read_fina_sim_data, load_ffn_adapter_bert
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
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
    parser.add_argument('-epoch_num', default=10, type=int)
    parser.add_argument('-num_labels', default=2, type=int) # 个数在11及其以上的均视作同一类
    parser.add_argument('-num_workers', default=4, type=int) 
    parser.add_argument('-ffn_adapter_size', default=64, type=int) 
    parser.add_argument('-prefix_len', default=64, type=int) 


    args = parser.parse_args()
    output_path = os.path.join('./output/ffn_adapter', args.output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #定义log参数
    logger = Logger(output_path,'main').logger

    #打印args
    print_args(args, logger)

    #读取数据
    data_path = './fina_sim'
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    train_data, dev_data, test_data = read_fina_sim_data(data_path, logger, args)

    #读取模型
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12'
    bert_model, bert_tokenizer, bert_config = load_ffn_adapter_bert(pretrained_model_path, logger=logger, args=args)
    bert_model = bert_model.to(args.device)
    
    # #准备数据
    train_dataset = FinaDataset(train_data, bert_tokenizer, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers, shuffle=True)

    dev_dataset = FinaDataset(dev_data, bert_tokenizer, args)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.collate_fn, num_workers=args.num_workers)

    test_dataset = FinaDataset(test_data, bert_tokenizer, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers)

    #配置optimizer和scheduler
    t_total = len(train_dataloader) * args.epoch_num
    optimizer, scheduler = get_optimizer_and_scheduler(bert_model, t_total, args.lr, 0)

    train(bert_model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, output_path, logger)

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

    for epoch in range(args.epoch_num):
        logger.info('#'*20 + 'Epoch{}'.format(epoch + 1) + '#'*20)
        iteration = tqdm(train_dataloader, desc='Training')
        model.zero_grad()
        for batch in iteration:
            batch = [x.to(args.device) for x in batch]
            input_ids, attention_mask, token_type_ids, labels = batch
            output = model.forward(input_ids, attention_mask, token_type_ids, labels=labels)
            loss = output.loss
            logits = output.logits

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_list.append(loss.item())
            batch_pred = logits.argmax(dim=-1) 
            acc_list.append(torch.sum((batch_pred == labels).to(torch.int32)).item() / len(batch_pred) * 100)


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
            input_ids, attention_mask, token_type_ids, labels = batch
            output = model.forward(input_ids, attention_mask, token_type_ids)
            batch_pred = torch.argmax(output.logits, dim=-1).cpu().tolist()
            y_pred += batch_pred
            y_true += labels.cpu().tolist()

    report = classification_report(y_true, y_pred, zero_division=0, digits=4)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    logger.info(report)
    return acc, f1

if __name__ == '__main__':
    main()
    
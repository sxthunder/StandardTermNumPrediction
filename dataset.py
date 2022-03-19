from posixpath import split
import torch
import random
import math
from torch.utils.data import IterableDataset
from transformers.utils.dummy_pt_objects import ProphetNetForConditionalGeneration

# class ClaDataset():
#     def __init__(self, data_list, tokenizer, args):
#         self.data_list = data_list
#         self.tokenizer = tokenizer
#         self.device = args.device
#         self.max_len = args.max_len
#         self.num_labels = args.num_labels

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, index):
#         raw_text, label = self.data_list[index]
#         label -= 1

#         raw_text = raw_text[:self.max_len - 2]

#         token_result = self.tokenizer.encode_plus(raw_text, padding='max_length', max_length=self.max_len)
#         return token_result['input_ids'], token_result['attention_mask'], label

#     def collate_fn(self, batch):
#         return [torch.tensor(x) for x in list(zip(*batch))]

class ClaDataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.shuffle = shuffle
        self.while_true = while_true

    def __iter__(self):
        # 死循环，训练时设定每个Epoch走多少步
        # self.idx = 0
        idx = 0

        # 整个多进程试试
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id=0
            num_workers=1

        else:
            worker_id=worker_info.id
            num_workers=worker_info.num_workers

        length_per_work = math.ceil(len(self.data_list) / num_workers)
        data_list = self.data_list[worker_id * length_per_work: (worker_id + 1) * length_per_work]

        if self.shuffle:
            random.shuffle(data_list)

        while True:
            if idx == len(data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration

            raw_text, label = data_list[idx]
            idx = (idx + 1)

            label -= 1

            raw_text = raw_text[:self.max_len - 2]

            token_result = self.tokenizer.encode_plus(raw_text, padding='max_length', max_length=self.max_len)
            yield token_result['input_ids'], token_result['attention_mask'], label

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))]

class FinaDataset():
    def __init__(self, data_list, tokenizer, args):
        self.x, self.y = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.num_labels = args.num_labels

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        s1, s2 = self.x[index]
        label = int(self.y[index]) 

        text_max_len = self.max_len - 3
        if len(s1) + len(s2) > text_max_len:
            if len(s1) > len(s2):
                s1 = s1[:text_max_len - len(s2)]
            else:
                s2 = s2[:text_max_len - len(s1)]

        token_result = self.tokenizer.encode_plus(s1, s2, padding='max_length', max_length=self.max_len)
        return token_result['input_ids'], token_result['attention_mask'], token_result['token_type_ids'], label

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))]

class PartitionDataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            # print(idx, len(self.data_list))
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration
                
            s = self.data_list[idx]
            idx += 1

            labels = []
            for i, char in enumerate(s):
                if i < len(s) - 1 and s[i + 1] == ',':
                    labels.append(1)
                elif char != ',':
                    labels.append(-1)
            s = s.replace(',', '')

            token_result = self.tokenizer.encode_plus(s, padding='max_length', max_length=self.max_len)
            labels += [0 for _ in range(self.max_len - len(labels))]

            yield token_result['input_ids'], token_result['attention_mask'], labels

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))]

class PartitionPredictDataset():
    def __init__(self, data_list, tokenizer, args):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.device = args.device
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        s = self.data_list[idx]

        # token_result = self.tokenizer.encode_plus(s, padding='max_length', max_length=self.max_len)
        # return token_result['input_ids'], token_result['attention_mask'] 

        input_ids, attention_mask = self.tokenize(s)
        return input_ids, attention_mask


    def tokenize(self, sen):
        input_ids = []
        for s in sen:
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])

        input_ids = [self.cls] + input_ids + [self.sep]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]

        return input_ids, attention_mask

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))]

# 特殊token标签为2，词结尾标签为1，词中间标签为0
class PartitionV2Dataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.p = args.p
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id

        # 从data_list中获取所有使用的标准词
        self.standard_list = list(set(sum([x[1].split('##') for x in self.data_list],[])))

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration
                

            # 从已经分好的词中采样(只有train中是这样)
            if self.while_true and random.random() < self.p:
                # 有可能从标准词中采样
                if random.random() < 0.5:
                    sampled_sen_list = random.sample(self.standard_list, random.randint(2, 5))
                    split_sen = ','.join(sampled_sen_list)
                    raw_sen = ''.join(sampled_sen_list)
                else:
                    split_sen = random.choice(self.data_list)[-1]

                    # 把split_sen中的,替换成？
                    raw_sen = split_sen.replace(',', '')

            # 正常读取数据
            else:
                raw_sen, standard_sen, _, split_sen = self.data_list[idx]
                idx += 1
                standard_count = len(standard_sen.split('##')) 

            raw_sen = raw_sen.replace(' ', '').replace('\x04', '')
            raw_sen = raw_sen[:self.max_len - 2]

            # 先假设其都是特殊token, 第一个cls取0不算loss
            labels = [[0, 0, 0]] + [[-1, -1, 1] for _ in range(len(raw_sen))]
            for s in split_sen.split(','):
                try:
                    split_idx = raw_sen.index(s)
                except:
                    pass

                for i in range(split_idx, split_idx + len(s) - 1):
                    labels[i + 1] = [1, -1, -1]
                labels[split_idx + len(s)] = [-1, 1, -1]

            input_ids, attention_mask = self.tokenize(raw_sen)
            labels += [[0, 0, 0] for _ in range(self.max_len - len(labels))]

            if self.while_true:
                yield input_ids, attention_mask, labels
            else:
                yield input_ids, attention_mask, labels, standard_count


    def tokenize(self, sen):
        input_ids = []
        for s in sen:
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])

        input_ids = [self.cls] + input_ids + [self.sep]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]

        return input_ids, attention_mask


    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

# 基于Partition的分类Dataset
class PartitionClaDataset(IterableDataset):
    def __init__(self, data_list, standard_to_code, code_to_idx, tokenizer, logger, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.standard_to_code = standard_to_code
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.use_which_partition = args.use_which_partition
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.p = args.p
        self.code_to_idx = code_to_idx

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration

            # 整个多进程试试
            worker_info = torch.utils.data.get_worker_info()
            
            if worker_info is None:
                worker_id=0
                num_workers=1

            else:
                worker_id=worker_info.id
                num_workers=worker_info.num_workers

            length_per_work = math.ceil(len(self.data_list) / num_workers)
            data_list = self.data_list[worker_id * length_per_work: (worker_id + 1) * length_per_work]

            # 使用组合数据(只有train中是这样)
            if self.while_true and random.random() < self.p:
                while True:
                    _, standard_sen_1, _, rule_split_sen_1, model_split_sen_1 = random.choice(data_list)
                    _, standard_sen_2, _, rule_split_sen_2, model_split_sen_2 = random.choice(data_list)

                    standard_sen = standard_sen_1 + '##' + standard_sen_2
                    rule_split_sen = rule_split_sen_1 + ',' + rule_split_sen_2
                    model_split_sen = model_split_sen_1 + '###' + model_split_sen_2

                    # 如果所有的加起来小雨max_len则返回（模拟最坏情况 model partition len + model partition 个数（sep） + 1（sep） + standard_sen 个数（生成token)
                    if sum([len(x) for x in model_split_sen.split('###')]) + len(model_split_sen) + 1 + len(standard_sen.split("##"))<= self.max_len:
                        break
                    
            else:
                raw_sen, standard_sen, label_num, rule_split_sen, model_split_sen = data_list[idx]
                idx += 1

            # 不用parition数据
            if self.use_which_partition == 0:
                split_sen_list = [raw_sen]

            # 使用rule partition数据
            elif self.use_which_partition == 1:
                split_sen_list = rule_split_sen.split(',')
            
            else:
                split_sen_list =  model_split_sen.split('###')
            split_sen_list = [x for x in split_sen_list if x != '']

            split_code_id_list = []
            for x in standard_sen.split('##'):
                split_code_id_list.append(self.code_to_idx[self.standard_to_code[x]])

            input_ids = [self.cls]
            for split_sen in split_sen_list:
                input_ids += self.tokenize(split_sen) + [self.sep] 


            # 如果是训练
            if self.while_true:
                attention_mask = []
                for i in range(len(input_ids) + len(split_code_id_list)):
                    if i < len(input_ids):
                        attention_mask.append([1 for _ in range(len(input_ids))] + [0 for _ in range(self.max_len - len(input_ids))])
                    else:
                        attention_mask.append([1 for _ in range(i + 1)] + [0 for _ in range(self.max_len - i - 1)])
                split_code_id_list.append(self.sep)
                label_ids = [-100 for _ in range(len(input_ids) - 1)]

                input_ids += split_code_id_list[:-1]
                label_ids += split_code_id_list

                label_ids += [-100 for _ in range(self.max_len - len(label_ids))]

                # 训练时才需要padding
                input_ids += [0 for _ in range(self.max_len - len(input_ids))]
                attention_mask += [[0 for _ in range(self.max_len)] for _ in range(self.max_len - len(attention_mask))]
            
            else:
                attention_mask = [1 for _ in range(len(input_ids))]

                label_ids = split_code_id_list


            yield input_ids, attention_mask, label_ids
            
    def tokenize(self, sen):
        input_ids = []
        for s in sen:
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])

        return input_ids

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

# 特殊token标签为2，词结尾标签为1，词中间标签为0, 每两个词之间加一个[MASK], 用以预测是否在这两个词之间分割，unused1 表示不分割，unused2 表示分割
class PartitionV3Dataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.p = args.p
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.mask = self.tokenizer.mask_token_id

        # 从data_list中获取所有使用的标准词
        self.standard_list = list(set(sum([x[1].split('##') for x in self.data_list],[])))

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration
                

            # 从已经分好的词中采样(只有train中是这样)
            if self.while_true and random.random() < self.p:
                # 一次性选择多个标准词，raw_sen直接将拼在一起 
                if random.random() < 0.5:
                    sampled_sen_list = random.sample(self.standard_list, random.randint(2, 7))
                    split_sen = ','.join(sampled_sen_list)
                    raw_sen = ''.join(sampled_sen_list)

                # 随机从data_list中选择一个，raw_sen直接拼在一起的
                else:
                    split_sen = random.choice(self.data_list)[-1]

                    # 把split_sen中的,替换成？
                    raw_sen = split_sen.replace(',', '')

            # 正常读取数据
            else:
                raw_sen, standard_sen, _, split_sen = self.data_list[idx]
                idx += 1
                standard_count = len(standard_sen.split('##'))

            raw_sen = raw_sen.replace(' ', '').replace('\x04', '')
            raw_sen = raw_sen[:math.ceil((self.max_len - 1) / 2)]

            # 先假设其都是特殊token, 第一个cls取0不算loss
            # 0-2 用作token的label标签 0表示词中间，1表示词结尾，2表示词中间
            # 3-4 用作mask的标签 3表示不分割，4表示分割
            labels = [[0, 0, 0, 0, 0]] + [[-1, -1, 1, 0, 0] for _ in range(len(raw_sen))]
            for s in split_sen.split(','):
                try:
                    split_idx = raw_sen.index(s)
                except:
                    pass

                for i in range(split_idx, split_idx + len(s) - 1):
                    labels[i + 1] = [1, -1, -1, 0, 0]
                labels[split_idx + len(s)] = [-1, 1, -1, 0, 0]

            # 在label之间插入
            new_labels = []
            new_labels.append(labels[0])
            for i in range(1, len(labels)):
            # for i in range(1, len(labels) -1):
                new_labels.append(labels[i])
                if labels[i][1] == 1:
                    new_labels.append([0, 0, 0, -1, 1])
                else:
                    new_labels.append([0, 0, 0, 1, -1])
            # new_labels.append(labels[-1])
            assert len(new_labels) == 2 * len(labels) - 1
            labels = new_labels

            input_ids, attention_mask, position_ids = self.tokenize(raw_sen)
            labels += [[0, 0, 0, 0, 0] for _ in range(self.max_len - len(labels))]

            if self.while_true:
                yield input_ids, attention_mask, position_ids, labels
            else:
                yield input_ids, attention_mask, position_ids, labels, standard_count


    def tokenize(self, sen):
        input_ids = []
        position_ids = []
        for idx, s in enumerate(sen):
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])
            input_ids.append(self.mask)

            # token和mask有相同的position
            position_ids.append(idx + 1)
            position_ids.append(idx + 1)

        # 添加cls
        # input_ids = [self.cls] + input_ids
        # position_ids = [0] + position_ids

        # 最后一个token不加mask 将最后一个mask改为sep
        # position_ids[-1] = position_ids[-1] + 1
        # input_ids[-1] = self.sep

        # 添加cls和sep
        input_ids = [self.cls] + input_ids + [self.sep]
        position_ids = [0] + position_ids + [position_ids[-1] + 1]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]
        position_ids += [0 for _ in range(self.max_len - len(position_ids))]

        return input_ids, attention_mask, position_ids


    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

class PartitionV3PredictDataset():
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.p = args.p
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.mask = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        raw_sen = self.data_list[index]
        raw_sen = raw_sen.replace(' ', '').replace('\x04', '')
        raw_sen = raw_sen[:math.ceil((self.max_len - 1) / 2)]

        input_ids, attention_mask, position_ids = self.tokenize(raw_sen)

        return input_ids, attention_mask, position_ids

    def tokenize(self, sen):
        input_ids = []
        position_ids = []
        for idx, s in enumerate(sen):
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])
            input_ids.append(self.mask)

            # token和mask有相同的position
            position_ids.append(idx + 1)
            position_ids.append(idx + 1)

        # 添加cls
        # input_ids = [self.cls] + input_ids
        # position_ids = [0] + position_ids

        # 最后一个token不加mask 将最后一个mask改为sep
        # position_ids[-1] = position_ids[-1] + 1
        # input_ids[-1] = self.sep

        # 添加cls和sep
        input_ids = [self.cls] + input_ids + [self.sep]
        position_ids = [0] + position_ids + [position_ids[-1] + 1]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]
        position_ids += [0 for _ in range(self.max_len - len(position_ids))]

        return input_ids, attention_mask, position_ids

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

# partition cla rank的dataset
class PartitionClaRankDataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.mask = self.tokenizer.mask_token_id
        self.use_which_partition = args.rank_use_which_partition

        max_code = 26 if args.code_metric == 'alphabet' else 22
        self.code_id_list = [i for i in range(max_code)]

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration
            
            raw_sen, rule_split_sen, model_split_sen, generate_pred_sen, label = self.data_list[idx]
            idx += 1
            
            # 选择使用什么做输入
            if self.use_which_partition == 0:
                input_sen = raw_sen
            elif self.use_which_partition == 1:
                input_sen = '[SEP]'.join(rule_split_sen.split(','))
            elif self.use_which_partition == 2:
                input_sen = '[SEP]'.join([x for x in model_split_sen.split(',') if x != ''])
            else:
                raise Exception('use which partition not supported')

            generate_pred_list = [eval(x) for x in generate_pred_sen.split('+++')] 
            label = eval(label) if isinstance(label, str) else label

            right_count = sum([len(x[0]) == len(label) for x in generate_pred_list]) # 看看当前样本有多少正确的

            origin_input_ids = self.tokenizer.encode(input_sen)
            for generate_pred in generate_pred_list:
                pred_ids, pred_scores = generate_pred
                count_label = int(len(label) == len(pred_ids)) 
                # 如果该样本下有80%正确，按照一定概率给他改错(仅在训练过程中出现)
                if self.while_true and right_count >= 4 and count_label == 1 and random.random() < 0.4:
                    pred_ids = self.noise_pred_ids(pred_ids)
                    count_label = int(len(label) == len(pred_ids)) 

                input_ids = origin_input_ids + pred_ids + [self.sep]
                attention_mask = [1 for _ in range(len(input_ids))]

                input_ids += [0 for _ in range(self.max_len - len(input_ids))]
                attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]

                yield input_ids, attention_mask, pred_scores[0], count_label 

    def noise_pred_ids(self, pred_ids):
        if len(pred_ids) == 0:
            pred_ids += [random.choice(self.code_id_list) for _ in range(random.randint(1, 3))] 
        else:
            # 50%概率增加
            if random.random() < 0.5:
                pred_ids += [random.choice(self.code_id_list) for _ in range(random.randint(1, 3))] 

            # 50%概率减少 
            else:
                pred_ids = random.sample(pred_ids, max(len(pred_ids) - random.randint(1, 3), len(pred_ids) - 1))
        return pred_ids


    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

# 每两个词之间加一个[MASK], 用以预测是否在这两个词之间分割，unused1 表示不分割，unused2 表示分割
class PartitionV1Dataset(IterableDataset):
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.p = args.p
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.mask = self.tokenizer.mask_token_id

        # 从data_list中获取所有使用的标准词
        self.standard_list = list(set(sum([x[1].split('##') for x in self.data_list],[])))

        if self.shuffle:
            random.shuffle(self.data_list)

    def __iter__(self):
        idx = 0
        while True:
            if idx == len(self.data_list):
                if self.while_true:
                    idx = 0
                else:
                    raise StopIteration
                

            # 从已经分好的词中采样(只有train中是这样)
            if self.while_true and random.random() < self.p:
                # 一次性选择多个标准词，raw_sen直接将拼在一起 
                if random.random() < 0.5:
                    sampled_sen_list = random.sample(self.standard_list, random.randint(2, 7))
                    split_sen = ','.join(sampled_sen_list)
                    raw_sen = ''.join(sampled_sen_list)

                # 随机从data_list中选择一个，raw_sen直接拼在一起的
                else:
                    split_sen = random.choice(self.data_list)[-1]

                    # 把split_sen中的,替换成？
                    raw_sen = split_sen.replace(',', '')

            # 正常读取数据
            else:
                raw_sen, standard_sen, _, split_sen = self.data_list[idx]
                idx += 1
                standard_count = len(standard_sen.split('##'))

            raw_sen = raw_sen.replace(' ', '').replace('\x04', '')
            raw_sen = raw_sen[:math.ceil((self.max_len - 1) / 2)]

            # 先假设其都是特殊token, 第一个cls取0不算loss
            # 第0位表示不分开，第1位表示分开
            # 正为1，负位-1，不算为0
            split_idx_list = []
            for s in split_sen.split(','):
                try:
                    split_idx = raw_sen.index(s)
                except:
                    pass
                split_idx_list.append(split_idx + len(s) - 1)

            # 在label之间插入
            new_labels = [[0, 0]]
            for i in range(len(raw_sen)):
            # for i in range(1, len(labels) -1):
                new_labels.append([0, 0])
                if i in split_idx_list:
                    new_labels.append([-1, 1])
                else:
                    new_labels.append([1, -1])
            labels = new_labels

            input_ids, attention_mask, position_ids = self.tokenize(raw_sen)
            labels += [[0, 0] for _ in range(self.max_len - len(labels))]

            if self.while_true:
                yield input_ids, attention_mask, position_ids, labels
            else:
                yield input_ids, attention_mask, position_ids, labels, standard_count

    def tokenize(self, sen):
        input_ids = []
        position_ids = []
        for idx, s in enumerate(sen):
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])
            input_ids.append(self.mask)

            # token和mask有相同的position
            position_ids.append(idx + 1)
            position_ids.append(idx + 1)

        # 添加cls和sep
        input_ids = [self.cls] + input_ids + [self.sep]
        position_ids = [0] + position_ids + [position_ids[-1] + 1]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]
        position_ids += [0 for _ in range(self.max_len - len(position_ids))]

        return input_ids, attention_mask, position_ids


    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 

class PartitionV1PredictDataset():
    def __init__(self, data_list, tokenizer, args, shuffle=False, while_true=False):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_len = args.max_len
        self.while_true = while_true
        self.shuffle = shuffle
        self.p = args.p
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id
        self.mask = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        raw_sen = self.data_list[index]
        raw_sen = raw_sen.replace(' ', '').replace('\x04', '')
        raw_sen = raw_sen[:math.ceil((self.max_len - 1) / 2)]

        input_ids, attention_mask, position_ids = self.tokenize(raw_sen)

        return input_ids, attention_mask, position_ids

    def tokenize(self, sen):
        input_ids = []
        position_ids = []
        for idx, s in enumerate(sen):
            token_res = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_res) == 0:
                input_ids.append(self.unk)
            else:
                input_ids.append(token_res[0])
            input_ids.append(self.mask)

            # token和mask有相同的position
            position_ids.append(idx + 1)
            position_ids.append(idx + 1)

        # 添加cls
        # input_ids = [self.cls] + input_ids
        # position_ids = [0] + position_ids

        # 最后一个token不加mask 将最后一个mask改为sep
        # position_ids[-1] = position_ids[-1] + 1
        # input_ids[-1] = self.sep

        # 添加cls和sep
        input_ids = [self.cls] + input_ids + [self.sep]
        position_ids = [0] + position_ids + [position_ids[-1] + 1]

        attention_mask = [1] * len(input_ids)

        input_ids += [0 for _ in range(self.max_len - len(input_ids))]
        attention_mask += [0 for _ in range(self.max_len - len(attention_mask))]
        position_ids += [0 for _ in range(self.max_len - len(position_ids))]

        return input_ids, attention_mask, position_ids

    def collate_fn(self, batch):
        return [torch.tensor(x) for x in list(zip(*batch))] 
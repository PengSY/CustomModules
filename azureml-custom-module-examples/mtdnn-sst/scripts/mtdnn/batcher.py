# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import json
import torch
import random
import pandas as pd

UNK_ID = 100
BOS_ID = 101


class BatchGen:
    def __init__(self,
                 data,
                 batch_size=32,
                 gpu=True,
                 is_train=True,
                 maxlen=128,
                 dropout_w=0.005,
                 do_batch=True,
                 weighted_on=False,
                 soft_label=False):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.weighted_on = weighted_on
        self.data = data
        self.pairwise_size = 1
        # soft label used for knowledge distillation
        self.soft_label_on = soft_label
        if do_batch:
            if is_train:
                indices = list(range(len(self.data)))
                random.shuffle(indices)
                data = [self.data[i] for i in indices]
            self.data = BatchGen.make_baches(data, batch_size)
        self.offset = 0
        self.dropout_w = dropout_w

    @staticmethod
    def make_baches(data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    @staticmethod
    def load(path, is_train=True, maxlen=128, factor=1.0, pairwise=False):
        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                if is_train:
                    if pairwise and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (not pairwise) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
            return data

    @staticmethod
    def load_parquet(path, is_train=True, maxlen=128, factor=1.0, pairwise=False):
        df = pd.read_parquet(path, engine="pyarrow")
        return BatchGen.load_dataframe(df,is_train,maxlen,factor,pairwise)

    @staticmethod
    def load_dataframe(df, is_train=True, maxlen=128, factor=1.0, pairwise=False):
        data = []
        df['token_id']=df['token_id'].apply(lambda x:eval(x))
        df['type_id']=df['type_id'].apply(lambda x:eval(x))
        for _, row in df.iterrows():
            row['factor'] = factor
            if is_train:
                if pairwise and (len(row['token_id'][0]) > maxlen or len(row['token_id'][1]) > maxlen):
                    continue
                if (not pairwise) and (len(row['token_id']) > maxlen):
                    continue
            data.append(row)
        return data


    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else:
            return arr

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(non_blocking=True)
        return v

    @staticmethod
    def todevice(v, device):
        v = v.to(device)
        return v

    def rebacth(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label': sample['label'],
                                 'true_label': olab})
        return newbatch

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            batch_size = len(batch)
            tok_len = max(len(x['token_id']) for x in batch)

            token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)

            for i, sample in enumerate(batch):
                select_len = min(len(sample['token_id']), tok_len)
                tok = sample['token_id']
                if self.is_train:
                    tok = self.__random_select__(tok)
                token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
                type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
                masks[i, :select_len] = torch.LongTensor([1] * select_len)
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2
            }
            batch_data = [token_ids, type_ids, masks]
            current_idx = 3
            valid_input_len = 3

            if self.is_train:
                labels = [sample['label'] for sample in batch]
                batch_data.append(torch.LongTensor(labels))
                batch_info['label'] = current_idx
                current_idx += 1
                # soft label generated by ensemble models for knowledge distillation
                if self.soft_label_on and (batch[0].get('softlabel', None) is not None):
                    sortlabels = [sample['softlabel'] for sample in batch]
                    sortlabels = torch.FloatTensor(sortlabels)
                    batch_info['soft_label'] = self.patch(sortlabels.pin_memory()) if self.gpu else sortlabels

            if self.gpu:
                for i, item in enumerate(batch_data):
                    batch_data[i] = self.patch(item.pin_memory())

            # meta 
            batch_info['uids'] = [sample['uid'] for sample in batch]
            batch_info['input_len'] = valid_input_len
            batch_info['pairwise_size'] = self.pairwise_size
            if not self.is_train:
                labels = [sample['label'] for sample in batch]
                batch_info['label'] = labels
            self.offset += 1
            yield batch_info, batch_data

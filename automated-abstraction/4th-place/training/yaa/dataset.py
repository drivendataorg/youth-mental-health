import types

import os, sys, logging
import resource
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from glob import glob
from functools import partial
from itertools import islice
import re
import math
import util
import types
#import albumentations as A
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import RandomSampler, default_collate
from torch.utils.data.distributed import DistributedSampler
import psutil
import ppts


logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DatasetMix():
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None):
        self.cfg = cfg
        self.data_type = data_type
        self.tokenizer=tokenizer
        self.model_config = model_config
        with util.timer('preprocess'):
            self.data = self.preprocess_data(data)

        if self.data_type=='train' and (self.cfg.aug_text>0 or self.cfg.aug_text2>0):
            self.gen_data = util.load_gen_data(cfg)

        if self.data_type=='train' and self.cfg.semi_ratio>0:
            semi_data = self.load_semi_data().to_records(index=False)
            yaa = util.load_yaa_train(self.cfg)
            uid2ind = dict(zip(yaa.uid.values, range(len(yaa))))
            inds = set([uid2ind[rec.uid] for rec in self.data])
            num = len(semi_data)
            semi_data = [rec for rec in semi_data if all([int(ind) in inds for ind in rec.uid.split("_")])]
            logger.info('semi data:%s, %s', num, len(semi_data))
            self.semi_data = semi_data
        if not self.cfg.is_classify:
            self.yid, self.nid = self.tokenizer.convert_tokens_to_ids(['Yes', 'No'])
            if 'phi' in self.cfg.model_name or 'qw' in self.cfg.model_name or 'gemma' in self.cfg.model_name:
                tokens = ['Yes', 'No'] + [chr(ord('A') + i) for i in range(12)]
            else:
                tokens = ['Yes', 'No'] + [str(i) for i in range(12)]
            ids  = self.tokenizer.convert_tokens_to_ids(tokens)
            self.token2id = dict(zip(tokens, ids))
            self.tokids = ids


    def load_semi_data(self):
        dfs = []
        for fpath in self.cfg.semi_fpath.split(" "):
            dfs.append(util.load_dump(fpath))
        df = pd.concat(dfs)
        if self.cfg.n_token>0:
            df = df[df.apply(lambda x: (len(x.NarrativeLE.split())+len(x.NarrativeCME.split()))<=self.cfg.n_token, axis=1)]
        return df

    def __len__(self):
        if self.data_type=='train':
            num = len(self.data)
            if self.cfg.semi_ratio>0:
                num += int(num*self.cfg.semi_ratio)
        else:
            num = len(self.data)
        if not self.cfg.is_classify:
            num = num * len(self.cfg.train_cols)
        return num

    def __getitem__(self, index):
        item = self.getitem(index)
        return item

    def preprocess_data(self, data):
        #data['NarrativeLE'] = 'law enforcement report:' + data.NarrativeLE
        #data['NarrativeCME'] = 'coroner/medical examiner report:' + data.NarrativeCME
        if self.data_type=='test' and self.cfg.is_oof:
            fpath = f"{self.cfg.data_dir}/{self.cfg.model_name}_KF{self.cfg.kfid}/pred_train.dump"
            logger.info('load oof from %s', fpath)
            oof = util.load_dump(fpath)
            uids = set(oof.uid.values)
            data = data.to_records(index=False)
            data = [rec for rec in data if all([uid in uids for uid in rec.uid.split("_")])]
        else:
            data = data.to_records(index=False)
        if self.data_type!='train' and self.cfg.sort_seq:
            data = sorted(data, key=lambda x: len(x.NarrativeLE.split())+len(x.NarrativeCME.split()), reverse=True)
        logger.info("num of data:%s, %s", self.data_type, len(data))
        return data

    def get_labels(self, rec, all_cols=False):
        weights = None
        if self.cfg.train_cols is None or all_cols:
            labels = [rec[l] for l in util.binary_labels]
            if self.cfg.use_weight:
                weights = [2*(1-2*util.binary_weight[l])*rec[l] + 2*util.binary_weight[l] for l in util.binary_labels]
            if self.data_type=='train' and self.cfg.semi_ratio>0:
                if rec.src.startswith('semi'):
                    labels.extend(rec.InjuryLocationType)
                    labels.extend(rec.WeaponType1)
                else:
                    label1 = np.zeros([util.n_InjuryLocationType])
                    label1[rec.InjuryLocationType-1] = 1
                    label2 = np.zeros([util.n_WeaponType1])
                    label2[rec.WeaponType1-1] = 1
                    labels.extend(label1)
                    labels.extend(label2)
            else:
                labels.append(rec.InjuryLocationType - 1)
                labels.append(rec.WeaponType1 - 1)
        else:
            labels = [rec[l] for l in self.cfg.train_cols]
            if self.cfg.use_weight:
                weights = [2*(1 - 2 * util.binary_weight[l]) * rec[l] + 2*util.binary_weight[l] for l in self.cfg.train_cols]
        return labels, weights

    def get_msgs(self, rec, col):
        t1, t2 = self._get_text(rec)
        if self.cfg.use_ppt2:
            r1, r2 = t1, t2
            ppt = ppts.PPT2
        else:
            ppt = ppts.PPT
            assert self.cfg.aug_order == 0
            r1 = f"law enforcement report\n{t1}"
            r2 = f"coroner/medical examiner report\n{t2}"
            if self.data_type=='train' and self.cfg.aug_order2>0 and np.random.rand()<self.cfg.aug_order2:
                r1, r2 = r2, r1
        if ('phi' in self.cfg.model_name or 'qw' in self.cfg.model_name or 'gemma' in self.cfg.model_name) and col in  ["InjuryLocationType", "WeaponType1"]:
            q = getattr(ppts, f'Q_{col}_abc') + ' Yes or No?'
        else:
            q = getattr(ppts, f'Q_{col}') + ' Yes or No?'
        ppt = ppt.format(question=q, r1=r1, r2=r2)
        if 'gemma' in self.cfg.model_name or 'qw' in self.cfg.model_name:
            messages = [
                {"role": "user", "content": "You are an expert to analyze suicide reports. " + ppt},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an expert to analyze suicide reports."},
                {"role": "user", "content": ppt},
            ]
        return messages

    def get_ar_input_ids(self, rec, index):
        col = self.cfg.train_cols[index%len(self.cfg.train_cols)]
        msgs = self.get_msgs(rec, col)
        input_ids = self.tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if self.data_type!='test':
            if rec.src.startswith('semi'):
                output_id = 0
            else:
                if col in ["WeaponType1", "InjuryLocationType"]:
                    if 'phi' in self.cfg.model_name or 'qw' in self.cfg.model_name or 'gemma' in self.cfg.model_name:
                        output_id = self.token2id[chr(ord('A')+(rec[col] - 1))]
                    else:
                        output_id = self.token2id[str(rec[col]-1)]
                else:
                    output_id = self.token2id["Yes"] if rec[col] == 1 else self.token2id["No"]
            #if rec.src.startsiwth('semi'):
            if 1==2:
                labels, _ = self.get_labels(rec, all_cols=True)
            else:
                if self.cfg.is_eval:
                    labels = [-100] * len(input_ids)
                else:
                    input_ids.append(output_id)
                    labels = [-100]*(len(input_ids)-1) + [output_id]
        else:
            labels = None
        return input_ids, labels, None, None

    def sample_rec(self, rec):
        for i in range(20):
            rec2 = self.data[np.random.randint(len(self.data))]
            if rec2.uid!=rec.uid:
                break
        return rec2

    def _get_text(self, rec):
        t1, t2 = rec.NarrativeLE, rec.NarrativeCME
        if self.data_type=='train' and self.cfg.aug_text2>0 and rec.src.startswith('yaa'):
            assert self.cfg.aug_text==0
            gen_texts = self.gen_data[rec.uid]
            if np.random.rand()<self.cfg.aug_text2:
                if np.random.rand()<0.5:
                    t1s = gen_texts[np.random.randint(len(gen_texts))][0]
                    if len(t1s)>0:
                        t1 = t1s[np.random.randint(len(t1s))]
                else:
                    t2s = gen_texts[np.random.randint(len(gen_texts))][1]
                    if len(t2s)>0:
                        t2 = t2s[np.random.randint(len(t2s))]
        if self.data_type=='train' and self.cfg.aug_text>0 and rec.src.startswith('yaa'):
            assert self.cfg.aug_text2==0
            gen_texts = self.gen_data[rec.uid]
            if np.random.rand()<self.cfg.aug_text:
                t1s = gen_texts[np.random.randint(len(gen_texts))][0]
                if len(t1s)>0:
                    t1 = t1s[np.random.randint(len(t1s))]
            if np.random.rand() < self.cfg.aug_text:
                t2s = gen_texts[np.random.randint(len(gen_texts))][1]
                if len(t2s)>0:
                    t2 = t2s[np.random.randint(len(t2s))]
        if self.data_type=='train' and self.cfg.aug_lower>0:
            if np.random.rand()<self.cfg.aug_lower:
                t1 = t1.lower()
            if np.random.rand()<self.cfg.aug_lower:
                t2 = t2.lower()
        if self.cfg.use_lower and self.data_type!='train':
            t1, t2 = t1.lower(), t2.lower()
        return t1, t2

    def get_text(self, rec):
        t1, t2 = self._get_text(rec)
        aug_missing, aug_mix, rec_mix = False, False, None
        if self.data_type == 'train':
            if self.cfg.aug_combine>0 and np.random.rand()<self.cfg.aug_combine:
                t1 = t1 + ' ' + t2
                t2 = t1
            if self.cfg.aug_missing>0 and np.random.rand()<self.cfg.aug_missing:
                aug_missing = True
                if np.random.rand()<0.5:
                    t1 = ''
                else:
                    t2 = ''
            if not aug_missing and self.cfg.aug_mix>0 and np.random.rand()<self.cfg.aug_mix:
                aug_mix = True
                rec_mix = self.sample_rec(rec)
                mix_t1, mix_t2 = self._get_text(rec_mix)
                if np.random.rand()<0.5:
                    t1 = mix_t1
                else:
                    t2 = mix_t2

            if self.cfg.aug_order>0 and np.random.rand()<self.cfg.aug_order:
                text = t2 + ' ' + t1
            else:
                text = t1 + ' ' + t2
            if text[0]==' ':
                text = text[1:]
            if text[-1] == ' ':
                text = text[:-1]
        else:
            if self.cfg.tta_order:
                text = t2 + ' ' + t1
            else:
                text = t1 + ' ' + t2
        return text, rec_mix

    def get_input_ids(self, rec, index):
        if not self.cfg.is_classify:
            return self.get_ar_input_ids(rec, index)
        text, rec_mix = self.get_text(rec)
        input_ids = self.tokenizer.encode(text, max_length=self.cfg.max_seq_len, truncation=True)
        labels_mix = None
        if self.data_type!='test':
            labels, weights = self.get_labels(rec)
            if rec_mix is not None:
                labels_mix = self.get_labels(rec_mix)
                #labels[:-2] = [max(a, b) for a, b in zip(labels[:-2], labels_mix[:-2])]
        else:
            labels, weights = None, None
        return input_ids, labels, labels_mix, weights

    def get_rec(self, index):
        if not self.cfg.is_classify:
            index = int(index/len(self.cfg.train_cols))
        if self.data_type == 'train' and index >= len(self.data):
            rec = self.semi_data[np.random.randint(len(self.semi_data))]
        else:
            rec = self.data[index]
        return rec

    def getitem(self, index, rec=None):
        if rec is None:
            rec = self.get_rec(index)
        if self.cfg.is_split:
            return self.get_split_item(index, rec)
        item = dict(uid=rec.uid)
        input_ids, labels, labels_mix, weights = self.get_input_ids(rec, index)
        input_len = len(input_ids)
        item['seq_len'] = input_len
        item['input_ids'] = np.array(input_ids)
        item['attention_mask'] = np.ones([len(input_ids)])
        if weights is not None:
            item['weights'] = np.array(weights)
        if labels is not None:
            item['labels'] = np.array(labels)
            if not self.cfg.is_classify:
                orig_labels, _ = self.get_labels(rec, all_cols=True)
                item['orig_labels'] = np.array(orig_labels)
        if not self.cfg.is_classify:
            col = self.cfg.train_cols[index%len(self.cfg.train_cols)]
            item['label_ind'] = util.col2ind[col]
            item['item_ind'] = index//len(self.cfg.train_cols)
            if self.cfg.semi_ratio>0:
                item['is_semi'] = rec.src.startswith('semi')
            if col =="InjuryLocationType":
                item['rweight'] = self.cfg.cat_rdrop1
                if self.cfg.semi_ratio>0:
                    item['tokids'] = np.array(self.tokids[2:2+util.n_InjuryLocationType])
                    #if rec.src.startswith('semi'):
                    if 1==1:
                        item['orig_labels'] = item['orig_labels'][len(util.binary_labels):len(util.binary_labels)+util.n_InjuryLocationType]
            elif col == "WeaponType1":
                item['rweight'] = self.cfg.cat_rdrop2
                if self.cfg.semi_ratio>0:
                    item['tokids'] = np.array(self.tokids[2:2+util.n_WeaponType1:])
                    #if rec.src.startswith('semi'):
                    if 1==1:
                        item['orig_labels'] = item['orig_labels'][-util.n_WeaponType1:]
            else:
                item['rweight'] = self.cfg.bi_rdrop
                if self.cfg.semi_ratio>0:
                    item['tokids'] = np.array(self.tokids[:2])
                    #if rec.src.startswith('semi'):
                    if 1==1:
                        orig_labels = item['orig_labels'][util.binary_labels.index(col)]
                        item['orig_labels'] = np.array([orig_labels, 1-orig_labels])


        if labels_mix is not None:
            item['labels_mix'] = np.array(labels_mix)
        if self.data_type!='train' and self.cfg.use_cache:
            input_ids = item['input_ids']
            for i in range(len(input_ids)-1, -1, -1):
                if np.all(input_ids[i-1:i+1] == self.tokenizer.encode("Question:\n", add_special_tokens=False)):
                    item['prompt_cache_len'] = i-1
                    break
        return item

    def get_split_input_ids(self, rec, index):
        text1, text2 = self._get_text(rec)
        rec_mix, labels_mix = None, None
        if self.data_type=='train' and self.cfg.aug_mix>0 and np.random.rand()<self.cfg.aug_mix:
            rec_mix = self.sample_rec(rec)
            mix_t1, mix_t2 = self._get_text(rec_mix)
            if np.random.rand() < 0.5:
                text1 = mix_t1
            else:
                text2 = mix_t2
        input_ids1 = self.tokenizer.encode(text1, max_length=self.cfg.max_seq_len, truncation=True)
        input_ids2 = self.tokenizer.encode(text2, max_length=self.cfg.max_seq_len, truncation=True)
        if self.data_type!='test':
            labels, weights = self.get_labels(rec)
            if rec_mix is not None:
                labels_mix = self.get_labels(rec_mix)
        else:
            labels = None
        return input_ids1, input_ids2, labels, labels_mix

    def get_split_item(self, index, rec):
        item = dict(uid=rec.uid)
        input_ids1, input_ids2, labels, labels_mix = self.get_split_input_ids(rec, index)
        input_len1 = len(input_ids1)
        input_len2 = len(input_ids2)
        item['seq_len1'] = input_len1
        item['seq_len2'] = input_len2
        item['input_ids1'] = np.array(input_ids1)
        item['input_ids2'] = np.array(input_ids2)
        item['attention_mask1'] = np.ones([len(input_ids1)])
        item['attention_mask2'] = np.ones([len(input_ids2)])
        mask1, mask2 = 1, 1
        if labels_mix is None and self.data_type=='train' and self.cfg.aug_missing>0 and np.random.rand()<self.cfg.aug_missing:
            if np.random.rand()<0.5:
                mask1 = 0
            else:
                mask2 = 0

        item['mask1'] = mask1
        item['mask2'] = mask2
        if labels is not None:
            item['labels'] = np.array(labels)
            if not self.cfg.is_classify:
                orig_labels = self.get_labels(rec, all_cols=True)
                item['orig_labels'] = np.array(orig_labels)
        if not self.cfg.is_classify:
            col = self.cfg.train_cols[index%len(self.cfg.train_cols)]
            item['label_ind'] = util.col2ind[col]
            item['item_ind'] = index // len(self.cfg.train_cols)
            if col =="InjuryLocationType":
                item['rweight'] = self.cfg.cat1_rdrop
            elif col == "WeaponType1":
                item['rweight'] = self.cfg.cat2_rdrop
            else:
                item['rweight'] = self.cfg.bi_rdrop
            #item['tokids'] = np.array(self.tokids)
        if labels_mix is not None:
            item['labels_mix'] = np.array(labels_mix)
        return item

    def collate_split(self, batch):
        new_batch = dict()
        for k in ['uid']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]

        input_lens1 = [item['seq_len1'] for item in batch]
        input_lens2 = [item['seq_len2'] for item in batch]
        max_len1 = max(input_lens1)
        max_len2 = max(input_lens2)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        for i, item in enumerate(batch):
            if self.tokenizer.padding_side == 'left':
                item['input_ids1'] = np.pad(item['input_ids1'], ((max_len1 - item['seq_len1'], 0)), "constant", constant_values=pad_token_id)
                item['input_ids2'] = np.pad(item['input_ids2'], ((max_len1 - item['seq_len2'], 0)), "constant", constant_values=pad_token_id)
                if 'attention_mask1' in item:
                    item['attention_mask1'] = np.pad(item['attention_mask1'], ((max_len1 - item['seq_len1'], 0)), "constant", constant_values=0)
                    item['attention_mask2'] = np.pad(item['attention_mask2'], ((max_len2 - item['seq_len2'], 0)), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
            else:
                item['input_ids1'] = np.pad(item['input_ids1'], ((0, max_len1 - item['seq_len1'])), "constant", constant_values=pad_token_id)
                item['input_ids2'] = np.pad(item['input_ids2'], ((0, max_len2 - item['seq_len2'])), "constant", constant_values=pad_token_id)
                if 'attention_mask1' in item:
                    item['attention_mask1'] = np.pad(item['attention_mask1'], ((0, max_len1 - item['seq_len1'])), "constant", constant_values=0)
                    item['attention_mask2'] = np.pad(item['attention_mask2'], ((0, max_len2 - item['seq_len2'])), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
        batch = default_collate(batch)
        batch.update(new_batch)
        if not self.cfg.is_eval:
            batch = {k: v for k, v in batch.items() if k in ["input_ids1", "labels", "attention_mask1", "input_ids2", "attention_mask2", "mask1", "mask2"]}

        return batch

    def collate(self, batch):
        if self.cfg.is_split:
            return self.collate_split(batch)
        new_batch = dict()
        for k in ['uid', 'label_ind', 'item_ind', 'prompt_cache_len', 'is_semi']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]

        input_lens = [item['seq_len'] for item in batch]
        max_len = max(input_lens)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        for i, item in enumerate(batch):
            if self.tokenizer.padding_side == 'left':
                item['input_ids'] = np.pad(item['input_ids'], ((max_len - item['seq_len'], 0)), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((max_len - item['seq_len'], 0)), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
            else:
                item['input_ids'] = np.pad(item['input_ids'], ((0, max_len - item['seq_len'])), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((0, max_len - item['seq_len'])), "constant", constant_values=0)
                if 'labels' in item and not self.cfg.is_classify:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
        batch = default_collate(batch)
        batch.update(new_batch)
        if not self.cfg.is_eval:
            cols = ["input_ids", "labels", "attention_mask", "labels_mix", "weights", "rweight", "tokids"]
            if self.cfg.semi_ratio>0:
                cols.extend(["is_semi", "orig_labels"])
            batch = {k:v for k, v in batch.items() if k in cols}

        return batch



class Dataset(DatasetMix, torch.utils.data.Dataset):
    pass


def gen_ds(args, data_type, data, **kwargs):
    if data_type=='train':
        ds_cls = globals()[args.ds_cls]
    else:
        ds_cls = globals()[args.val_ds_cls]
    ds = ds_cls(args, data_type, data, **kwargs)
    return ds


if __name__ == '__main__':
    import util
    from transformers import AutoTokenizer
    args = util.parser.parse_args([])
    tokenizer = AutoTokenizer.from_pretrained()

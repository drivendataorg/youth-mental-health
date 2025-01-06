import os, sys, logging
import json
from glob import glob
import pandas as pd
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, StaticCache
from trainer import Trainer, TrainingArguments
#import trl
#from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import StoppingCriteria, StopStringCriteria, StoppingCriteriaList
from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from collections import defaultdict

import util
from dataset import gen_ds
#import warnings
#warnings.filterwarnings('always')


sys.path.insert(0, '../')


logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, last_token)):
                return True
        return False



def load_unsloth_model(args, model_id):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_id, dtype=getattr(torch, args.torch_dtype), use_cache=args.use_cache,
                                                         load_in_4bit=args.use_4bit)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def find_all_linear_names(args, model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if args.use_4bit else (bnb.nn.Linear8bitLt if args.use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'regression_head' in lora_module_names:
        lora_module_names.remove('regression_head')
    return list(lora_module_names)


def load_model(args, model_id):
    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(args, model_id)

    else:
        device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.is_classify:
            if args.train_cols is None:
                num_labels = len(util.binary_labels) + 18
            else:
                num_labels = len(args.train_cols)
            if args.avg_pool:
                from nn import CustDebertaV2ForSequenceClassification
                cls = CustDebertaV2ForSequenceClassification
            else:
                cls = AutoModelForSequenceClassification
            model = cls.from_pretrained(model_id, trust_remote_code=True, device_map=device_map,
                                                                       torch_dtype=getattr(torch, args.torch_dtype), num_labels=num_labels)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), device_map='cuda',
                                                         load_in_8bit=args.use_8bit)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    if model.config.pad_token_id is None:
        pad_token_id = model.config.eos_token_id
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[-1]
        model.config.pad_token_id = pad_token_id
        tokenizer.pad_token_id = pad_token_id
    if 'llama' in args.model_name:
        pad_token_id = 128009
        model.config.pad_token_id = pad_token_id
        tokenizer.pad_token_id = pad_token_id
        tokenizer.pad_token = tokenizer.decode(pad_token_id)
        logger.info('set llama pad token_id %s, %s', tokenizer.pad_token_id, tokenizer.pad_token)
    logger.info('pad token id:%s, %s, %s', tokenizer.decode(model.config.pad_token_id), model.config.pad_token_id, tokenizer.pad_token_id)
    return model, tokenizer

def prepare_dataset(args, **kwargs):
    if args.do_eval:
        data = util.load_data(args)
        kf = KFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        #kf = StratifiedKFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        splits = kf.split(data, data.src)
        for i in range(args.kn):
            train_inds, val_inds = next(splits)
            if i==args.kfid:
                break
        train_data = data.iloc[train_inds]
        val_data = data.iloc[val_inds]

        train_ds = gen_ds(args, 'train', train_data, **kwargs)
        val_ds = gen_ds(args, 'val', val_data, **kwargs)
        logger.info('train ds:%s, val_ds:%s', len(train_ds), len(val_ds))
        dl = torch.utils.data.DataLoader(val_ds, batch_size=args.val_batch_size, pin_memory=True, num_workers=args.dataloader_num_workers,
                                         shuffle=False, drop_last=False, collate_fn=val_ds.collate)
    elif args.do_test:
        test_args = deepcopy(args)
        test_args.data_type = 'test'
        test_data = util.load_data(test_args)

        test_ds = gen_ds(args, 'test', test_data, **kwargs)
        logger.info('test ds:%s', len(test_ds))
        dl = torch.utils.data.DataLoader(test_ds, batch_size=args.val_batch_size, pin_memory=True, num_workers=args.dataloader_num_workers,
                                         shuffle=False, drop_last=False, collate_fn=test_ds.collate)
    return dl



def eval_vllm(args):
    from vllm import LLM, SamplingParams
    output_dir = f"{args.output_dir}/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    model = LLM(model=args.backbone,
          dtype=args.torch_dtype,
          enforce_eager=False,
          gpu_memory_utilization=0.9,
          #swap_space=4,
          #kv_cache_dtype="fp8_e5m2",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=8192,
          stops=[']]']

          #worker_use_ray=True,
         )
    tokenizer = model.get_tokenizer()

    ds = prepare_dataset(args, tokenizer=tokenizer)
    preds = []
    for batch in enumerate(ds):
        texts = batch['texts']

def eval_classify(args, model, tokenizer, output_dir):
    ds = prepare_dataset(args, tokenizer=tokenizer)
    uids, preds, labels = [], [], []
    for i, batch in tqdm(enumerate(ds)):
        if i>=args.eval_num:
            break
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.use_bf16 else torch.float16) as autocast:
                    if args.is_split:
                        outputs1 = model(batch['input_ids1'].cuda(), attention_mask=batch['attention_mask1'].cuda())
                        outputs2 = model(batch['input_ids2'].cuda(), attention_mask=batch['attention_mask2'].cuda())
                        logits, _ = torch.max(torch.stack([outputs1.logits, outputs2.logits], axis=-1), axis=-1)
                    else:
                        outputs = model(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda())
                        logits = outputs.logits
            else:
                outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits
        uids.extend(batch['uid'])
        probs = logits
        if args.train_cols is None:
            probs[:,  :len(util.binary_labels)] = torch.sigmoid(probs[:, :len(util.binary_labels)])
            probs[:, len(util.binary_labels):len(util.binary_labels)+util.n_InjuryLocationType] = torch.softmax( probs[:, len(util.binary_labels):len(util.binary_labels)+util.n_InjuryLocationType], axis=-1)
            probs[:,  -util.n_WeaponType1:] = torch.softmax(probs[:,  -util.n_WeaponType1:], axis=-1)
        else:
            probs = torch.sigmoid(probs)
        preds.extend(probs.float().cpu().numpy())
        if 'labels' in batch:
            labels.extend(batch['labels'].numpy())
    preds = pd.DataFrame(dict(uid=uids, pred=preds))
    if len(labels)>0:
        preds['label'] = labels
    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('pred saved to: %s', fpath)
    if args.do_eval:
        thrs, best_preds, labels = util.search_thr(preds, cols=args.train_cols)
        fpath = f"{output_dir}/thrs{args.suffix}_{args.data_type}.dump"
        util.dump((thrs, best_preds, labels), fpath)
        logger.info('thrs saved to: %s', fpath)
    return preds


def restore_args(args, output_dir):
    restore_args = util.load_json(f"{output_dir}/args.json")
    for k in ['is_classify', 'is_split', 'avg_pool', 'use_ppt2']:
        setattr(args, k, restore_args.get(k, False))
        logger.info("restored args:%s, %s", k, getattr(args, k))
    for k in ['seed', 'data_seed', 'train_cols']:
        v = getattr(args, k)
        if v is None:
            v = restore_args.get(k, None)
        setattr(args, k, v)
    return args

def pred_use_cache_bak(model, ds, tokenizer):
    assert args.val_batch_size == 1
    Yid, Nid = tokenizer.convert_tokens_to_ids(['Yes', 'No'])
    InjuryLocationType_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(6)])
    WeaponType1_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(12)])
    uids, probs, label_inds, labels = [], [], [], []
    past_label_ind = -1
    prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=4096, device="cuda", dtype=torch.bfloat16)
    for i, batch in tqdm(enumerate(ds)):
        if i >= args.eval_num:
            break
        if 'orig_labels' in batch:
            labels.append(batch['orig_labels'].numpy())
        label_ind = batch['label_ind'][0]
        if label_ind==past_label_ind:
            past_key_values = deepcopy(past_key_values)
        else:
            prompt_cache = model(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda(), past_key_values=prompt_cache).past_key_values
            prompt_cache_len = batch['prompt_cache_len'][0]
            prompt_cache.key_cache = [x[:, :, :prompt_cache_len, :] for x in prompt_cache.key_cache]
            prompt_cache.value_cache = [x[:, :, :prompt_cache_len, :] for x in prompt_cache.value_cache]
            past_key_values = prompt_cache

        outputs = model.generate(batch['input_ids'][:, prompt_cache_len:].cuda(), use_cache=True, past_key_value=prompt_cache, max_new_tokens=1, \
                             attention_mask=batch['attention_mask'][:, prompt_cache_len:].cuda(), return_dict_in_generate=True, output_scores=True, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
        scores = outputs.scores[0]
        label_ind = batch['label_ind']
        for j in range(len(batch['uid'])):
            if label_ind[j] == len(util.binary_labels):
                probs.append(torch.softmax(scores[j, InjuryLocationType_ids], axis=-1).float().cpu().numpy())
            elif label_ind[j] == (len(util.binary_labels) + 1):
                probs.append(torch.softmax(scores[j, WeaponType1_ids], axis=-1).float().cpu().numpy())
            else:
                probs.append(torch.softmax(scores[j, [Yid, Nid]], axis=-1).float().cpu().numpy()[0])
        uids.append(batch['uid'])
        label_inds.append(batch['label_ind'])
        if 'orig_labels' in batch:
            labels.append(batch['orig_labels'].numpy())

    return uids, probs, label_inds, labels

def pred_use_cache(model, ds, tokenizer):
    assert args.val_batch_size == 1
    Yid, Nid = tokenizer.convert_tokens_to_ids(['Yes', 'No'])
    InjuryLocationType_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(6)])
    WeaponType1_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(12)])
    uids, probs, label_inds, labels = [], [], [], []
    past_item_ind = -1
    for i, batch in tqdm(enumerate(ds)):
        if i >= args.eval_num:
            break
        if 'orig_labels' in batch:
            labels.append(batch['orig_labels'].numpy())
        item_inds = batch['item_ind']
        item_ind = item_inds[0]
        input_ids, attention_mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
        if item_ind==past_item_ind:
            past_key_values = deepcopy(prompt_cache)
            prompt_cache_len = batch['prompt_cache_len'][0]
            input_ids, attention_mask = input_ids[:, prompt_cache_len:], attention_mask[:, prompt_cache_len:]
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.use_bf16 else torch.float16) as autocast:
                    outputs = model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits
                    scores = last_pool(logits, attention_mask)
        else:
            prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=4096, device="cuda", dtype=getattr(torch, args.torch_dtype))
            past_item_ind = item_ind
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.use_bf16 else torch.float16) as autocast:
                    outputs = model(input_ids, attention_mask=attention_mask, past_key_values=prompt_cache)
                    logits = outputs.logits
                    scores = last_pool(logits, attention_mask)
            prompt_cache = outputs.past_key_values
            prompt_cache_len = batch['prompt_cache_len'][0]
            for x in prompt_cache.key_cache:
                x[:, :, prompt_cache_len:, :] = 0
            for x in prompt_cache.value_cache:
                x[:, :, prompt_cache_len:, :] = 0

        for j in range(len(batch['uid'])):
            if batch['label_ind'][j] == len(util.binary_labels):
                probs.append(torch.softmax(scores[j, InjuryLocationType_ids], axis=-1).float().cpu().numpy())
            elif batch['label_ind'][j] == (len(util.binary_labels) + 1):
                probs.append(torch.softmax(scores[j, WeaponType1_ids], axis=-1).float().cpu().numpy())
            else:
                probs.append(torch.softmax(scores[j, [Yid, Nid]], axis=-1).float().cpu().numpy()[0])
        uids.append(batch['uid'])
        label_inds.append(batch['label_ind'])
        if 'orig_labels' in batch:
            labels.append(batch['orig_labels'].numpy())

    return uids, probs, label_inds, labels

def main(args):
    logger.info("model:%s", args.model_name)
    args.is_eval = True
    output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
    args = restore_args(args, output_dir)
    if args.use_vllm:
        preds = eval_vllm(args)
    os.makedirs(output_dir, exist_ok=True)
    if args.restore_step is not None:
        ckpt_dir = f"{output_dir}/checkpoint-{args.restore_step}"
    else:
        ckpt_dir = sorted(glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    logger.info('restore from ckpt:%s', ckpt_dir)
    model, tokenizer = load_model(args, ckpt_dir)
    logger.info('num of params %s', util.get_num_of_paras(model))

    if args.is_classify:
        return eval_classify(args, model, tokenizer, output_dir)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    ds = prepare_dataset(args, tokenizer=tokenizer)
    if args.use_cache:
        uids, probs, label_inds, labels = pred_use_cache(model, ds, tokenizer)
    else:
        Yid, Nid = tokenizer.convert_tokens_to_ids(['Yes', 'No'])
        if 'phi' in args.model_name or 'qw' in args.model_name or 'gemma' in args.model_name:
            InjuryLocationType_ids = tokenizer.convert_tokens_to_ids([chr(ord('A') +i) for i in range(6)])
            WeaponType1_ids = tokenizer.convert_tokens_to_ids([chr(ord('A')+i) for i in range(12)])
        else:
            InjuryLocationType_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(6)])
            WeaponType1_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(12)])
        uids, probs, label_inds, labels = [], [], [], []
        for i, batch in tqdm(enumerate(ds)):
            if i>=args.eval_num:
                break
            outputs = model.generate(batch['input_ids'].cuda(), use_cache=False, max_new_tokens=1, \
                                     attention_mask=batch['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
            scores = outputs.scores[0]
            label_ind = batch['label_ind']
            for j in range(len(batch['uid'])):
                if label_ind[j]==len(util.binary_labels):
                    probs.append(torch.softmax(scores[j, InjuryLocationType_ids], axis=-1).float().cpu().numpy())
                elif label_ind[j]==(len(util.binary_labels)+1):
                    probs.append(torch.softmax(scores[j, WeaponType1_ids], axis=-1).float().cpu().numpy())
                else:
                    probs.append(torch.softmax(scores[j, [Yid, Nid]], axis=-1).float().cpu().numpy()[0])
            uids.append(batch['uid'])
            label_inds.append(label_ind)
            if 'orig_labels' in batch:
                labels.append(batch['orig_labels'].numpy())
    uids, label_inds = np.concatenate(uids), np.concatenate(label_inds)
    unique_uids = sorted(np.unique(uids))
    uid2ind = dict(zip(unique_uids, range(len(unique_uids))))
    num = len(uids)//len(args.train_cols)
    assert num == len(unique_uids)
    preds = np.zeros([num, len(util.binary_labels)+18])
    for uid, prob, label_ind in zip(uids, probs, label_inds):
        if label_ind == len(util.binary_labels):
            preds[uid2ind[uid], label_ind:label_ind+util.n_InjuryLocationType] = prob
        elif label_ind == (len(util.binary_labels) + 1):
            preds[uid2ind[uid], -util.n_WeaponType1:] = prob
        else:
            preds[uid2ind[uid], label_ind] = prob

    preds = pd.DataFrame(dict(uid=list(unique_uids), pred=list(preds)))
    if len(labels)>0:
        #inds = [util.col2ind[col] for col in args.train_cols]
        labels = np.concatenate(labels)
        uid2label = dict(zip(uids, labels))
        labels = [uid2label[uid] for uid in unique_uids]
        preds['label'] = labels
    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('pred saved to: %s', fpath)
    if args.do_eval:
        thrs, best_preds, labels = util.search_thr(preds)
        fpath = f"{output_dir}/thrs{args.suffix}_{args.data_type}.dump"
        util.dump((thrs, best_preds, labels), fpath)
        logger.info('thrs saved to: %s', fpath)
    return preds

def last_pool(hidden_states, attention_mask):
    ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
    ends = ends.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
    hidden_states = torch.gather(hidden_states, 1, ends.to(hidden_states.device)).squeeze(1)
    return hidden_states


def main_bak(args):
    logger.info("model:%s", args.model_name)
    args.is_eval = True
    output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
    args = restore_args(args, output_dir)
    if args.use_vllm:
        preds = eval_vllm(args)
    os.makedirs(output_dir, exist_ok=True)
    if args.restore_step is not None:
        ckpt_dir = f"{output_dir}/checkpoint-{args.restore_step}"
    else:
        ckpt_dir = sorted(glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    logger.info('restore from ckpt:%s', ckpt_dir)
    model, tokenizer = load_model(args, ckpt_dir)
    logger.info('num of params %s', util.get_num_of_paras(model))

    if args.is_classify:
        return eval_classify(args, model, tokenizer, output_dir)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    ds = prepare_dataset(args, tokenizer=tokenizer)
    if args.use_cache:
        uids, probs, label_inds, labels = pred_use_cache(model, ds, tokenizer)
    else:
        Yid, Nid = tokenizer.convert_tokens_to_ids(['Yes', 'No'])
        InjuryLocationType_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(6)])
        WeaponType1_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(12)])
        uids, probs, label_inds, labels = [], [], [], []
        for i, batch in tqdm(enumerate(ds)):
            if i>=args.eval_num:
                break
            with torch.no_grad():
                if torch.cuda.is_available():
                    #input_ids, attention_mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
                    #with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.use_bf16 else torch.float16) as autocast:
                    if 1==1:
                        outputs = model(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda())
                        #logits = outputs.logits
                        #scores = last_pool(logits, attention_mask)
                        scores = outputs.logits[:, -1]
                else:
                    outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                    scores = last_pool(logits, batch['attention_mask'])
            label_ind = batch['label_ind']
            for j in range(len(batch['uid'])):
                if label_ind[j]==len(util.binary_labels):
                    probs.append(torch.softmax(scores[j, InjuryLocationType_ids], axis=-1).float().cpu().numpy())
                elif label_ind[j]==(len(util.binary_labels)+1):
                    probs.append(torch.softmax(scores[j, WeaponType1_ids], axis=-1).float().cpu().numpy())
                else:
                    probs.append(torch.softmax(scores[j, [Yid, Nid]], axis=-1).float().cpu().numpy()[0])
            uids.append(batch['uid'])
            label_inds.append(label_ind)
            if 'orig_labels' in batch:
                labels.append(batch['orig_labels'].numpy())
    uids, label_inds = np.concatenate(uids), np.concatenate(label_inds)
    unique_uids = sorted(np.unique(uids))
    uid2ind = dict(zip(unique_uids, range(len(unique_uids))))
    num = len(uids)//len(args.train_cols)
    assert num == len(unique_uids)
    preds = np.zeros([num, len(util.binary_labels)+18])
    for uid, prob, label_ind in zip(uids, probs, label_inds):
        if label_ind == len(util.binary_labels):
            preds[uid2ind[uid], label_ind:label_ind+util.n_InjuryLocationType] = prob
        elif label_ind == (len(util.binary_labels) + 1):
            preds[uid2ind[uid], -util.n_WeaponType1:] = prob
        else:
            preds[uid2ind[uid], label_ind] = prob

    preds = pd.DataFrame(dict(uid=list(unique_uids), pred=list(preds)))
    if len(labels)>0:
        #inds = [util.col2ind[col] for col in args.train_cols]
        labels = np.concatenate(labels)
        uid2label = dict(zip(uids, labels))
        labels = [uid2label[uid] for uid in unique_uids]
        preds['label'] = labels
    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('pred saved to: %s', fpath)
    if args.do_eval:
        thrs, best_preds, labels = util.search_thr(preds)
        fpath = f"{output_dir}/thrs{args.suffix}_{args.data_type}.dump"
        util.dump((thrs, best_preds, labels), fpath)
        logger.info('thrs saved to: %s', fpath)
    return preds







if __name__ == "__main__":
    args = util.parser.parse_args()
    util.set_logger()
    if args.debug:
        args.backbone = 'HuggingFaceTB/SmolLM-135M'
        args.num_train_epochs = 2
        args.max_seq_len = 8
        args.num = 10
        args.eval_steps = 2
        args.batch_size = 1
        args.val_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.do_train = True
        args.seed = 9527
        args.kn = 2
        args.use_full = True
        args.ds_cls = 'PretrainDataset'
        args.val_ds_cls = 'PretrainDataset'
    preds = []
    for kfid in args.kfids.split():
        args.kfid = int(kfid)
        pred = main(args)
        preds.append(pred)
    if args.do_eval:
        preds = pd.concat(preds)
        logger.info('kf thrs')
        thrs, best_preds, labels = util.search_thr(preds)
        fpath = f"{args.data_dir}/{args.model_name}_thrs{args.suffix}_{args.data_type}.dump"
        util.dump((thrs, best_preds, labels), fpath)
        logger.info('thrs saved to: %s', fpath)


import os, sys, json, logging
from copy import deepcopy
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import time
from argparse import ArgumentParser
import numpy as np
import subprocess
from contextlib import contextmanager
import re
import hashlib
from sklearn.metrics import f1_score


logger = logging.getLogger(__name__)


parser = ArgumentParser(conflict_handler='resolve')
parser.add_argument("-d", "--debug",  action="store_true")
parser.add_argument("-data_type", default='train')
parser.add_argument("-no_val", action="store_true")
parser.add_argument("-ds", "--dataset", default='yaa')
parser.add_argument("-ds_cls", default="Dataset")
parser.add_argument("-val_ds_cls", default="Dataset")
parser.add_argument("-data_type", default='train')
parser.add_argument("-data_dir", default='../data')
parser.add_argument("-kn", type=int)
parser.add_argument("-kfid", type=int, default=0)
parser.add_argument("-kfids")
parser.add_argument("-model_name")
parser.add_argument("-backbone")
parser.add_argument("-val_pct", default=0.1)
parser.add_argument("-seed", type=int)
parser.add_argument("-data_seed", type=int)
parser.add_argument("-n_xtoken", type=int)
parser.add_argument("-prefix")
parser.add_argument("-max_seq_len", type=int, default=81920)
parser.add_argument("-max_gen_len", type=int, default=16)
parser.add_argument("-disable_tqdm",  action="store_true")
parser.add_argument("-use_fp16", action="store_true")
parser.add_argument("-use_bf16", action="store_true")
parser.add_argument("-save_opt",  action="store_true")
parser.add_argument("-remove_unused_columns",  action="store_true")
parser.add_argument("-gradient_checkpointing",  action="store_true")
parser.add_argument("-use_badam",  action="store_true")
parser.add_argument("-use_full",  action="store_true")
parser.add_argument("-switch_block_every",  type=int, default=32)
parser.add_argument("-use_score_scaling",  action="store_true")
parser.add_argument("-torch_compile",  action="store_true")
parser.add_argument("-use_double_quant",  action="store_true")
parser.add_argument("-m", "--method_name")

parser.add_argument("-bs", "--batch_size", type=int)
parser.add_argument("-mbs", "--min_batch_size", type=int)
parser.add_argument("-vbs", "--val_batch_size", type=int)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-lr_scheduler_type", default='linear')
parser.add_argument("-lr_scheduler_kwargs", type=json.loads, help='lr scheduler parameters')

parser.add_argument("-init_kl_coef", type=float, default=0.2)
parser.add_argument("-max_grad_norm", type=float, default=1)
parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("-optim", default="adamw_torch")
parser.add_argument("-dataloader_num_workers", type=int, default=2)
parser.add_argument("-warmup_steps", type=int, default=0)
parser.add_argument("-warmup_ratio", type=float, default=0)
parser.add_argument("-weight_decay", type=float, default=1e-2)
parser.add_argument("-output_dir", default="../data")
parser.add_argument("-verbose", type=int, default=16)
parser.add_argument("-report_to", default="tensorboard")
parser.add_argument("-min_cnt", type=int, default=10)
parser.add_argument("-evaluation_strategy", default='steps')
parser.add_argument("-save_strategy", default='steps')
parser.add_argument("-eval_steps", type=int, default=1000)
parser.add_argument("-eval_delay", type=int, default=0)
parser.add_argument("-save_total_limit", type=int, default=1000000000)
parser.add_argument("-epochs", "--num_train_epochs", type=int, default=10)
parser.add_argument("-max_steps", type=int, default=-1)
parser.add_argument("-es", "--early_stopping_patience", type=int, default=1)
parser.add_argument("-metric_for_best_model", default='loss')
parser.add_argument("-num", type=int, default=10000000000)
parser.add_argument("-temp", type=float, default=1.0)
parser.add_argument("-topp", type=float, default=1.0)
parser.add_argument("-topk", type=int, default=-1)
parser.add_argument("-n_best", type=int, default=1)
parser.add_argument("-max_temp", type=float, default=5)
parser.add_argument("-min_temp", type=float, default=0.01)
parser.add_argument("-temp_decay_step", type=int, default=10000)
parser.add_argument("-n_frozen", type=int, default=0)
parser.add_argument("-frozen_emb",  action="store_true")
parser.add_argument("-use_lora",  action="store_true")
parser.add_argument("-lora_rank", type=int, default=32)
parser.add_argument("-lora_alpha", type=int, default=16)
parser.add_argument("-lora_dropout", type=float, default=0)
parser.add_argument("-lora_modules", nargs='+', default=None)
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-freeze_lm",  action="store_true")
parser.add_argument("-unfreeze_lm_head",  action="store_true")
parser.add_argument("-use_rslora",  action="store_true",default=None)
parser.add_argument("-ext_ratio",  type=float, default=0)
parser.add_argument("-use_orcamath",  action="store_true")
parser.add_argument("-use_mustard",  action="store_true")
parser.add_argument("-restore",  action="store_true")
parser.add_argument("-do_train",  action="store_true")
parser.add_argument("-do_eval",  action="store_true")
parser.add_argument("-do_predict",  action="store_true")
parser.add_argument("-do_test",  action="store_true")
parser.add_argument("-use_unsloth",  action="store_true")
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-use_4bit", action="store_true")
parser.add_argument("-use_8bit", action="store_true")
parser.add_argument("-no_validate", action="store_true")
parser.add_argument("-hard_ratio", type=float, default=0)
parser.add_argument("-suffix", default="")

parser.add_argument("-modelid")
parser.add_argument("-test_fname", default='smoke_test_features_bWOfr2M.csv')
parser.add_argument("-train_cols", nargs='+', default=None)
parser.add_argument("-tta_order", action="store_true")
parser.add_argument("-avg_pool", action="store_true")
parser.add_argument("-use_lower", action="store_true")
parser.add_argument("-aug_order", type=float, default=0)
parser.add_argument("-aug_order2", type=float, default=0)
parser.add_argument("-aug_lower", type=float, default=0)
parser.add_argument("-aug_missing", type=float, default=0)
parser.add_argument("-aug_text", type=float, default=0)
parser.add_argument("-aug_text2", type=float, default=0)
parser.add_argument("-aug_mix", type=float, default=0)
parser.add_argument("-aug_combine", type=float, default=0)
parser.add_argument("-output_name", default='output')
parser.add_argument("-torch_dtype", default='bfloat16')
parser.add_argument("-n_gpu_layer", type=int)
parser.add_argument("-n_thread", type=int)
parser.add_argument("-n_sample", type=int, default=1)
parser.add_argument("-att_dp", type=float, default=0)
parser.add_argument("-cls_dp", type=float, default=0)
parser.add_argument("-pool_dp", type=float, default=0)
parser.add_argument("-lora")
parser.add_argument("-lora_start_layer", type=int, default=0)
parser.add_argument("-use_sampler", action="store_true")
parser.add_argument("-is_eval", action="store_true")
parser.add_argument("-is_classify", action="store_true")
parser.add_argument("-is_split", action="store_true")
parser.add_argument("-cat_ls", type=float, default=0)
parser.add_argument("-bi_ls", type=float, default=0)
parser.add_argument("-rdrop", type=float, default=0)
parser.add_argument("-bi_rdrop", type=float, default=0)
parser.add_argument("-cat_rdrop1", type=float, default=0)
parser.add_argument("-cat_rdrop2", type=float, default=0)
parser.add_argument("-w_bi", type=float, default=0.3333)
parser.add_argument("-w_lt", type=float, default=0.3333)
parser.add_argument("-w_wt", type=float, default=0.3333)
parser.add_argument("-semi_ratio", type=float, default=0)
parser.add_argument("-semi_fpath")
parser.add_argument("-dp_start", type=int, default=0)
parser.add_argument("-n_gen", type=int, default=1)
parser.add_argument("-n_token", type=int, default=-1)
parser.add_argument("-gen_seed", type=int)
parser.add_argument("-restore_step", type=int)
parser.add_argument("-use_ppt2", action="store_true")
parser.add_argument("-eval_num", type=int, default=1000000000)
parser.add_argument("-cpu_offload_gb", type=float, default=0)
parser.add_argument("-sort_seq",  action="store_true")
parser.add_argument("-use_kl", action="store_true")
parser.add_argument("-temperature", type=float, default=0)
parser.add_argument("-use_vllm", action="store_true")
parser.add_argument("-use_weight", action="store_true")
parser.add_argument("-use_cache", action="store_true")
parser.add_argument("-disable_eager", action="store_true")
parser.add_argument("-is_oof", action="store_true")
parser.add_argument("-trust_remote_code", action="store_true")
parser.add_argument("-vll_quant")

binary_labels = ['DepressedMood',
       'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
       'SuicideAttemptHistory', 'SuicideThoughtHistory',
       'SubstanceAbuseProblem', 'MentalHealthProblem', 'DiagnosisAnxiety',
       'DiagnosisDepressionDysthymia', 'DiagnosisBipolar', 'DiagnosisAdhd',
       'IntimatePartnerProblem', 'FamilyRelationship', 'Argument',
       'SchoolProblem', 'RecentCriminalLegalProblem', 'SuicideNote',
       'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
       'DisclosedToOtherFamilyMember', 'DisclosedToFriend']
binary_weight = {'DepressedMood': 0.328,
 'MentalIllnessTreatmentCurrnt': 0.2585,
 'HistoryMentalIllnessTreatmnt': 0.3725,
 'SuicideAttemptHistory': 0.2095,
 'SuicideThoughtHistory': 0.4095,
 'SubstanceAbuseProblem': 0.229,
 'MentalHealthProblem': 0.48725,
 'DiagnosisAnxiety': 0.13375,
 'DiagnosisDepressionDysthymia': 0.36225,
 'DiagnosisBipolar': 0.0655,
 'DiagnosisAdhd': 0.0595,
 'IntimatePartnerProblem': 0.27975,
 'FamilyRelationship': 0.15425,
 'Argument': 0.221,
 'SchoolProblem': 0.10175,
 'RecentCriminalLegalProblem': 0.07,
 'SuicideNote': 0.3385,
 'SuicideIntentDisclosed': 0.27425,
 'DisclosedToIntimatePartner': 0.08475,
 'DisclosedToOtherFamilyMember': 0.09775,
 'DisclosedToFriend': 0.0635}
n_InjuryLocationType = 6
n_WeaponType1 = 12
all_labels = binary_labels + ["InjuryLocationType", "WeaponType1"]
col2ind = dict(zip(all_labels, range(len(all_labels))))

def average_f1(predictions: pd.DataFrame, labels: pd.DataFrame):
    """Score a set of predictions using the competition metric. F1 is averaged
    across all target variables. For categorical variables, micro-averaged
    F1 score is used.

    Args:
        predictions (pd.DataFrame): Dataframe of predictions, with one column
            for each target variable. The index should be the uid.
        labels (pd.DataFrame): Dataframe of ground truth values, with one column
            for each target variable. The index should be the uid.
    """
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    # Check that column order and row order are the same
    assert (predictions.columns == labels.columns).all()
    assert (predictions.index == labels.index).all()

    # All values should be integers
    assert (predictions.dtypes == int).all()

    CATEGORICAL_VARS = ["InjuryLocationType", "WeaponType1"]
    BINARY_VARS = np.setdiff1d(labels.columns, CATEGORICAL_VARS)

    # Calculate F1 score averaged across binary variables
    binary_f1 = f1_score(
        labels[BINARY_VARS],
        predictions[BINARY_VARS],
        average="macro",
    )
    f1s = [binary_f1]

    # Calculate F1 score for each categorical variable
    for cat_col in CATEGORICAL_VARS:
        f1s.append(f1_score(labels[cat_col], predictions[cat_col], average="micro"))

    return np.average(f1s, weights=[len(BINARY_VARS), 1, 1])


def score(preds, labels):
    preds = preds.set_index('uid')
    labels = labels.set_index('uid')
    return average_f1(preds, labels)

def probs2preds(preds, thrs):
    all_probs = np.stack(preds.pred.values, axis=0)
    best_preds = []
    for i in range(len(binary_labels)):
        probs = all_probs[:, i]
        thr = thrs[i]
        best_preds.append(probs>thr)
    pred_InjuryLocationType = np.argmax(all_probs[:, len(binary_labels):len(binary_labels)+n_InjuryLocationType], axis=-1)
    pred_WeaponType1 = np.argmax(all_probs[:, -n_WeaponType1:], axis=-1)
    best_preds.append(pred_InjuryLocationType)
    best_preds.append(pred_WeaponType1)
    columns = binary_labels + ['InjuryLocationType', 'WeaponType1']
    best_preds = pd.DataFrame(list(zip(*best_preds)), columns=columns, dtype="int")
    best_preds['uid'] = preds.uid.values
    return best_preds

def logits2preds(preds, thrs):
    logits = np.stack(preds.pred.values, axis=0)
    best_preds = []
    for i in range(len(binary_labels)):
        probs = sigmoid(logits[:, i])
        thr = thrs[i]
        best_preds.append(probs>thr)
    pred_InjuryLocationType = np.argmax(logits[:, len(binary_labels):len(binary_labels)+n_InjuryLocationType], axis=-1)
    pred_WeaponType1 = np.argmax(logits[:, -n_WeaponType1:], axis=-1)
    best_preds.append(pred_InjuryLocationType)
    best_preds.append(pred_WeaponType1)
    columns = binary_labels + ['InjuryLocationType', 'WeaponType1']
    best_preds = pd.DataFrame(list(zip(*best_preds)), columns=columns, dtype="int")
    best_preds['uid'] = preds.uid.values
    return best_preds

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def _search_thr(probs, labels, step=0.05):
    best_thr, best_s = -1, 0
    for thr in np.arange(0, 1, step):
        s = f1_score(labels, probs>thr, average='binary')
        if s>best_s:
            best_thr, best_s = thr, s
    return best_thr, best_s

def search_col_thr(preds, is_classify=True, cols=None, step=0.05):
    all_probs = np.stack(preds.pred.values, axis=0)
    labels = np.stack(preds.label.values, axis=0)
    thrs, ss, best_preds = [], [], []
    for i in range(len(cols)):
        #probs = sigmoid(logits[:, i])
        probs = all_probs[:, i]
        thr, s = _search_thr(probs, labels[:, i], step=step)
        thrs.append(thr)
        ss.append(s)
        best_preds.append(probs>thr)
    logger.info('avg binary score is %s, thr and ss:%s', np.mean(ss), list(zip(cols, thrs, ss)))
    best_preds = pd.DataFrame(list(zip(*best_preds)), columns=cols, dtype="int")
    best_preds['uid'] = preds.uid.values
    labels = pd.DataFrame(list(labels), columns=cols)
    labels['uid'] = preds.uid.values
    logger.info('best score is %s', score(best_preds, labels))
    return thrs, best_preds, labels

def search_thr(preds, is_classify=True, cols=None, step=0.05):
    if cols is not None:
        return search_col_thr(preds, is_classify, cols, step=step)
    all_probs = np.stack(preds.pred.values, axis=0)
    labels = np.stack(preds.label.values, axis=0)
    thrs, ss, best_preds = [], [], []
    for i in range(len(binary_labels)):
        #probs = sigmoid(logits[:, i])
        probs = all_probs[:, i]
        thr, s = _search_thr(probs, labels[:, i], step=step)
        thrs.append(thr)
        ss.append(s)
        best_preds.append(probs>thr)
    logger.info('avg binary score is %s, thr and ss:%s', np.mean(ss), list(zip(binary_labels, thrs, ss)))
    pred_InjuryLocationType = np.argmax(all_probs[:, len(binary_labels):len(binary_labels)+n_InjuryLocationType], axis=-1)
    s = f1_score(labels[:, -2], pred_InjuryLocationType, average='micro')
    logger.info('f1 for InjuryLocationType is %s', s)
    pred_WeaponType1 = np.argmax(all_probs[:, -n_WeaponType1:], axis=-1)
    s = f1_score(labels[:, -1], pred_WeaponType1, average='micro')
    logger.info('f1 for WeaponType1 is %s', s)
    best_preds.append(pred_InjuryLocationType)
    best_preds.append(pred_WeaponType1)
    columns = binary_labels + ['InjuryLocationType', 'WeaponType1']
    best_preds = pd.DataFrame(list(zip(*best_preds)), columns=columns, dtype="int")
    best_preds['uid'] = preds.uid.values
    labels = pd.DataFrame(list(labels), columns=columns)
    labels['uid'] = preds.uid.values
    logger.info('best score is %s', score(best_preds, labels))
    return thrs, best_preds, labels

def load_gen_data(args):
    rsts = defaultdict(list)
    model_rsts = dict()
    num = 0
    for fdir in sorted(glob(f"{args.data_dir}/gen/*")):
        model = "_".join(os.path.basename(fdir).split("_")[:2])
        model_rsts[model] = dict()
        for fpath in glob(f"{fdir}/*.json"):
            k = os.path.basename(fpath).split(".json")[0]
            recs = load_json(fpath)[k]
            les = [r[0] for r in recs[0] if r[1]=='stop']
            cmes = [r[0] for r in recs[1] if r[1]=='stop']
            if len(recs[0])!=len(les):
                print('les not stop', fpath)
            if len(recs[1])!=len(cmes):
                print('cmes not stop', fpath)
            if k not in model_rsts[model]:
                model_rsts[model][k] = [[], []]
            num += 1
            model_rsts[model][k][0].extend(les)
            model_rsts[model][k][1].extend(cmes)
    logger.info("models %s, total num:%s", model_rsts.keys(), num)
    for model, model_rst in model_rsts.items():
        for k, v in model_rst.items():
            rsts[k].append(v)
    return rsts

def parse_le_cme(text):
    pass


def load_gen2_data(args, data_dir='gen2'):
    rsts = []
    for fdir in tqdm(sorted(glob(f"{args.data_dir}/{data_dir}/*"))):
        model = os.path.basename(fdir)
        for fpath in glob(f"{fdir}/*.json"):
            uid = os.path.basename(fpath).split(".json")[0]
            assert uid not in rsts
            rec = load_json(fpath)[uid][0]
            if rec[1]!='stop':
                continue
            text = rec[0]+'\n</medical>'
            try:
                le = re.search('<law>.*?</law>', text, flags=re.DOTALL)
                cme = re.search('<medical>.*?</medical>', text, flags=re.DOTALL)
                if le and cme:
                    le = le.group()
                    cme = cme.group()
                else:
                    continue
                le = le[5:-6].strip()
                cme = cme[9:-10].strip()
                if len(le.split())>10 and len(cme.split())>10:
                    rsts.append([uid, le, cme, model])
            except Exception as e:
                logger.error(e)
                continue
            if len(rsts)>=args.num:
                break
        if len(rsts)>=args.num:
            break
    logger.info("total num:%s", len(rsts))
    df = pd.DataFrame(rsts, columns=["uid", "NarrativeLE", "NarrativeCME", "model"])
    return df

def load_yaa_train(args):
    fpath1 = f'{args.data_dir}/{args.dataset}/train_features.csv'
    df1 = pd.read_csv(fpath1, nrows=args.num)
    fpath2 = f'{args.data_dir}/{args.dataset}/train_labels.csv'
    df2 = pd.read_csv(fpath2)
    df = df1.merge(df2, on=["uid"])
    return df.iloc[:20] # NOTE - change


def load_data(args):
    if args.dataset=='yaa':
        if args.data_type=='train':
            df = load_yaa_train(args)
        elif args.data_type=='test':
            fpath = f'{args.data_dir}/{args.dataset}/{args.test_fname}'
            df = pd.read_csv(fpath, nrows=args.num)
        else:
            raise NotImplementedError(args.data_type)
    elif args.dataset=='gen2':
        df = load_gen2_data(args)
    elif args.dataset=='gen3':
        df = load_gen2_data(args, data_dir='gen3')
    elif args.dataset=='gen5':
        df = load_gen2_data(args, data_dir='gen5')
        df = df.sort_values(['model', 'uid']).groupby('model').sample(n=3000, random_state=3233)
        logger.info('total num of gen5 is %s', len(df))
    else:
        raise NotImplementedError(args.dataset)
    df['src'] = args.dataset
    return df

def load_kf_preds(args):
    preds = []
    for kfid in args.kfid.split():
        output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
        pred = load_dump(f"{output_dir}/pred{args.suffix}_{args.data_type}.dump")
        preds.append(pred)
    if args.data_type!='test':
        preds = pd.concat(preds)
    return preds



def set_logger(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(level)


def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

@contextmanager
def timer(name):
    t0 = time.time()
    #print('{} start'.format(name))
    logger.info('%s start', name)
    yield
    #print('{} done in {} seconds'.format(name, time.time() - t0))
    logger.info('%s done in %s seconds', name, time.time()-t0)

def load_dump(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, fpath, protocol=2):
    fdir = os.path.dirname(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def load_json(fpath):
    with open(fpath) as f:
        dictionary = json.load(f)
    return dictionary


def load_json_lines(fpath, num=1e16, exclude_keys=None, post_process=None):
    if exclude_keys is None:
        exclude_keys = []
    data = []
    with open(fpath) as f:
        for l in f:
            dic = json.loads(l)
            for k in exclude_keys:
                if k in dic:
                    _ = dic.pop(k)
            if post_process is not None:
                dic = post_process(dic)
            data.append(dic)
            if len(data)>=num:
                break
    return data


def dump_json(dictionary, fpath, ensure_ascii=False):
    with open(fpath, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=ensure_ascii)


def dump_json_lines(dicts, fpath, ensure_ascii=False):
    with open(fpath, 'w', encoding='utf8') as f:
        for d in dicts:
            json.dump(d, f, ensure_ascii=ensure_ascii)
            f.write(os.linesep)


def timestamp():
    return time.strftime('%Y%m%d%H%M%S')

def get_num_of_paras(m):
    num1, num2 = 0, 0
    for p in m.parameters():
        if p.requires_grad:
            num1 += p.numel()
        else:
            num2 += p.numel()
    return num1/1000/1000, num2/1000/1000



if __name__ == "__main__":
    args = parser.parse_args([])
    args.data_type = 'train'
    #df = load_data(args)
    load_gen_data(args)

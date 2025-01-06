import os, sys, logging
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import subprocess
import torch

sys.path.insert(0, 'yaa')
from yaa import util

def get_model_names():
    model_names = defaultdict(list)
    for fpath in glob(f"../data/*KF*"):
        model_name = os.path.basename(fpath)
        m = model_name.split("_KF")[0]
        model_names[m].append(model_name)
    for k, vs in model_names.items():
        model_names[k] = sorted(vs)
    return model_names

def load_model_preds(model_names, model_weights):
    for i, (k, vs) in enumerate(model_names.items()):
        for j, v in enumerate(vs):
            pred = util.load_dump(f"../data/{v}/pred_test.dump").sort_values('uid').reset_index()
            if j==0:
                preds = pred
                preds['pred'] = preds.pred/len(vs)
            else:
                preds['pred'] = preds.pred + pred.pred/len(vs)
        if i==0:
            all_preds = preds
            all_preds['pred'] = all_preds.pred.apply(lambda x: x*model_weights[k])
        else:
            all_preds['pred'] = all_preds.pred + preds.pred.apply(lambda x: x*model_weights[k])
    return all_preds


def main():
    # workdir yaa
    os.system('ls')

    # prepare
    os.system(f'mkdir -p yaa/data/yaa && cd yaa/data/yaa && ln -s ../../../data/test_features.csv test_features.csv')
    os.system(f'mkdir -p yaa/data/yaa && cd yaa/data/yaa && ln -s ../../../data/submission_format.csv submission_format.csv')


    # code dir yaa/yaa
    os.chdir('yaa/yaa')

    #model_fpath = os.path.basename(glob(f"../data/*KF*")[-1])
    #model_name = model_fpath.split('_KF')[0]
    model_names = get_model_names()

    if torch.cuda.is_available():
        torch_dtype = 'float16'
        max_seq_len = 8192
    else:
        torch_dtype = 'float32'
        max_seq_len = 8
    #output = subprocess.run('ls ', shell=True, capture_output=True, text=True)
    #print(output.stdout)
    #print(output.stderr)
    os.system('ls -ltrh ../data/')
    os.system('ls -ltrh ../data/yaa/')
    os.system('ls -ltrh')
    for model_name, vs in model_names.items():
        kfids = " ".join([v.split("_KF")[-1] for v in vs])
        print(model_name, kfids)
        if model_name.startswith('AR'):
            if 'gemma'in model_name:
                vbs = 1
            elif 'llama' in model_name:
                vbs = 1
            else:
                vbs = 1
        else:
            vbs = 2
        cmd = f'python eval.py -kn {len(vs)} -kfids "{kfids}" -data_type test -ds yaa -vbs {vbs} -model_name {model_name} -dataloader_num_workers 4 -test_fname test_features.csv \
        -torch_dtype {torch_dtype} -use_fp16 -do_test -max_seq_len {max_seq_len} -sort_seq'
        print(cmd)
        os.system(cmd)

    #thrs, _, _ = util.load_dump(f"../data/{model_name}_KF0/thrs_train.dump")
    #preds = util.load_dump(f"../data/{model_name}_KF0/pred_test.dump")
    thrs, _, _ = util.load_dump(f"../data/thrs.dump")
    if len(model_names) == 1:
        model_weights = {list(model_names.keys())[0]: np.ones(len(util.binary_labels) + util.n_InjuryLocationType + util.n_WeaponType1)}
    elif os.path.exists('../data/model_weights.dump'):
        print('1111111111111', 'use model weights')
        model_weights = util.load_dump('../data/model_weights.dump')
    else:
        model_weights = {model_name: np.ones(len(util.binary_labels) + util.n_InjuryLocationType + util.n_WeaponType1)/len(model_names) for model_name in model_names.items()}
        ar_models = [model_name for model_name in model_names.keys() if model_name.startsiwht('AR')]
        for model_name, w in model_weights.items():
            if model_name.startsiwth('AR'):
                w[len(util.binary_labels):] = 0
            else:
                w[len(util.binary_labels):] = 1/(len(model_names)-len(ar_models))



    preds = load_model_preds(model_names, model_weights)
    preds = util.probs2preds(preds, thrs)
    preds['InjuryLocationType'] = preds['InjuryLocationType'] + 1
    preds['WeaponType1'] = preds['WeaponType1'] + 1
    sample_submission = pd.read_csv('../../data/submission_format.csv')
    preds = sample_submission[["uid"]].merge(preds[sample_submission.columns], on='uid')
    preds.to_csv('../../submission.csv', index=False)

if __name__ == "__main__":
    main()

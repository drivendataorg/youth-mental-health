import os, sys, logging
from glob import glob
from collections import defaultdict

import pandas as pd
import numpy as np

import util

logger = logging.getLogger(__name__)


def get_model_names(model_names):
    model_names = sorted(set([model_name.split("_KF")[0] for model_name in model_names.split()]))
    model_names = sorted([os.path.basename(fpath) for model_name in model_names for fpath in glob(f"../data/{model_name}_KF*")])
    return model_names

def main(model_names, step=0.05, model_weights=None):
    kf_preds = defaultdict(list)
    model_names = get_model_names(model_names)
    logger.info('model names %s', model_names)
    for model_name in model_names:
        m = model_name.split("_KF")[0]
        pred = util.load_dump(f"../data/{model_name}/pred_train.dump")
        kf_preds[m].append(pred)
    kf_preds = [[k, pd.concat(vs).sort_values('uid').reset_index(drop=True)] for k, vs in kf_preds.items()]
    model_names, kf_preds = zip(*kf_preds)
    for i, kf_pred in enumerate(kf_preds):
        if i==0:
            all_uids = set(kf_pred.uid.values)
        else:
            all_uids = all_uids.intersection(set(kf_pred.uid.values))
    kf_preds = [kf_pred[kf_pred.uid.isin(all_uids)].reset_index(drop=True) for kf_pred in kf_preds]
    if len(model_names)==1:
        model_weights = {model_names[0]:np.ones(len(util.binary_labels)+util.n_InjuryLocationType+util.n_WeaponType1)}
    elif model_weights is None:
        if os.path.exists('../data/model_weights.dump'):
            logger.info('use model weights ../data/model_weights.dump')
            model_weights = util.load_dump('../data/model_weights.dump')
        else:
            model_weights = {model_name: np.ones(len(util.binary_labels) + util.n_InjuryLocationType + util.n_WeaponType1) / len(model_names) for model_name in model_names.items()}
            ar_models = [model_name for model_name in model_names.keys() if model_name.startsiwht('AR')]
            for model_name, w in model_weights.items():
                if model_name.startsiwth('AR'):
                    w[len(util.binary_labels):] = 0
                else:
                    w[len(util.binary_labels):] = 1 / (len(model_names) - len(ar_models))

    for i, (model_name, kf_pred) in enumerate(zip(model_names, kf_preds)):
        if i==0:
            preds = kf_pred
            preds['pred'] = preds.pred.apply(lambda x: x*model_weights[model_name])
        else:
            preds['pred'] = preds.pred.values + kf_pred.pred.apply(lambda x: x*model_weights[model_name])
    thrs, best_preds, labels = util.search_thr(preds, step=step)
    return thrs, best_preds, labels

if __name__ == "__main__":
    util.set_logger()
    if len(sys.argv)>2:
        step = float(sys.argv[2])
    else:
        step = 0.05
    thrs, best_preds, labels = main(sys.argv[1], step=step)
    fpath = "../data/thrs.dump"
    util.dump((thrs, best_preds, labels), fpath)
    logger.info('thrs saved to: %s', fpath)

import os, sys, logging
from glob import glob
from collections import defaultdict
import numpy as np
from functools import partial
from copy import deepcopy
import optuna

import pandas as pd

import util

logger = logging.getLogger(__name__)


def get_model_names(model_names):
    model_names = sorted(set([model_name.split("_KF")[0] for model_name in model_names.split()]))
    logger.info('model names:%s', model_names)
    model_names = sorted([os.path.basename(fpath) for model_name in model_names for fpath in glob(f"../data/{model_name}_KF*")])
    return model_names

def load_preds(model_names):
    kf_preds = defaultdict(list)
    for model_name in model_names:
        m = model_name.split("_KF")[0]
        pred = util.load_dump(f"../data/{model_name}/pred_train.dump")
        kf_preds[m].append(pred)
    kf_preds = [[k, pd.concat(vs).sort_values('uid').reset_index(drop=True)] for k, vs in kf_preds.items()]
    model_names, kf_preds = zip(*kf_preds)
    return model_names, kf_preds


def objective(trial, preds, labels, step=0.05):
    weights = []
    for i in range(len(preds)):
        weights.append(trial.suggest_float(f"w{i:02d}", 0, 1))
    weights = np.array(weights)
    weights = weights / weights.sum()
    for i, pred in enumerate(preds):
        if i == 0:
            new_pred = deepcopy(pred)
            new_pred = new_pred * weights[i]
        else:
            new_pred += pred * weights[i]
    _, s = util._search_thr(new_pred, labels, step=step)
    loss = 1 - s
    return loss


def objective_cat(trial, preds, labels):
    weights = []
    for i in range(len(preds)):
        weights.append(trial.suggest_float(f"w{i:02d}", 0, 1))
    weights = np.array(weights)
    weights = weights / weights.sum()
    for i, pred in enumerate(preds):
        if i == 0:
            new_pred = deepcopy(pred)
            new_pred = new_pred * weights[i]
        else:
            new_pred += pred * weights[i]

    s = util.f1_score(labels,  np.argmax(new_pred, axis=-1), average='micro')
    loss = 1 - s
    return loss

def main(model_names, step=0.05):
    model_names = get_model_names(model_names)
    logger.info('model names %s', model_names)
    model_names, preds = load_preds(model_names)
    if len(model_names)==1:
        logger.info("use default weight")
        model_weights = {model_names[0]:np.ones(len(util.binary_labels)+util.n_InjuryLocationType+util.n_WeaponType1)}
    else:
        labels = np.stack(preds[0].label.values, axis=0)
        all_probs = [np.stack(pred.pred.values, axis=0) for pred in preds]
        model_weights = defaultdict(list)
        all_names = util.binary_labels + ["InjuryLocationType", "WeaponType1"]
        for i in range(len(all_names)):
            if i==len(util.binary_labels):
                func = partial(objective_cat, preds=[probs[:, len(util.binary_labels):len(util.binary_labels)+util.n_InjuryLocationType] for probs in all_probs], labels=labels[:, i])
            elif i==(len(util.binary_labels)+1):
                func = partial(objective_cat, preds=[probs[:, -util.n_WeaponType1:] for probs in all_probs], labels=labels[:, i])
            else:
                func = partial(objective, preds=[probs[:, i] for probs in all_probs], labels=labels[:, i], step=step)
            sampler = optuna.samplers.TPESampler(seed=42+i)
            study = optuna.create_study(study_name="optimizing weights", direction="minimize", sampler=sampler)
            study.optimize(func, n_trials=10)
            best_params = sorted(study.best_params.items(), key=lambda x: x[0])
            best_params = [x[1] for x in best_params]

            s = sum(best_params)
            for p, model_name in zip(best_params, model_names):
                if i==len(util.binary_labels):
                    model_weights[model_name].extend([p/s]*util.n_InjuryLocationType)
                elif i==(len(util.binary_labels)+1):
                    model_weights[model_name].extend([p/s]*util.n_WeaponType1)
                else:
                    model_weights[model_name].append(p/s)
            print(all_names[i], study.best_value)



        for k, v in model_weights.items():
            model_weights[k] = np.array(v)
    return model_weights



if __name__ == "__main__":
    util.set_logger()
    if len(sys.argv)>2:
        step = float(sys.argv[2])
    else:
        step = 0.05
    model_weights = main(sys.argv[1], step=step)
    logger.info("model weights\n:%s", model_weights)
    fpath = "../data/model_weights.dump"
    util.dump(model_weights, fpath)
    logger.info('model_weights saved to: %s', fpath)

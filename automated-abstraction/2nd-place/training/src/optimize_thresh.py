import numpy as np
import pandas as pd
from helpers import find_optimal_threshold, get_score, convert_predictions
from cfg import DEFINE
from sklearn.metrics import f1_score, classification_report
import optuna
from optuna import Trial

pred_paths = [
            "predictions/longformer_large_mlmv2_noshuffle.npy", # 8538
            "predictions/deberta_large_manualdesc_1664.npy", # 8608
            # "predictions/flan_xl_lora.npy", # 8553
]

data_path = "../proc_data/train_5folds.csv"

df = pd.read_csv(data_path).sort_values(by=["fold", "uid"]).reset_index()
labels = df[list(DEFINE.column_dict.keys())].values
all_preds = []
for path in pred_paths:
    preds = np.load(path)
    all_preds.append(preds)
all_preds = np.array(all_preds)


def objective(trial: optuna.Trial):
    num_params = len(pred_paths)
    x = []
    for i in range(num_params):
        x.append(trial.suggest_float(f"x_{i}", 0, 1))
    w = []
    for i in range(num_params):
        w.append(x[i]/ sum(x))
    
    for i in range(num_params):
        trial.set_user_attr(f"w{i}", w[i])

    new_preds = []
    for i in range(len(all_preds)):
        new_preds.append(all_preds[i] * w[i])
    new_preds = np.sum(new_preds, axis=0)
    _, res = get_score(labels, new_preds)
    return res["zero_f1"]

# # Optimizer ensemble weights using Optuna
# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
# study.optimize(objective, n_trials=100)
# print(study.best_params)
# print(study.best_trial.user_attrs)

# # w = {"w0": 0.37, "w1": 0.63}
# final_preds = []
# for i in range(len(all_preds)):
#     final_preds.append(all_preds[i] * study.best_trial.user_attrs.get(f"w{i}"))
#     # final_preds.append(all_preds[i] * w[f"w{i}"])
# final_preds = np.sum(final_preds, axis=0)
    
final_preds = np.mean(all_preds, axis=0)
# saved_predictions = convert_predictions(final_preds, 0, {})
# df[list(DEFINE.column_dict.keys())] = saved_predictions
# df.to_csv("../proc_data/preds_5folds.csv")
# df[list(DEFINE.column_dict.keys())] = final_preds[:, :-16]
# df.to_csv("../proc_data/logits_5folds.csv")

binary_preds = final_preds[:, :21]
binary_labels = labels[:, :21]

var_f1 = []
thresh_list = []
max_thresh_list = []
for var in range(21):
    p = np.expand_dims(binary_preds[:, var], axis=1)
    l = np.expand_dims(binary_labels[:, var], axis=1)
    # # Optimize threshold for each binary variable
    # min_threshold, max_threshold, best_f1, zero_f1 = find_optimal_threshold(l, p, -1, 1, 41)
    # # Simply using threshold 0
    max_threshold = min_threshold = 0
    p[p >= min_threshold] = 1
    p[p < min_threshold] = 0

    binary_f1 = f1_score(
        l,
        p,
        average="binary",
    )

    print(classification_report(l, p))
    var_f1.append(binary_f1)
    print(f"{list(DEFINE.column_dict.keys())[var]:^30}: {binary_f1:.4f} +++ Thresh: {min_threshold:.4f}")
    thresh_list.append(round(min_threshold,2))
    max_thresh_list.append(round(max_threshold,2))


_, res = get_score(labels, np.mean(all_preds, axis=0))

print(f"Original binary F1: {res["zero_binary_f1s"]}")
print(f"Original F1: {res["zero_f1"]}")
print(f"Optimal binary F1: {sum(var_f1)/len(var_f1)}")
print(f"Optimal F1: {np.average([sum(var_f1)/len(var_f1), res["InjuryLocation"], res["WeaponType"]], weights=[21, 1, 1])}")

average_thresh_list = []
for mi, ma in zip(thresh_list, max_thresh_list):
    average_thresh_list.append((mi + ma)/2)
print(average_thresh_list)
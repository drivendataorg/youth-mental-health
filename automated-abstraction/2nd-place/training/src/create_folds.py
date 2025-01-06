import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

train_feature_path = "../data/train_features.csv"
train_label_path = "../data/train_labels.csv"
test_feature_path = "../data/smoke_test_features.csv"
test_label_path = "../data/smoke_test_labels.csv"
save_dir = "../proc_data/"
train_feature_df = pd.read_csv(train_feature_path).sort_values(by=["uid"])
train_label_df = pd.read_csv(train_label_path).sort_values(by=["uid"]).drop(columns=["uid"])
test_feature_df = pd.read_csv(test_feature_path).sort_values(by=["uid"])
test_label_df = pd.read_csv(test_label_path).sort_values(by=["uid"]).drop(columns=["uid"])

train_df = pd.concat([train_feature_df, train_label_df], axis=1)
test_df = pd.concat([test_feature_df, test_label_df], axis=1)

n_folds = 5
train_df.loc[:, "fold"] = -1
targets = train_df.drop(columns=["uid", "NarrativeLE", "NarrativeCME", "fold"], axis=1)
print(targets.head())
mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True)
for fold, (train, val) in enumerate(mskf.split(X=train_df, y=targets)):
    train_df.loc[val, "fold"] = fold

train_df.to_csv(os.path.join(save_dir, f"train_{n_folds}folds.csv"), index=False)
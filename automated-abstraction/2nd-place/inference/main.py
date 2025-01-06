import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
import pandas as pd
import torch
import gc
import numpy as np

from torch.utils.data import DataLoader
from cfg import PRED_CFG, DEFINE
from helpers import convert_predictions, inference_fn, inference_flan_fn
from transformers import AutoTokenizer, DataCollatorWithPadding, T5ForConditionalGeneration, GenerationConfig, DataCollatorForSeq2Seq
from dataset import TestDataset, FlanTestDataset
from peft import PeftModel
from model import LongformerForMaskedLM, DebertaForMaskedLM, DebertaV1ForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(PRED_CFG.test_path)

PRED_CFG.tokenizer = AutoTokenizer.from_pretrained(PRED_CFG.path[0] + "tokenizer/")
# sort by length to speed up inference
test_df["tokenize_length"] =  [len(PRED_CFG.tokenizer(nar_le)["input_ids"]) + len(PRED_CFG.tokenizer(nar_cme)["input_ids"]) for (nar_le, nar_cme) in zip(test_df["NarrativeLE"].values, test_df["NarrativeCME"].values)]
test_df = test_df.sort_values(by=["tokenize_length"], ascending=True).reset_index(drop=True)

all_logits = []

for i in range(len(PRED_CFG.path)):
    PRED_CFG.tokenizer = AutoTokenizer.from_pretrained(PRED_CFG.path[i] + "tokenizer/")

    if "flan" in PRED_CFG.path[i]:
        model = T5ForConditionalGeneration.from_pretrained(PRED_CFG.pretrain_model)
        config = GenerationConfig.from_pretrained(PRED_CFG.pretrain_model)
        config.max_new_tokens = 80
        config.truncation = True
        model.generation_config = config
        model = PeftModel.from_pretrained(model, PRED_CFG.path[i])

        data_collator = DataCollatorForSeq2Seq(
            PRED_CFG.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        test_dataset = FlanTestDataset(PRED_CFG, test_df, list(DEFINE.column_dict.keys()))
        test_loader = DataLoader(test_dataset,
                        batch_size=PRED_CFG.flan_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                        num_workers=PRED_CFG.num_workers, pin_memory=True, drop_last=False)
        logits = inference_flan_fn(test_loader, model, device)
        # print(logits.shape)
    else:
        if "manualdesc" in PRED_CFG.path[i]:
            binary_dict = {k: v for (k, v) in DEFINE.column_dict_manual.items() if k not in ["InjuryLocationType", "WeaponType1"]}
        else:
            binary_dict = {k: v for (k, v) in DEFINE.column_dict.items() if k not in ["InjuryLocationType", "WeaponType1"]}
        
        # test_df = test_df.rename(columns=DEFINE.column_dict)
        test_dataset = TestDataset(PRED_CFG, test_df, binary_dict, DEFINE.il_dict, DEFINE.wt_dict)
        test_loader = DataLoader(test_dataset,
                        batch_size=PRED_CFG.batch_size,
                        shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer=PRED_CFG.tokenizer, padding="longest"),
                        num_workers=PRED_CFG.num_workers, pin_memory=True, drop_last=False)

        config = torch.load(PRED_CFG.path[i] + "config.pth")

        if "deberta" in PRED_CFG.path[i]:
            model = DebertaForMaskedLM(config=config)
        if "long" in PRED_CFG.path[i]: 
            model = LongformerForMaskedLM(config=config)
        if "dv1" in PRED_CFG.path[i]:
            model = DebertaV1ForMaskedLM(config=config)
        for filepath in os.listdir(PRED_CFG.path[i]):
            if filepath.endswith("_best.pth"):
                checkpoint_path = os.path.join(PRED_CFG.path[i], filepath)
        model.load_state_dict(torch.load(checkpoint_path)["model"])

        logits = inference_fn(test_loader, model, config, PRED_CFG.batch_size, device)
        # print(logits.shape)
    all_logits.append(logits)  

    del model, logits
    torch.cuda.empty_cache()
    gc.collect()
    # print(logits.shape)
    
all_logits = np.array(all_logits)
# all_logits = np.mean(all_logits, axis=0)
# print(all_logits.shape)

ws = [0.126, 0.126, 0.126, 0.126, 0.126, 0.074, 0.074, 0.074, 0.074, 0.074]
# ws = [0.62/3, 0.62/3, 0.62/3, 0.38/3, 0.38/3, 0.38/3]

# ws = [0.078, 0.078, 0.078, 0.078, 0.078, 0.04, 0.04, 0.04, 0.04, 0.04, 0.082, 0.082, 0.082, 0.082, 0.082]
# # ws = [0.082, 0.082, 0.082, 0.082, 0.082, 0.118, 0.118, 0.118, 0.118, 0.118]
# ws = [0.066, 0.066, 0.066, 0.066, 0.066, 0.014, 0.014, 0.014, 0.014, 0.014, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.08, 0.08, 0.08]

weighted_logits = []
for i in range(len(all_logits)):
    weighted_logits.append(all_logits[i] * ws[i])
all_logits = np.sum(weighted_logits, axis=0)

# 
#1 threshold_list = [-0.4, -0.1, -0.35, 0.75, 0.4, 0.85, 0.15, 0.05, -0.1, -1.0, -0.8, 0.1, -1.0, 0.35, -1.0, 0.15, 0.1, -0.2, 0.25, -0.5, 0.0]
#2 threshold_list = [0.75, 0.3, -0.2, 0.575, 0.45, -0.05, 0.3, 0.525, 0.6, -0.175, -0.525, -0.375, -0.55, -0.65, -0.375, -0.175, -0.25, -0.5, 0.05, -0.5, -0.05]
#3 [0.45, -0.15, 0.0, 0.775, 0.25, 0.2, 0.65, -0.025, 0.1, 0.0, -0.85, -0.2, -1.0, 0.8, 0.05, 0.175, 0.0, 0.25, 0.0, -0.15, -0.7]

threshold_list = 21 * [0]
# threshold_list = [0.35, -0.35, 0.6, -0.35, 0.5, -0.05, 0.45, 0.55, 0.45, 0.725, -0.725, 0.15, -1.0, 0.475, 0.7, -0.95, -0.05, 0.6, -0.65, -0.05, 0.25]
# threshold_list = [-0.35, -0.2, 0.75, 0.375, 0.25, -0.05, 0.9, 0.85, 0.8, -0.15, -0.325, -0.45, -0.95, 0.45, 0.25, -0.45, -0.2, 0.15, 0.25, -0.25, 0.35]
# softer_threshold_list = [th/2 for th in threshold_list]

preds = convert_predictions(all_logits, threshold_list, DEFINE.column_dict)

test_df = test_df[["uid"]]
test_df[list(DEFINE.column_dict.keys())] = preds
test_df = test_df.sort_values(by=["uid"]).set_index("uid")
test_df.to_csv("submission.csv", index=True)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from cfg import CFG, PRED_CFG, DEFINE
from helpers import convert_predictions, inference_fn, average_f1
from transformers import AutoTokenizer, DataCollatorWithPadding
from dataset import TestDataset
from model import LongformerForMaskedLM, DebertaForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_pred_dir = "./predictions/"
os.makedirs(save_pred_dir, exist_ok=True)

# binary_dict = {k: v for (k, v) in DEFINE.column_dict.items() if k not in ["InjuryLocationType", "WeaponType1"]}
binary_dict = {k: v for (k, v) in DEFINE.column_manual_dict.items() if k not in ["InjuryLocationType", "WeaponType1"]}


for model_name in PRED_CFG.models.keys():
    model_preds = []
    for fold in range(5):
        df = pd.read_csv(CFG.data_path)
        df = df.rename(columns=DEFINE.column_dict)
        test_df = df[df["fold"] == fold].reset_index()
        print(test_df)
        print(f"Number of validation data: {test_df.shape[0]}")

        PRED_CFG.tokenizer = AutoTokenizer.from_pretrained(PRED_CFG.models[model_name][fold] + "tokenizer/")
        test_dataset = TestDataset(PRED_CFG, test_df, binary_dict, DEFINE.il_dict, DEFINE.wt_dict)
        test_loader = DataLoader(test_dataset,
                        batch_size=PRED_CFG.batch_size,
                        shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer=PRED_CFG.tokenizer, padding="longest"),
                        num_workers=PRED_CFG.num_workers, pin_memory=True, drop_last=False)

        config = torch.load(PRED_CFG.models[model_name][fold] + "config.pth")
        if "long" in model_name:
            model = LongformerForMaskedLM(config=config)
        if "deberta" in model_name:
            model = DebertaForMaskedLM(config=config)
        model.load_state_dict(torch.load(PRED_CFG.models[model_name][fold] + f"fold{fold}_best.pth")["model"])
        logits = inference_fn(test_loader, model, config, PRED_CFG.batch_size, device)
        model_preds.append(logits)
    model_preds = np.array(model_preds)
    model_preds = model_preds.reshape(model_preds.shape[0] * model_preds.shape[1], model_preds.shape[2])
    np.save(os.path.join(save_pred_dir, f"{model_name}.npy"), model_preds)

    preds = convert_predictions(model_preds, 0, DEFINE.column_dict)


label_df = pd.read_csv(CFG.data_path).sort_values(by=["fold", "uid"])
pred_df = label_df[["uid"]]
pred_df[list(DEFINE.column_dict.keys())] = preds
pred_df = pred_df.sort_values(by=["uid"]).set_index("uid")

label_df = label_df.drop(columns=["fold", "NarrativeLE", "NarrativeCME"]).sort_values(by=["uid"]).set_index("uid")


results, average = average_f1(pred_df, label_df)
print(results)
print(average)
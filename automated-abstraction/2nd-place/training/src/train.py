import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CURL_CA_BUNDLE"] = ""
import gc
import time
import pandas as pd

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from cfg import CFG, DEFINE
from model import LongformerForMaskedLM, DebertaForMaskedLM
from transformers import AutoTokenizer, AutoConfig

from utils import seed_everything, get_logger
from helpers import get_class_score, get_optimizer_params, get_scheduler, train_fn, valid_fn, get_score, train_fn_ema
from dataset import TrainDataset

os.makedirs("./saved_models/", exist_ok=True)
LOGGER = get_logger(CFG.log_file_name)
seed_everything(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOGGER.info(f"========== fold: {CFG.fold} training ==========")
df = pd.read_csv(CFG.data_path)
# Select template
if CFG.manual:
    df = df.rename(columns=DEFINE.column_manual_dict)
else:
    df = df.rename(columns=DEFINE.column_dict)
# Select fold or train on all data
if CFG.fold != 5:
    train_df = df[df["fold"] != CFG.fold].reset_index()
    valid_df = df[df["fold"] == CFG.fold].reset_index()
else:
    train_df = df
    valid_df = df[df["fold"] == 0].reset_index()
print(f"Number of training data: {train_df.shape[0]}")
print(f"Number of validation data: {valid_df.shape[0]}")

if CFG.manual:
    valid_labels = valid_df[list(DEFINE.column_manual_dict.values())].values
    binary_dict = {k: v for (k, v) in DEFINE.column_manual_dict.items() if k not in ["InjuryLocationType", "WeaponType1"]}
else:
    valid_labels = valid_df[list(DEFINE.column_dict.values())].values
    binary_dict = {k: v for (k, v) in DEFINE.column_dict.items() if k not in ["InjuryLocationType", "WeaponType1"]}

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
tokenizer.save_pretrained(CFG.output_dir + "tokenizer/")
CFG.tokenizer = tokenizer

train_dataset = TrainDataset(CFG, train_df, binary_dict, DEFINE.il_dict, DEFINE.wt_dict, CFG.shuffle)
valid_dataset = TrainDataset(CFG, valid_df, binary_dict, DEFINE.il_dict, DEFINE.wt_dict, False)

train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.batch_size * 2,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

config = AutoConfig.from_pretrained(CFG.model_name)
config.no_id = train_dataset.no_id
config.yes_id = train_dataset.yes_id 
config.mask_token_id = train_dataset.mask_token_id
if "ongformer" in CFG.model_name:
    model = LongformerForMaskedLM.from_pretrained(CFG.model_name, config=config)
if "v3" in CFG.model_name:
    print("DeBERTA-v3")
    model = DebertaForMaskedLM.from_pretrained(CFG.model_name, config=config)

torch.save(model.config, os.path.join(CFG.output_dir, "config.pth"))
model.to(device)

optimizer_parameters = get_optimizer_params(model,
                                            encoder_lr=CFG.encoder_lr,
                                            decoder_lr=CFG.decoder_lr,
                                            weight_decay=CFG.weight_decay)

optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
num_train_steps = int(len(train_df) / CFG.batch_size * CFG.epochs)
scheduler = get_scheduler(CFG, optimizer, num_train_steps)

criterion = CrossEntropyLoss() 
best_score = 0

# # EMA settings
# ema_decay = 0.99  # Decay rate for EMA

# # Initialize EMA weights as a copy of the model's weights
# ema_model = copy.deepcopy(model)
# for param in ema_model.parameters():
#     param.requires_grad = False  # EMA parameters shouldn't be updated by gradients


for epoch in range(CFG.epochs):
    start_time = time.time()
    # train
    avg_loss = train_fn(train_loader, config, model, criterion, optimizer, epoch, scheduler, device)
    avg_val_loss, predictions = valid_fn(valid_loader, config, model, criterion, device)

    # ema + rdrop training
    # avg_loss, ema_model = train_fn_ema(train_loader, config, model, criterion, optimizer, epoch, scheduler, device, ema_model, ema_decay)
    # avg_val_loss, predictions = valid_fn(valid_loader, config, ema_model, criterion, device)
    final_predictions, scores = get_score(valid_labels, predictions)
    scores = get_class_score(valid_labels, predictions, scores)

    score = scores["zero_f1"]
    elapsed = time.time() - start_time

    LOGGER.info(f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s")
    LOGGER.info(f'Epoch {epoch+1} - Scores: ')
    for k, v in scores.items():
        LOGGER.info(f"{k:^30}: {v:.4f}")
    if score > best_score:
        print("SAVED")
        best_score = score
        LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
        torch.save({'model': model.state_dict(),
        # torch.save({'model': ema_model.state_dict(),
                    'predictions': final_predictions},
                    CFG.output_dir + f"fold{CFG.fold}_best.pth")

torch.cuda.empty_cache()
gc.collect()
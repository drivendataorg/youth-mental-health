# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    apex=True
    print_freq=20
    num_workers=4
    model="microsoft/deberta-v3-large"
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=1.0
    num_warmup_steps=0
    epochs=5
    encoder_lr=5e-5
    decoder_lr=5e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=4
    max_len=2048
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    train=True
    
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import argparse
import pickle

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--n_folds', type=int, default=4)
args = parser.parse_args()

CFG.seed = CFG.seed + args.fold
device = f'cuda:{args.fold%2}'
CFG.n_fold = args.n_folds
CFG.trn_fold=[args.fold]

# NOTE - change data = pd.read_csv('data.csv')
data = pd.read_csv('data/train_features.csv').iloc[:100]
label_df = pd.read_csv('data/train_labels.csv').iloc[:100]
LABEL_NAMES = list(label_df.columns)
LABEL_NAMES.remove('uid')
data = data.merge(label_df, how='inner', on='uid')
    
data['Narrative'] = data['NarrativeLE'] + '\n' + data['NarrativeCME']
train_data = data[['Narrative']+LABEL_NAMES]
data = train_data.sample(frac=1, random_state=42+2).reset_index(drop=True)
skf = KFold(n_splits=4, shuffle=True, random_state=42)
for i, (_, val_index) in enumerate(skf.split(data)):
     data.loc[val_index, "fold"] = i
     
sub_train_data = pd.read_csv('data/data_with_preds10151_10152.csv')
# if using a version with predictions, remove
df_pred_cols = list(sub_train_data.filter(regex=r"pred_[0-9]|Unnamed", axis=1).columns)
if df_pred_cols:
    sub_train_data = sub_train_data.drop(columns=df_pred_cols)
    
# LABEL_NAMES = list(data.columns)
# LABEL_NAMES.remove('Narrative')
# LABEL_NAMES.remove('fold')
CFG.target_cols = LABEL_NAMES

# ====================================================
# Utils
# ====================================================
def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

train = data

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['Narrative'].values
        self.labels = df[LABEL_NAMES].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item]).to(torch.long)
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        dtype = last_hidden_state.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(dtype)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9).to(dtype)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        """
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        n_layers = len(self.model.encoder.layer)
        for param in self.model.encoder.layer[:n_layers//2].parameters():
            param.requires_grad = False
        """ 
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, (len(CFG.target_cols)-2)*2+6+12)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output
    
# ====================================================
# Loss
# ====================================================
import torch
from torch import nn

from sklearn.metrics import f1_score

def average_f1(predictions, labels):
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    binary_f1 = f1_score(
        labels[:, :-2],
        predictions[:, :-2],
        average="macro",
    )
    f1s = [binary_f1]

    # Calculate F1 score for each categorical variable
    f1s.append(f1_score(labels[:, -2], predictions[:, -2], average="micro"))
    f1s.append(f1_score(labels[:, -1], predictions[:, -1], average="micro"))
    return np.average(f1s, weights=[labels.shape[1]-2, 1, 1])

def average_f1_cls(predictions, labels):
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    binary_f1 = f1_score(
        labels[:, :-2],
        predictions[:, :-2],
        average=None,
    )
    f1s = list(binary_f1)

    # Calculate F1 score for each categorical variable
    f1s.append(f1_score(labels[:, -2], predictions[:, -2], average='micro'))
    f1s.append(f1_score(labels[:, -1], predictions[:, -1], average='micro'))
    return f1s

class F1Loss(nn.Module):
    def __init__(self, average='macro', epsilon=1e-7):
        """
        Combined F1 Loss function with support for Macro and Micro averaging.

        Args:
            average (str): Averaging method, either 'macro' or 'micro'.
            epsilon (float): Small value to prevent division by zero.
        """
        super(F1Loss, self).__init__()
        assert average in ['macro', 'micro'], "average must be either 'macro' or 'micro'"
        self.average = average
        self.epsilon = epsilon

    def forward(self, labels, logits):
        """
        Compute the F1 loss based on the chosen average method.

        Args:
            logits (Tensor): Predicted raw scores or logits from the model, shape [batch_size, num_classes].
            labels (Tensor): True class labels, shape [batch_size].

        Returns:
            Loss based on 1 - F1 score using the specified average method.
        """

        # Convert labels to one-hot encoding based on logits shape
        if len(labels.shape) == 1:
            num_classes = logits.shape[-1]
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes).float()
            probs = logits
        else:
            num_classes = labels.shape[-1]
            labels_one_hot = labels.float()
            probs = logits[..., -1]
        
        if self.average == 'macro':
            tp = torch.sum(labels_one_hot*probs, dim=0)
            fp = torch.sum((1-labels_one_hot)*probs, dim=0)
            fn = torch.sum(labels_one_hot*(1-probs), dim=0)
            tp_ = tp + self.epsilon
            
            precision = tp_ / (tp_ + fp)
            recall = tp_ / (tp_ + fn)
            
            # Calculate F1 score for each class and return the macro average as the loss
            f1_per_class = (precision+recall) / 2
            macro_f1_loss = 1 - f1_per_class.mean()  # 1 - Macro F1 score as loss
            return macro_f1_loss

        elif self.average == 'micro':   
            tp = torch.sum(labels_one_hot*probs, dim=-1)
            fp = torch.sum((1-labels_one_hot)*probs, dim=-1)
            fn = torch.sum(labels_one_hot*(1-probs), dim=-1)
            tp_ = tp + self.epsilon
            
            precision = tp_ / (tp_ + fp)
            recall = tp_ / (tp_ + fn)
            
            # Calculate micro F1 score
            micro_f1 = (precision+recall) / 2

            # Return 1 - Micro F1 score as loss
            return 1 - micro_f1.mean()  # Average over classes
        
macro_f1_score = F1Loss(average='macro')
micro_f1_score = F1Loss(average='micro')
def average_f1_loss(preds, labels):
    preds1, preds2, preds3 = format_preds(preds)
    preds1 = preds1.softmax(dim=-1)
    preds2 = preds2.softmax(dim=-1)
    preds3 = preds3.softmax(dim=-1)
    binary_f1 = macro_f1_score(labels[:, :-2], preds1)
    f1s = [binary_f1]

    # Calculate F1 score for each categorical variable
    f1s.append(micro_f1_score(labels[:, -2]-1, preds2))
    f1s.append(micro_f1_score(labels[:, -1]-1, preds3))
    f1s[0] = f1s[0] * (labels.shape[1]-2)
    return sum(f1s) / labels.shape[1]

def format_preds(preds):
    preds3 = preds[:, -12:]
    preds2 = preds[:, -18:-12]
    preds1 = preds[:, :-18]
    preds1 = preds1.reshape([preds1.shape[0], preds1.shape[1]//2, 2])
    
    #print(len(preds1))
    return preds1, preds2, preds3

LOSS_FN1 = torch.nn.BCEWithLogitsLoss()
LOSS_FN2 = torch.nn.CrossEntropyLoss()
def base_loss(preds, labels):
    preds1, preds2, preds3 = format_preds(preds)
    loss1 = LOSS_FN2(preds1.reshape([-1, 2]), labels[:, :-2].reshape([-1])) * (labels.shape[1]-2)
    loss2 = LOSS_FN2(preds2, labels[:, -2]-1)
    loss3 = LOSS_FN2(preds3, labels[:, -1]-1)
    loss = (loss1+loss2+loss3) / labels.shape[1]
    return loss

def mixed_loss(preds, labels):
    loss = base_loss(preds, labels) + average_f1_loss(preds, labels)
    return loss / 2
    
def find_thrs(labels, preds):
    preds1, preds2, preds3 = format_preds(preds)
    preds1 = torch.tensor(preds1).softmax(dim=-1).numpy()
    preds = np.zeros(labels.shape)
    preds[:, -2] = np.argmax(preds2, axis=-1)
    preds[:, -1] = np.argmax(preds3, axis=-1)
    thrs = np.zeros([labels.shape[1]-2]) + 0.5
    for i in range(labels.shape[1]-2):
        best_f1 = 0
        for thr in range(10, 90, 5):
            thr = thr / 100
            preds[:, i] = (preds1[:, i, -1]>thr).astype(int)
            f1 = average_f1(preds, labels)
            if best_f1 < f1:
                thrs[i] = thr
                best_f1 = f1
            #print(i, thr, f1)
        preds[:, i] = (preds1[:, i, -1]>thrs[i]).astype(int)
    return thrs
    
def get_score(labels, preds, thrs=None):
    preds1, preds2, preds3 = format_preds(preds)
    preds = np.zeros(labels.shape)
    if thrs is None:
        preds[:, :-2] = np.argmax(preds1, axis=-1)
    else:
        preds1 = torch.tensor(preds1).softmax(dim=-1).numpy()
        for i, thr in enumerate(thrs):
            preds[:, i] = (preds1[:, i, -1]>thr).astype(int)
    preds[:, -2] = np.argmax(preds2, axis=-1) + 1
    preds[:, -1] = np.argmax(preds3, axis=-1) + 1
    f1s = average_f1_cls(preds, labels)
    f1 = average_f1(preds, labels)
    
    return f1, f1s

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        loss_ga = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss_ga).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            LOGGER.info(f'Fold {fold} '
                  'Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').to(torch.float32).numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info(f'Fold {fold} '
                  'EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = pd.concat([folds[folds['fold'] != fold], sub_train_data]).reset_index(drop=True)
    train_dataset = TrainDataset(CFG, train_folds)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    valid_dataset = TrainDataset(CFG, valid_folds)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    #torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = base_loss # RMSELoss(reduction="mean")
    
    best_score = 0
    best_new_score = 0
    #model = torch.nn.DataParallel(model)
    
    for epoch in range(CFG.epochs):
        if epoch == round(CFG.epochs/3):
            train_folds = folds[folds['fold']!=fold].reset_index(drop=True)
            train_dataset = TrainDataset(CFG, train_folds)
            train_loader = DataLoader(train_dataset,
                                      batch_size=CFG.batch_size,
                                      shuffle=True,
                                      num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        model = model.to(torch.float16)
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        #with open(f'fold{args.fold}_pred_cache.pkl', 'wb') as f:
        #    pickle.dump([valid_labels, predictions], f)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)
        scores = {l:s for l, s in zip(LABEL_NAMES, scores)}
        thrs = find_thrs(valid_labels, predictions)
        new_score, new_scores = get_score(valid_labels, predictions, thrs=thrs) 
        new_scores = {l:s for l, s in zip(LABEL_NAMES, new_scores)}
        elapsed = time.time() - start_time

        LOGGER.info(f'Fold {fold} Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Fold {fold} Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        LOGGER.info(f'Fold {fold} Epoch {epoch+1} - New Score: {new_score:.4f}  New Scores: {new_scores}, thrs: {thrs}')
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Fold {fold} Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'thrs': thrs},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
        if best_new_score < new_score:
            best_new_score = new_score
            LOGGER.info(f'Fold {fold} Epoch {epoch+1} - Save Best New Score: {best_new_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'thrs': thrs},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_newbest.pth")
        model = model.to(torch.float32)

    #predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
    #                         map_location=torch.device('cpu'))['predictions']
    #valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    #return valid_folds
    
if __name__ == '__main__':
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)
            """
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
            """
        """
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
        """
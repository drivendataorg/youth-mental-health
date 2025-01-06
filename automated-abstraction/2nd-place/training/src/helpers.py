import time
import torch
import numpy as np

from cfg import CFG, DEFINE
from tqdm import tqdm
from utils import AverageMeter, timeSince
from dataset import collate
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if "lstm_head" not in n and not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if "lstm_head" not in n and any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        # {'params': [p for n, p in model.named_parameters() if "lm_head" in n],
        {'params': [p for n, p in model.named_parameters() if "lstm_head" in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

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
    

def train_fn(train_loader, config, model, criterion, optimizer, epoch, scheduler, device):
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=CFG.apex)
    losses = AverageMeter()
    start = time.time()
    global_step = 0

    for step, (inputs, labels, prefix_token_len) in enumerate(train_loader):
        inputs = collate(inputs, prefix_token_len)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.amp.autocast("cuda", enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1, config.vocab_size), labels.view(-1))

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        if CFG.batch_scheduler:
            scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print("Epoch: [{0}][{1}/{2}] "
                  "Elapsed {remain:s} "
                  "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                  "Grad: {grad_norm:.4f} "
                  "LR: {lr:.8f}"
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
    return losses.avg


def find_optimal_threshold(labels, preds, start, end, num):
    thresholds = np.linspace(start, end, num)
    min_threshold = 0
    max_threshold = 0

    # Calculate with threshold = 0
    new_preds = []
    for pred in preds:
        new_pred = (pred >= 0).astype(int)
        new_preds.append(new_pred)
    zero_f1 = f1_score(new_preds, labels, average="macro")

    best_f1 = 0
    for th in thresholds:
        new_preds = []
        for pred in preds:
            new_pred = (pred >= th).astype(int)
            new_preds.append(new_pred)
        f1 = f1_score(new_preds, labels, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            min_threshold = th

    best_f1 = 0
    thresholds = thresholds.tolist()
    thresholds.reverse()
    for th in thresholds:
        new_preds = []
        for pred in preds:
            new_pred = (pred >= th).astype(int)
            new_preds.append(new_pred)
        f1 = f1_score(new_preds, labels, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            max_threshold = th

    return min_threshold, max_threshold, best_f1, zero_f1
    

def get_score(labels, preds):
    binary_preds = preds[:, :21].copy()
    binary_labels = labels[:, :21].copy()

    min_threshold, max_threshold, best_f1, zero_f1 = find_optimal_threshold(binary_labels, binary_preds, -2, 2, 40)
    binary_preds[binary_preds >= min_threshold] = 1
    binary_preds[binary_preds < min_threshold] = 0

    binary_f1 = f1_score(
        binary_labels,
        binary_preds,
        average="macro",
    )

    il_preds = preds[:, 21:27]
    il_labels = labels[:, 21]
    il_preds = np.argmax(il_preds, axis=1) + 1
    wt_preds = preds[:, 27:]
    wt_labels = labels[:, 22]
    wt_preds = np.argmax(wt_preds, axis=1) + 1

    il_f1 = f1_score(il_labels, il_preds, average="micro")
    wt_f1 = f1_score(wt_labels, wt_preds, average="micro")
    f1s = [binary_f1, il_f1, wt_f1]
    zero_f1s = [zero_f1, il_f1, wt_f1]
    il_preds = np.expand_dims(il_preds, axis=1)
    wt_preds = np.expand_dims(wt_preds, axis=1)
    predictions = np.concatenate((np.concatenate((binary_preds, il_preds), axis=1), wt_preds), axis=1)

    scores = {}
    scores["InjuryLocation"] = il_f1
    scores["WeaponType"] = wt_f1
    scores["zero_binary_f1s"] = zero_f1
    scores["zero_f1"] = np.average(zero_f1s, weights=[21, 1, 1])
    scores["binary_f1s"] = binary_f1
    scores["best_f1"] = np.average(f1s, weights=[21, 1, 1])
    scores["best_threshold"] = round((min_threshold + max_threshold)/2, 4)
    return predictions, scores


def get_class_score(labels, preds, scores):
    binary_preds = preds[:, :21]
    binary_labels = labels[:, :21]

    var_f1 = []
    for var in range(21):
        p = np.expand_dims(binary_preds[:, var], axis=1)
        l = np.expand_dims(binary_labels[:, var], axis=1)

        p[p >= 0] = 1
        p[p < 0] = 0

        binary_f1 = f1_score(
            l,
            p,
            average="binary",
        )
        var_f1.append(binary_f1)
        scores[list(DEFINE.column_dict.keys())[var]] = round(binary_f1, 4)
    del binary_preds
    del binary_labels
    return scores


def inference_fn(test_loader, model, config, batch_size, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            y_preds = y_preds[:, config.yes_id] - y_preds[:, config.no_id]
            y_preds = y_preds.view(batch_size, -1)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

def convert_predictions(preds, threshold, column_dict):
    binary_preds = preds[:, :21]
    binary_preds = (binary_preds >= threshold).astype(int)
    binary_preds = binary_preds.astype(int)

    il_preds = preds[:, 21:27]
    il_preds = np.argmax(il_preds, axis=1) + 1
    wt_preds = preds[:, 27:]
    wt_preds = np.argmax(wt_preds, axis=1) + 1
    il_preds = np.expand_dims(il_preds, axis=1).astype(int)
    wt_preds = np.expand_dims(wt_preds, axis=1).astype(int)
    final_preds = np.concatenate((np.concatenate((binary_preds, il_preds), axis=1), wt_preds), axis=1)

    return final_preds

def average_f1(predictions, labels):
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

    results = {}
    for var in BINARY_VARS:
        results[var] = f1_score(
            labels[var],
            predictions[var],
            average="binary",
        )

    f1s = [binary_f1]
    print(f"Binary F1: {binary_f1}")
    # Calculate F1 score for each categorical variable
    for cat_col in CATEGORICAL_VARS:
        results[cat_col] = f1_score(labels[cat_col], predictions[cat_col], average="micro")
        print(f"{cat_col} {f1_score(labels[cat_col], predictions[cat_col], average='micro')}")
        f1s.append(f1_score(labels[cat_col], predictions[cat_col], average="micro"))

    return results, np.average(f1s, weights=[len(BINARY_VARS), 1, 1])



# Function to update EMA weights
def update_ema(model, ema_model, decay):
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = decay * ema_param.data + (1 - decay) * model_param.data

  
def train_fn_ema(train_loader, config, model, criterion, optimizer, epoch, scheduler, device, ema_model, ema_decay):
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, (inputs, labels, prefix_token_len) in enumerate(train_loader):
        inputs = collate(inputs, prefix_token_len)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.amp.autocast("cuda", enabled=CFG.apex):
            y_preds = model(inputs)
            loss_1 = criterion(y_preds.view(-1, config.vocab_size), labels.view(-1))

            y_preds_2 = model(inputs)
            loss_2 = criterion(y_preds_2.view(-1, config.vocab_size), labels.view(-1))

            kl_loss = compute_kl_loss(y_preds, y_preds_2)
            loss = 0.5 * (loss_1 + loss_2) + 0.1 * kl_loss

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        update_ema(model, ema_model, ema_decay)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        if CFG.batch_scheduler:
            scheduler.step()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print("Epoch: [{0}][{1}/{2}] "
                  "Elapsed {remain:s} "
                  "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                  "Grad: {grad_norm:.4f} "
                  "LR: {lr:.8f}"
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
    return losses.avg, ema_model


def valid_fn(valid_loader, config, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()
    for step, (inputs, labels, prefix_token_len) in enumerate(valid_loader):
        inputs = collate(inputs, prefix_token_len)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1, config.vocab_size), labels.view(-1))
        
        losses.update(loss.item(), batch_size)
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print("Epoch: [{0}][{1}] "
                  "Elapsed {remain:s} "
                  "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                  .format(step, len(valid_loader),
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          loss=losses))
        
        y_preds = y_preds[:,config.yes_id] - y_preds[:, config.no_id]
        y_preds = y_preds.view(batch_size, -1)
        preds.append(y_preds.to("cpu").numpy())
    
    predictions = np.concatenate(preds) 
    
    return losses.avg, predictions
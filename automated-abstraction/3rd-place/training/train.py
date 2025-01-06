import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
from time import time
import random
import warnings
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from bitsandbytes.optim import AdamW8bit
import torch
from torch import nn

from sklearn.metrics import f1_score

import torch
from torch import nn
import transformers
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from peft import get_peft_config, prepare_model_for_kbit_training, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

from torch import nn
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from transformers import BitsAndBytesConfig

tqdm.pandas()

print(f'Torch Version: {torch.__version__}')

def run_code(model_cfg, fold):
    class CFG:
        NUM_EPOCHS = 5
        PRETRAIN_EPOCHS = 1
        BATCH_SIZE = model_cfg['batch_size']
        GRADIENT_ACCUM = 1
        DROPOUT = 0.0
        MODEL_NAME = model_cfg['base_model_path']
        SEED = model_cfg['seed'] 
        CV_SEED = model_cfg['cv_seed'] 
        NUM_WARMUP_RATE = 0.2
        NUM_CYCLES = 0.4
        LABEL_SMOOTHING = 0.1
        LR_MAX = 1e-4
        FREEZE_RATE = 0
        LORA_RANK = model_cfg['rank']
        LORA_ALPHA = model_cfg['alpha']
        #LORA_MODULES = ['o_proj', 'v_proj']
        MAX_LENGTH = model_cfg['max_length']
        N_FOLDS = 5
    
    CFG.SEED = CFG.SEED + fold
    CFG.FOLD = fold
    DEVICE = 'cuda'
    
    def set_seeds(seed):
        """Set seeds for reproducibility """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
           
    set_seeds(seed=CFG.SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    #tokenizer.padding_side = 'left'
    
    class PATHS:
        label_path = '/competition-cdc-narratives/data/final/aa_track/public/train_labels.csv'
        extra_data_folders = [
            'data/data_with_preds10151_10152.csv',
        ]
        semi_soft_data_path = [
            'data/11071_padright_p1_alldata_to_1106.csv',
            'data/11071_padright_p2_alldata_to_1106.csv',
        ]
    N_PARTS = CFG.N_FOLDS
        
    label_df = pd.read_csv(PATHS.label_path)
    LABEL_NAMES = list(label_df.columns)
    LABEL_NAMES.remove('uid')
    CFG.NUM_LABELS = (len(LABEL_NAMES)-2)+6+12
    
    SOFT_LABEL_THRS = [
        0.475, 0.4, 0.425, 0.35, 0.425, 0.4375, 0.4625, 0.3, 0.425, 0.3875, 0.35,
        0.4125, 0.3, 0.425, 0.4375, 0.375, 0.375, 0.3, 0.325, 0.3125, 0.35
    ]
        
    def get_labels_from_soft(soft_labels):
        output = np.zeros([soft_labels.shape[0], len(LABEL_NAMES)])
        labels1 = soft_labels[:, :-18]
        for i in range(labels1.shape[1]):
            output[:, i] = (labels1[:, i]>SOFT_LABEL_THRS[i]).astype(int)
        output[:, -2] = soft_labels[:, -18:-12].argmax(axis=-1) + 1
        output[:, -1] = soft_labels[:, -12:].argmax(axis=-1) + 1
        return output
    
    all_semi_data = pd.concat([pd.read_csv(path) for path in PATHS.semi_soft_data_path])
    semi_soft_data = all_semi_data[all_semi_data['source']!='base']
    SOFT_LABEL_NAMES = [name for name in list(semi_soft_data.columns) if 'pred_'==name[:5]]
    print(f'len(semi_soft_data): {len(semi_soft_data)}')
    semi_soft_data = semi_soft_data.sample(n=min([len(semi_soft_data), 12000]), 
                                           random_state=CFG.SEED+1).reset_index(drop=True)
    semi_soft_labels = semi_soft_data[SOFT_LABEL_NAMES].values
    semi_soft_data[LABEL_NAMES] = get_labels_from_soft(semi_soft_labels)
    semi_train_data = semi_soft_data[['Narrative']+LABEL_NAMES+SOFT_LABEL_NAMES]
    semi_train_data['fold'] = -1
    semi_train_data[['post_'+ln for ln in LABEL_NAMES]] = semi_train_data[LABEL_NAMES]
    
    sub_train_data = []
    for path in PATHS.extra_data_folders:
        sub_train_data = sub_train_data + [pd.read_csv(path)]
    sub_train_data = pd.concat(sub_train_data)
    print(f'len(sub_train_data): {len(sub_train_data)}')
    sub_train_data = sub_train_data.sample(n=4000, random_state=CFG.SEED).reset_index(drop=True)
    sub_train_data = sub_train_data[['Narrative']+LABEL_NAMES+SOFT_LABEL_NAMES]
    sub_train_data['fold'] = -1
    sub_train_data[['post_'+ln for ln in LABEL_NAMES]] = sub_train_data[LABEL_NAMES]
    
    cv_scores = {
        'DepressedMood': 0.7700170357751277, 'MentalIllnessTreatmentCurrnt': 0.8073394495412843, 
        'HistoryMentalIllnessTreatmnt': 0.8895104895104896, 'SuicideAttemptHistory': 0.9242053789731051, 
        'SuicideThoughtHistory': 0.8365508365508366, 'SubstanceAbuseProblem': 0.8166259168704156, 
        'MentalHealthProblem': 0.9227467811158797, 'DiagnosisAnxiety': 0.9411764705882353,
        'DiagnosisDepressionDysthymia': 0.9382022471910112, 'DiagnosisBipolar': 0.9606299212598425,
        'DiagnosisAdhd': 0.920353982300885, 'IntimatePartnerProblem': 0.9156626506024097, 
        'FamilyRelationship': 0.7404844290657439, 'Argument': 0.9237113402061857, 'SchoolProblem': 0.8,
        'RecentCriminalLegalProblem': 0.7142857142857142, 'SuicideNote': 0.897841726618705, 'SuicideIntentDisclosed': 0.76, 
        'DisclosedToIntimatePartner': 0.6947368421052632, 'DisclosedToOtherFamilyMember': 0.6243386243386243, 
        'DisclosedToFriend': 0.6265060240963856
    }
    cv_scores = list(cv_scores.values())
    max_score = max(cv_scores)
    min_score = min(cv_scores)
    fix_rates = (np.array(cv_scores)-min_score) / (max_score-min_score)
    fix_rates = (1-fix_rates) * 0.2
    data_with_preds = pd.read_csv('data/10202_valpreds.csv')
    pred_cols = [col for col in list(data_with_preds.columns) if 'pred_'==col[:5]]
    preds = data_with_preds[pred_cols].values
    preds1 = preds[:, :-18]
    preds1_bool = (preds1>0.5).astype(int)
    
    data_df = all_semi_data[all_semi_data['source']=='base']
    data_df = data_df[['uid', 'Narrative']+SOFT_LABEL_NAMES]
    data_df = data_df.merge(label_df, how='inner', on='uid')
    train_data = data_df[['uid', 'Narrative']+LABEL_NAMES+SOFT_LABEL_NAMES]
    labels_df = train_data[LABEL_NAMES]
    labels1 = labels_df.values[:, :-2]
    diff_all = labels1 - preds1
    for i, ln in enumerate(LABEL_NAMES[:-2]):
        diff_l = diff_all[:, i]
        q = 1 - fix_rates[i]
        """
        diff_ids = np.where(diff_l>0)[0]
        diff = abs(diff_l[diff_ids])
        thr = np.quantile(diff, q=q)
        fix_ids = diff_ids[diff>thr]
        labels_df.loc[fix_ids, ln] = preds1_bool[fix_ids, i]
        print(f'label {ln} p changed: {len(fix_ids)/len(diff_l)}')
        """
        diff_ids = np.where(diff_l<0)[0]
        diff = abs(diff_l[diff_ids])
        thr = np.quantile(diff, q=q)
        fix_ids = diff_ids[diff>thr]
        labels_df.loc[fix_ids, ln] = preds1_bool[fix_ids, i]
        print(f'label {ln} n changed: {len(fix_ids)/len(diff_l)}')
        
    train_data[['post_'+ln for ln in LABEL_NAMES]] = labels_df
    data = train_data.sample(frac=1, random_state=CFG.CV_SEED+2).reset_index(drop=True)
    skf = KFold(n_splits=N_PARTS, shuffle=True, random_state=CFG.CV_SEED)
    for i, (_, val_index) in enumerate(skf.split(data)):
        data.loc[val_index, "fold"] = i
    
    class train_dataset():
        def __init__(self, data_df, is_train=True):
            self.texts = data_df['Narrative'].values
            if is_train:
                self.labels = data_df[['post_'+ln for ln in LABEL_NAMES]].values
            else:
                self.labels = data_df[LABEL_NAMES].values
            soft_labels = data_df[SOFT_LABEL_NAMES].values
            self.soft_labels3 = soft_labels[:, -12:]
            self.soft_labels2 = soft_labels[:, -18:-12]
            self.soft_labels1 = soft_labels[:, :-18]
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, item):
            report = self.texts[item]
            tokens = tokenizer(report+tokenizer.eos_token, max_length=CFG.MAX_LENGTH,
                               padding='longest', truncation=True, add_special_tokens=True)
            tokens['input_ids'] = torch.tensor(tokens['input_ids'])
            tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])
            
            label = torch.tensor(self.labels[item]).to(torch.long)
            soft_label1 = torch.tensor(self.soft_labels1[item])
            soft_label2 = torch.tensor(self.soft_labels2[item])
            soft_label3 = torch.tensor(self.soft_labels3[item])
            return tokens, label, soft_label1, soft_label2, soft_label3

    def collate(batch):
        tokens = tokenizer.pad([data[0] for data in batch], padding='longest')
        label = torch.stack([data[1] for data in batch], dim=0)
        soft_label1 = torch.stack([data[2] for data in batch], dim=0)
        soft_label2 = torch.stack([data[3] for data in batch], dim=0)
        soft_label3 = torch.stack([data[4] for data in batch], dim=0)
        return tokens['input_ids'], tokens['attention_mask'], label, soft_label1, soft_label2, soft_label3
    
    # Load model for classification with 3 target label
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.MODEL_NAME,
        num_labels=CFG.NUM_LABELS, device_map='auto')

    base_model.gradient_checkpointing_enable()
    base_model.config.pretraining_tp = 1 
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    def find_target_modules(model):
        # Initialize a Set to Store Unique Layers
        unique_layers = set()
        
        # Iterate Over All Named Modules in the Model
        for name, module in model.named_modules():
            # Check if the Module Type Contains 'Linear4bit'
            if "Linear" in str(type(module)):
                # Extract the Type of the Layer
                layer_type = name.split('.')[-1]
                if '_proj' in layer_type:
                    # Add the Layer Type to the Set of Unique Layers
                    unique_layers.add(layer_type)
    
        # Return the Set of Unique Layers Converted to a List
        return list(unique_layers)
    
    target_modules = find_target_modules(base_model)
    print(target_modules)
    
    n_layers = base_model.config.num_hidden_layers
    lora_config = LoraConfig(
        r=CFG.LORA_RANK,  # the dimension of the low-rank matrices
        lora_alpha = CFG.LORA_ALPHA, # scaling factor for LoRA activations vs pre-trained weight activations
        lora_dropout= CFG.DROPOUT, 
        bias='none',
        layers_to_transform=[i for i in range(n_layers) if i >= CFG.FREEZE_RATE*n_layers],
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules) # Only Use Output and Values Projection
    
    # Create LoRa Model
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)
    # Trainable Parameters
    model.print_trainable_parameters()
    
    class MyModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.base_model = model
            #self.head = model.score
            
        def forward(self, input_ids, attention_mask):
            output = self.base_model(input_ids, attention_mask=attention_mask)[0]
            return output
    
    my_model = MyModel(model).to(torch.float16)
    #my_model.load_state_dict(model_params['model'], strict=False)
    
    # Verfy The Trainable Layers
    MODEL_LAYERS_ROWS = []
    TRAINABLE_PARAMS = []
    N_TRAINABLE_PARAMS = 0
    
    for name, param in my_model.named_parameters():
        # Layer Parameter Count
        n_parameters = int(torch.prod(torch.tensor(param.shape)))
        # Only Trainable Layers
        if param.requires_grad:
            # Add Layer Information
            MODEL_LAYERS_ROWS.append({
                'param': n_parameters,
                'name': name,
                'dtype': param.data.dtype,
            })
            # Append Trainable Parameter
            TRAINABLE_PARAMS.append({ 'params': param })
            # Add Number Of Trainable Parameters"
            N_TRAINABLE_PARAMS += n_parameters
            
    print(pd.DataFrame(MODEL_LAYERS_ROWS).head())
    
    print(f"""
    ===============================
    N_TRAINABLE_PARAMS: {N_TRAINABLE_PARAMS:,}
    N_TRAINABLE_LAYERS: {len(TRAINABLE_PARAMS)}
    ===============================
    """)
    
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
                f1_per_class = (2*precision*recall) / (precision+recall)
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
                micro_f1 = (2*precision*recall) / (precision+recall)
    
                # Return 1 - Micro F1 score as loss
                return 1 - micro_f1.mean()  # Average over classes
            
    macro_f1_score = F1Loss(average='macro')
    micro_f1_score = F1Loss(average='micro')
    def average_f1_loss(preds, labels):
        preds1, preds2, preds3 = format_preds(preds)
        preds1 = preds1.sigmoid()
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
        #preds1 = preds1.reshape([preds1.shape[0], preds1.shape[1]//2, 2])
        
        #print(len(preds1))
        return preds1, preds2, preds3
    
    class BCEWithLogitsLoss(nn.Module):
        def __init__(self, label_smoothing=0.0, reduction='mean'):
            super(BCEWithLogitsLoss, self).__init__()
            assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
            self.label_smoothing = label_smoothing
            self.reduction = reduction
            self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)
    
        def forward(self, input, target):
            if self.label_smoothing > 0:
                positive_smoothed_labels = 1.0 - self.label_smoothing
                negative_smoothed_labels = self.label_smoothing
                target = target * positive_smoothed_labels + \
                    (1 - target) * negative_smoothed_labels
    
            loss = self.bce_with_logits(input, target)
            return loss
    
    def LOSS_FN_SOFT(preds, labels):
        p = F.log_softmax(preds, 1)
        loss = -(labels*p).sum() / (labels).sum()
        return loss
    
    LOSS_FN = BCEWithLogitsLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    LOSS_FN2 = torch.nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    LOSS_FN0 = torch.nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING, reduce=False)
    def base_loss(preds, labels, labels1, labels2, labels3):
        preds1, preds2, preds3 = format_preds(preds)
        loss1 = LOSS_FN(preds1, labels[:, :-2]) * (labels.shape[1]-2)
        loss2 = LOSS_FN2(preds2, labels[:, -2]-1)
        loss3 = LOSS_FN2(preds3, labels[:, -1]-1)
        loss = (loss1+loss2+loss3) / labels.shape[1]
        return loss
    
    def base_loss_soft(preds, labels, labels1, labels2, labels3):
        preds1, preds2, preds3 = format_preds(preds)
        loss1 = LOSS_FN(preds1, labels1) * (len(LABEL_NAMES)-2)
        loss2 = LOSS_FN_SOFT(preds2, labels2)
        loss3 = LOSS_FN_SOFT(preds3, labels3)
        loss = (loss1+loss2+loss3) / len(LABEL_NAMES)
        return loss
    
    def mixed_loss(preds, labels):
        loss = base_loss(preds, labels) + average_f1_loss(preds, labels)
        return loss / 2
        
    def find_thrs(labels, preds):
        preds1, preds2, preds3 = format_preds(preds)
        preds1 = torch.tensor(preds1).sigmoid().numpy()
        preds = np.zeros(labels.shape)
        preds[:, -2] = np.argmax(preds2, axis=-1)
        preds[:, -1] = np.argmax(preds3, axis=-1)
        thrs = np.zeros([labels.shape[1]-2]) + 0.5
        for i in range(labels.shape[1]-2):
            best_f1 = 0
            for thr in range(10, 90, 5):
                thr = thr / 100
                preds[:, i] = (preds1[:, i]>thr).astype(int)
                f1 = average_f1(preds, labels)
                if best_f1 < f1:
                    thrs[i] = thr
                    best_f1 = f1
                #print(i, thr, f1)
            preds[:, i] = (preds1[:, i]>thrs[i]).astype(int)
        return thrs
        
    def get_score(labels, preds, thrs=None):
        preds1, preds2, preds3 = format_preds(preds)
        preds = np.zeros(labels.shape)
        preds1 = torch.tensor(preds1).sigmoid().numpy()
        if thrs is None:
            preds[:, :-2] = (preds1>0.5).astype(int)
        else:
            for i, thr in enumerate(thrs):
                preds[:, i] = (preds1[:, i]>thr).astype(int)
        preds[:, -2] = np.argmax(preds2, axis=-1) + 1
        preds[:, -1] = np.argmax(preds3, axis=-1) + 1
        f1s = average_f1_cls(preds, labels)
        f1 = average_f1(preds, labels)
        
        return f1, f1s
    
    st = time()
    #warnings.filterwarnings("error")
    
    print(f'start fold {CFG.FOLD}')
    train_data = data[data['fold']!=CFG.FOLD]
    vaild_data = data[data['fold']==CFG.FOLD]
    sub_train_data = pd.concat([sub_train_data, semi_train_data, 
                                train_data]).sample(frac=1.0, random_state=CFG.SEED+2)
    TRAIN_DATASET = train_dataset(sub_train_data, is_train=True)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=CFG.BATCH_SIZE, collate_fn=collate,
                                  shuffle=True, drop_last=False, num_workers=8)
                        
    VALID_DATASET = train_dataset(vaild_data, is_train=False)
    VALID_DATALOADER = DataLoader(VALID_DATASET, batch_size=CFG.BATCH_SIZE, collate_fn=collate,
                                  shuffle=False, drop_last=False, num_workers=8)
    
    
    
    N_SAMPLES = len(sub_train_data) * CFG.PRETRAIN_EPOCHS
    N_STEPS = np.ceil(N_SAMPLES/(CFG.BATCH_SIZE*CFG.GRADIENT_ACCUM))
    N_SAMPLES = len(train_data) * (CFG.NUM_EPOCHS-CFG.PRETRAIN_EPOCHS)
    N_STEPS += np.ceil(N_SAMPLES/(CFG.BATCH_SIZE*CFG.GRADIENT_ACCUM))
    OPTIMIZER = torch.optim.AdamW(my_model.parameters(), lr=CFG.LR_MAX)
    # Cosine Learning Rate With Warmup
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=OPTIMIZER, num_cycles=CFG.NUM_CYCLES,
        num_warmup_steps=round(N_STEPS*CFG.NUM_WARMUP_RATE),
        num_training_steps=N_STEPS
    )
    print(f'BATCH_SIZE: {CFG.BATCH_SIZE}, N_STEPS: {N_STEPS}')
    
    best_score = 0
    loss_func = base_loss_soft
    for epoch in range(CFG.NUM_EPOCHS):    
        if epoch == CFG.PRETRAIN_EPOCHS:
            TRAIN_DATASET = train_dataset(train_data, is_train=True)
            TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=CFG.BATCH_SIZE, collate_fn=collate,
                                          shuffle=True, drop_last=False, num_workers=8)
            loss_func = base_loss
            
        loss_sum = 0
        loss_count = 0
        loss_cache = []
    
        ste = time()    
        my_model.train()
        ga_steps = 0
        print(f'start epoch {epoch}')
        for step, (input_ids, attention_mask, labels, soft_labels1, soft_labels2, soft_labels3) in enumerate(TRAIN_DATALOADER):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            soft_labels1 = soft_labels1.to(DEVICE) 
            soft_labels2 = soft_labels2.to(DEVICE)
            soft_labels3 = soft_labels3.to(DEVICE)
            
            # Forward Pass
            logits = my_model(input_ids=input_ids, attention_mask=attention_mask)
               
            # Backward Pass
            loss = loss_func(logits, labels, soft_labels1, soft_labels2, soft_labels3)
            
            #loss = torch.mean(loss*weights)
            loss.backward()
            
            ga_steps += 1
            if ga_steps>=CFG.GRADIENT_ACCUM or step==len(TRAIN_DATALOADER)-1:
                # optimizer step
                OPTIMIZER.step()
        
                # Zero Out Gradients
                OPTIMIZER.zero_grad()
            
                # Update Learning Rate Scheduler
                lr_scheduler.step()
                
                loss_cache.append(loss)
                len_cache = len(loss_cache)
                mean_loss = sum(loss_cache) / len_cache
                loss_cache = []
                
                #mean_loss = mean_loss.item()
                loss_sum += mean_loss.detach().cpu() * len_cache
                loss_count += len_cache
                ga_steps = 0
                del input_ids, attention_mask, labels, soft_labels1, soft_labels2, soft_labels3
                gc.collect()
                torch.cuda.empty_cache()
            else:
                loss_cache.append(loss)
            
            if (step + 1) % 40 == 0:  
                metrics = 'µ_loss: {:.3f}'.format(loss_sum/loss_count)
                metrics += ', step_loss: {:.3f}'.format(loss)
                lr = OPTIMIZER.param_groups[0]['lr']
                print(f'{epoch+1:02}/{CFG.NUM_EPOCHS:02} | {step+1:04}/{N_STEPS} lr: {lr:.2E}, {metrics}', end='')
                print(f'\nSteps per epoch: {step+1} complete | Time elapsed: {time()- st}')
        
        my_model.eval()
        pred_cache = []
        label_cache = []
        for input_ids, attention_mask, labels, soft_labels1, soft_labels2, soft_labels3 in VALID_DATALOADER:
            # Forward Pass
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            with torch.no_grad():
                logits = my_model(input_ids=input_ids, attention_mask=attention_mask)
            pred_cache.append(logits.detach().cpu().float())
            label_cache.append(labels.detach().cpu().float())
            gc.collect()
            torch.cuda.empty_cache()
        pred_cache = torch.cat(pred_cache, dim=0).numpy()
        label_cache = torch.cat(label_cache, dim=0).numpy()
        score, scores = get_score(label_cache, pred_cache)
        scores = {l:s for l, s in zip(LABEL_NAMES, scores)}
        thrs = find_thrs(label_cache, pred_cache)
        new_score, new_scores = get_score(label_cache, pred_cache, thrs=thrs)
        new_scores = {l:s for l, s in zip(LABEL_NAMES, new_scores)}
        print(f'valid score: {score}, new score: {new_score}, thrs: {thrs}')
        print(f'detailed scores: {new_scores}')
        print()
        
    model_params = dict([(k,v.cpu()) for k, v in my_model.named_parameters() if v.requires_grad])
    torch.save({'model': model_params, 'thrs': thrs, 
                'oof_preds': pred_cache, 'oof_uids': vaild_data['uid'].values},
               f'{model_cfg["save_name"]}_{CFG.FOLD+1}.pth')

from config import MODEL_CFGS

if __name__ == '__main__':
    for model_cfg in MODEL_CFGS:
        FOLDS = [0, 1, 2, 3, 4]
        for fold in FOLDS:
            run_code(model_cfg, fold)
            gc.collect()
            torch.cuda.empty_cache()
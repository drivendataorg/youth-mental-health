# ====================================================
# Library
# ====================================================
from test_config import CFG
import os
import gc
import sys
import copy
import json
import time
import math
import pickle
import random

import numpy as np
import pandas as pd
#from tqdm.auto import tqdm
import gc 

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
device = 'cuda'

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

LABEL_NAMES = [
    'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt', 
    'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem', 'MentalHealthProblem', 
    'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia', 'DiagnosisBipolar', 'DiagnosisAdhd', 
    'IntimatePartnerProblem', 'FamilyRelationship', 'Argument', 'SchoolProblem', 
    'RecentCriminalLegalProblem', 'SuicideNote', 'SuicideIntentDisclosed', 
    'DisclosedToIntimatePartner', 'DisclosedToOtherFamilyMember', 'DisclosedToFriend', 
    'InjuryLocationType', 'WeaponType1'
]
CFG.NUM_LABELS = (len(LABEL_NAMES)-2)+6+12

# ====================================================
# Utils
# ====================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.SEED)

# ====================================================
# Dataset
# ====================================================
class TestDataset():
    def __init__(self, data_df, is_train=False):
        #self.uids = data[:, 0]
        self.Narratives = data_df['Narrative'].values
    
    def __len__(self):
        return len(self.Narratives)

    def __getitem__(self, idx):     
        report = self.Narratives[idx]
        tokens = tokenizer(report+tokenizer.eos_token, max_length=CFG.MAX_LENGTH,
                           padding='longest', truncation=True, add_special_tokens=True)
            
        #print(np.array(label_tokens['input_ids'])[label_ids])
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        return input_ids, attention_mask

# ====================================================
# Model
# ====================================================
class MyModel1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model.model.model
        self.head = model.score
        
    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0][:, -1]
        output = self.head(hidden_states)
        return output

class MyModel2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        
    def forward(self, input_ids, attention_mask):
        output = self.base_model(input_ids, attention_mask=attention_mask)[0]
        return output

MyModels = [MyModel1, MyModel2]
    
# ====================================================
# Helper functions
# ====================================================
def format_prob_preds(preds):
    preds3 = preds[:, -12:]
    preds2 = preds[:, -18:-12]
    preds1 = preds[:, :-18]
    preds1 = torch.tensor(preds1).sigmoid().numpy()
    preds2 = torch.tensor(preds2).softmax(dim=-1).numpy()
    preds3 = torch.tensor(preds3).softmax(dim=-1).numpy()
    return preds1, preds2, preds3

def get_final_preds(preds1, preds2, preds3, thrs=None):
    preds = np.zeros([preds1.shape[0], len(LABEL_NAMES)])
    if thrs is None:
        preds[:, :-2] = (preds1>0.5).astype(int)
    else:
        for i, thr in enumerate(thrs):
            preds[:, i] = (preds1[:, i]>thr).astype(int)
    preds[:, -2] = np.argmax(preds2, axis=-1) + 1
    preds[:, -1] = np.argmax(preds3, axis=-1) + 1
    return preds.astype(int)

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

def test_fn(valid_loader, model, device):
    model.eval()
    preds = []
    for step, (input_ids, attention_mask) in enumerate(valid_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask=attention_mask)
        preds.append(y_preds.to('cpu').to(torch.float32).numpy())
        del input_ids, attention_mask

    predictions = np.concatenate(preds)
    return predictions

"""
def sorted_infer(data_df, model, tokenizer):
    test_dataset = TestDataset(new_data_df, is_train=False)
    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.BATCH_SIZE,
                              shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
    input_ids_list = []
    attention_mask_list = []
    for input_ids, attention_mask in test_loader:
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    input_ids = np.concatenate(input_ids_list, axis=0)
    attention_mask = np.concatenate(attention_mask_list, axis=0)
    ids = np.arange(len(input_ids))
    new_ids = sorted(ids, key=lambda x:len(input_ids[x]))
    input_ids[input_ids]
"""
    
if __name__ == '__main__':
    data_df = pd.read_csv(CFG.DATA_PATH)
    data_df['Narrative'] = data_df['NarrativeLE'] + '\n' + data_df['NarrativeCME']

    preds1_sum = 0
    preds2_sum = 0
    preds3_sum = 0
    weight_sum = 0
    thrs_sum = 0
    for model_config in CFG.MODEL_CONFIGS:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config['base_model_path'],
            num_labels=CFG.NUM_LABELS,
            torch_dtype=torch.float16
        ) 
        target_modules = find_target_modules(base_model)
        CFG.MAX_LENGTH = model_config['max_length']

        for lora_config in model_config['lora_configs']:
            print(lora_config['path'])
            tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_path'])
            if  lora_config['pad_left']:
                tokenizer.padding_side = 'left'
            base_model.config.pad_token_id = tokenizer.pad_token_id
                
            # Create LoRa Model
            _lora_config = LoraConfig(
                r = lora_config['rank'], 
                lora_alpha = lora_config['alpha'],
                lora_dropout= 0, 
                bias='none',
                inference_mode=True,
                task_type=TaskType.SEQ_CLS,
                target_modules=target_modules
            ) # Only Use Output and Values Projection
            model = get_peft_model(base_model, _lora_config)
            model = MyModels[lora_config['model_type']](model).to(torch.bfloat16)
            model_params = torch.load(lora_config['path'], map_location='cpu')
            try:
                thrs_sum += model_params['thrs_list'][0] * lora_config['weight']
            except:
                thrs_sum += model_params['thrs'] * lora_config['weight']
            model.load_state_dict(model_params['model'], strict=False)
            model = model.to(torch.float16).to(device)
            
            test_dataset = TestDataset(data_df, is_train=False)
            test_loader = DataLoader(test_dataset,
                                      batch_size=model_config['batch_size'],
                                      shuffle=False,
                                      num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
            raw_preds = test_fn(test_loader, model, device)
            preds1, preds2, preds3 = format_prob_preds(raw_preds)
            preds1_sum += preds1 * lora_config['weight']
            preds2_sum += preds2 * lora_config['weight']
            preds3_sum += preds3 * lora_config['weight']
            weight_sum += lora_config['weight']
            
            del model, model_params
            gc.collect()
            torch.cuda.empty_cache()
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        
    preds1 = preds1_sum / weight_sum
    preds2 = preds2_sum / weight_sum
    preds3 = preds3_sum / weight_sum
    thrs = thrs_sum / weight_sum
    final_preds = get_final_preds(preds1, preds2, preds3, thrs=thrs)
    submission = data_df[['uid']]
    submission[LABEL_NAMES] = final_preds
    submission.to_csv(CFG.SUBMISSION_PATH, index=False)
import torch
import random
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, cfg, df, binary_dict, il_dict, wt_dict, shuffle):
        self.cfg = cfg
        self.df = df
        self.nar_le = df["NarrativeLE"].values
        self.nar_cme = df["NarrativeCME"].values
        self.binary_dict =  binary_dict
        self.il_dict = il_dict
        self.wt_dict = wt_dict
        self.shuffle = shuffle

        self.tokenizer = cfg.tokenizer
        self.no_id = self.tokenizer("no")["input_ids"][1]
        self.yes_id = self.tokenizer("yes")["input_ids"][1] 
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.nar_le)

    def prepare_input(self, nar_le, nar_cme, labels):
        ### Create Prefix: Variable + <mask> ###
        var_list = []
        for v in self.binary_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")

        for v in self.il_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")

        for v in self.wt_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")   


        num_location = len(self.il_dict.keys())
        num_weapon = len(self.wt_dict.keys())
        binary_labels = [self.no_id] * (len(labels) + num_location + num_weapon - 2)
        il_label = labels[-2]
        wt_label = labels[-1]
        labels = labels[:-2]
        for idx, label in enumerate(labels):
            if label == 1:
                binary_labels[idx] = self.yes_id
        binary_labels[len(labels) + il_label - 1] = self.yes_id
        binary_labels[len(labels) + num_location + wt_label - 1] = self.yes_id


        text = nar_le + " " + nar_cme
        

        if self.shuffle:
            tmp = list(zip(var_list, binary_labels))
            random.shuffle(tmp)
            var_list, binary_labels = zip(*tmp)
        
        descriptions = "".join(var_list)
        text = descriptions + text

        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
        )

        ### Global attention of Longformer model ###
        global_attention_mask = inputs["attention_mask"]
        last_mask_ids = 0
        for i  in range(len(inputs["input_ids"])):
            if inputs["input_ids"][i] == self.mask_token_id:
                last_mask_ids = i
        prefix_token_len = last_mask_ids + 1
        for i in range(prefix_token_len):
            global_attention_mask[i] = 2
        inputs["attention_mask"] = global_attention_mask

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        labels = torch.tensor(binary_labels, dtype=torch.long)

        return inputs, labels, prefix_token_len

    def __getitem__(self, item):
        label_cols = list(self.binary_dict.values()) + ["Injury Location Type", "Weapon Type"]
        labels = [self.df.loc[item, k] for k in label_cols]
        inputs, labels, prefix_token_len = self.prepare_input(
                                self.nar_le[item],
                                self.nar_cme[item],
                                labels)
        return inputs, labels, prefix_token_len

def collate(inputs, prefix_token_len):
    # For longformer only
    mask_len = int(inputs["attention_mask"].sum(axis=1).max()) - prefix_token_len.min()
    
    # For deberta and others
    # mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df, binary_dict, il_dict, wt_dict):
        self.cfg = cfg
        self.df = df
        self.nar_le = df["NarrativeLE"].values
        self.nar_cme = df["NarrativeCME"].values
        self.binary_dict =  binary_dict
        self.il_dict = il_dict
        self.wt_dict = wt_dict

        self.tokenizer = cfg.tokenizer
        self.no_id = self.tokenizer("no")["input_ids"][1]
        self.yes_id = self.tokenizer("yes")["input_ids"][1] 
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.nar_le)

    def prepare_input(self, nar_le, nar_cme):
        text = nar_le + " " + nar_cme

        ### Create Prefix: Variable + <mask> ###
        var_list = []
        for v in self.binary_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")
        for v in self.il_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")
        for v in self.wt_dict.values():
            var_list.append(f"{v} {self.tokenizer.mask_token} ")   
        
        descriptions = "".join(var_list)
        text = descriptions + text
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=1664,
            truncation=True,
        )

        ### Global attention of Longformer model ###
        global_attention_mask = inputs["attention_mask"]
        last_mask_ids = 0
        for i  in range(len(inputs["input_ids"])):
            if inputs["input_ids"][i] == self.mask_token_id:
                last_mask_ids = i
        prefix_token_len = last_mask_ids + 1
        for i in range(prefix_token_len):
            global_attention_mask[i] = 2
        inputs["attention_mask"] = global_attention_mask

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        return inputs

    def __getitem__(self, item):
        inputs = self.prepare_input(
                                self.nar_le[item],
                                self.nar_cme[item])
        return inputs
    
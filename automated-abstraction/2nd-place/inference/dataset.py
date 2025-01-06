import torch
from torch.utils.data import Dataset
from cfg import DEFINE
 
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
            truncation=True,
            max_length=1664,
        )

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
    

class FlanTestDataset(Dataset):
    def __init__(self, cfg, df, variables):
        self.df = df
        self.cfg = cfg
        self.nar_le = df["NarrativeLE"].values
        self.nar_cme = df["NarrativeCME"].values
        self.prompts = [DEFINE.column_dict[v] for v in variables[:-2]] + \
                        [DEFINE.il_dict[v] for v in DEFINE.il_dict.keys()] + \
                        [DEFINE.wt_dict[v] for v in DEFINE.wt_dict.keys()]
        self.tokenizer = cfg.tokenizer

    def __len__(self):
        return len(self.nar_le)

    def prepare_input(self, nar_le, nar_cme):
        text = ""
        for idx, p in enumerate(self.prompts):
            text += p + f" <extra_id_{idx}> "
        text += nar_le + " " + nar_cme

        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            padding="max_length",
            max_length=1532,
            truncation=True,
        )

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        return inputs

    def __getitem__(self, item):
        inputs = self.prepare_input(
                                self.nar_le[item],
                                self.nar_cme[item])
        return inputs
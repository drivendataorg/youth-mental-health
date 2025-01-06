SYSTEM_TEMPLATE = """You are a data generator which generates detailed stories for training language models according to user's need."""
USER_TEMPLATE1 = """The story below is an example report-style story which contains 2 paragraphs as 2 segments: a law enforcement report and a coroner/medical examiner report of a youth suicide victim.
------------
"""

USER_TEMPLATE2 = """
------------

Now write a new story in the same style and format but with new settings as below:
------------
"""
USER_TEMPLATE3 = """
------------

You should only output the new story with the 2 paragraphs of a youth suicide victim based on the given settings. Do not output titles or anything other than the story. Make sure you write 2 paragraphs which are not fully the same. Single paragraph does not necessarily include all the new settings, but make sure all the settings are covered in these 2 paragraphs."""

if __name__ == '__main__':
    import os
    import gc
    import re
    from time import time
    import random
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    import torch
    import transformers
    import argparse
    from sklearn.metrics import accuracy_score
    from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaConfig

    from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
    import torch.nn.functional as F

    from torch import nn
    #from spmd_util import partition_module

    tqdm.pandas()

    print(f'Torch Version: {torch.__version__}')

    class CFG:
        NUM_EPOCHS = 1
        BATCH_SIZE = 2
        GRADIENT_ACCUM = 32
        DROPOUT = 0.0
        MODEL_NAME = 'assets/gemma2_9b_it' # NOTE - check what winner is using
        # /home/ubuntu/.cache/huggingface/hub/models--microsoft--deberta-v3-large
        SEED = 1222
        #P_MAX_LENGTH = 192
        #R_MAX_LENGTH = 384
        NUM_WARMUP_RATE = 0.1
        LR_MAX = 2e-4
        FREEZE_RATE = 0
        NUM_LABELS = 3
        LORA_RANK = 64
        LORA_ALPHA = 128
        #LORA_MODULES = ['o_proj', 'v_proj']
        MAX_LENGTH = 3072
        IGNORE_INDEX = -100

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    #class args:
    #    id = 0
    DEVICE = f'cuda:{args.id%2}'
    CFG.SEED = CFG.SEED + args.id

    def set_seeds(seed):
        """Set seeds for reproducibility """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set seed for all TPU cores
        #xm.set_rng_state(seed, device=xm.xla_device())  

    set_seeds(seed=CFG.SEED)

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    # NOTE - change
    data_df = pd.read_csv('data/train_features.csv').iloc[:100]
    label_df = pd.read_csv('data/train_labels.csv').iloc[:100]
    LABEL_NAMES = list(label_df.columns)
    LABEL_NAMES.remove('uid')
    data_df = data_df.merge(label_df, how='inner', on='uid')
    assert len(data_df) == len(label_df)
    
    columns = list(label_df.columns)[1:]
    values = data_df[LABEL_NAMES].values
    dist_dict = {}
    for i in range(len(columns)):
        col = columns[i]
        col_values = values[:, i]
        max_v = max(col_values)
        if max_v == 1:
            onehots = np.eye(2)
            dist = onehots[col_values].mean(0)
        else:
            onehots = np.eye(max_v)
            dist = onehots[col_values-1].mean(0)
        dist_dict[col] = dist
    #print(dist_dict)

    LE_text = data_df['NarrativeLE'].values
    LE_words = [len(s.split()) for s in LE_text]
    CME_text = data_df['NarrativeCME'].values
    CME_words = [len(s.split()) for s in CME_text]

    dist_settings = {
        'FamilyRelationship': 0.4, 'RecentCriminalLegalProblem': 0.4, 
        'DisclosedToIntimatePartner': 0.4, 
        'DisclosedToOtherFamilyMember': 0.4, 'DisclosedToFriend': 0.4
    }
    
    new_dist_dict = {}
    for col in dist_dict:
        dist = dist_dict[col]
        if len(dist) == 2:
            if col in dist_settings:
                max_dist = dist_settings[col]
            else:
                max_dist = 0.2
            if dist[1] < max_dist:
                gap = max_dist - dist[1]
                dist[0] = dist[0] - gap
                dist[1] = dist[1] + gap
        else:
            dist = np.ones([len(dist)])/len(dist)
            #dist[-2:] /= 5
            #dist = dist / dist.sum()

        new_dist_dict[col] = dist
    print(new_dist_dict)
    #data_df = data_df.iloc[:4]
    
    from torch.utils.data import Dataset, DataLoader

    LABEL_SETTINGS_MAPPING = {
        'DepressedMood': 'The person was perceived to be depressed at the time. You should design a plot to describe that.',
        'MentalIllnessTreatmentCurrnt': 'The person was currently in treatment for a mental health or substance abuse problem. You should design a plot to describe that.',
        'HistoryMentalIllnessTreatmnt': 'The person had history of ever being treated for a mental health or substance abuse problem. You should design a plot to describe that.',
        'SuicideAttemptHistory': 'The person had history of attempting suicide previously. You should design a plot to describe that.',
        'SuicideThoughtHistory': 'The person had history of suicidal thoughts or plans. You should design a plot to describe that.',
        'SubstanceAbuseProblem': 'The person struggled with a substance abuse problem. You should design a plot to describe that.',
        'MentalHealthProblem': 'The person had a mental health problem at the time. You should design a plot to describe that.',
        'DiagnosisAnxiety': 'The person was diagnosed with Anxiety.',
        'DiagnosisDepressionDysthymia': 'The person was diagnosed with Depression Dysthymia.',
        'DiagnosisBipolar': 'The person was diagnosed with Bipolar.',
        'DiagnosisAdhd': 'The person was diagnosed with Adhd.',
        
        'IntimatePartnerProblem': 'The person had intimate partner problems which appear to have contributed to suicide.',
        'FamilyRelationship': 'The person had relationship problems with a family member (other than an intimate partner), which appear to have contributed to suicide.',
        'Argument': 'The person had an argument or conflict appears to have contributed to suicide.',
        'SchoolProblem': 'The person had problem(s) at or related to school appear to have contributed to suicide.',
        'RecentCriminalLegalProblem': 'The person had Criminal legal problem(s) appear to have contributed to suicide.',
        
        'SuicideNote': 'The person left a suicide note.',
        'SuicideIntentDisclosed': 'The person disclosed their thoughts and/or plans to die by suicide to someone else within the last month.',
        'DisclosedToIntimatePartner': 'Suicide intent was disclosed to a previous or current intimate partner.',
        'DisclosedToOtherFamilyMember': 'Suicide intent was disclosed to a family member (other than an intimate partner).',
        'DisclosedToFriend': 'Suicide intent was disclosed to a friend.',

        'InjuryLocationType': {
            'House, apartment': "Injury location is a house or apartment of somebody, which can be anyone.",
            'Motor vehicle (excluding school bus and public transportation)': "Injury location is a non-public motor vehicle, which can be car, pickup, truck, bus, motorcycle or other",
            'Natural area (e.g., field, river, beaches, woods)': "Injury location is a natural area, which can be field, river, beach, woods, forest, lake, mountain, cave, desert, valley or other",
            'Park, playground, public use area': "Injury location is a public use area, which can be park, playground, zoo, garden, pool, plaza, square, gym, promenade or other",
            'Street/road, sidewalk, alley': "Injury location is a street, road, sidewalk or alley.", 
            'Other': "Injury location is a place which is not any of apartment, motor vehicle, natural area, public use area, street/road, sidewalk and alley."
        },
        'WeaponType1': {
            'Blunt instrument': 'The person committed suicide with a blunt instrument, you should make up a specific type.',
            'Drowning': 'The cause of death is drowning.',
            'Fall': 'The cause of death is fall.',
            'Fire or burns': 'The cause of death is fire or burns.',
            'Firearm': 'The person committed suicide with a firearm, you should make up a specific type of real firearm.',
            'Hanging, strangulation, suffocation': 'The cause of death is hanging, strangulation or suffocation.',
            'Motor vehicle including buses, motorcycles': 'The cause of death is a motor vehicle, which can be car, pickup, truck, bus, motorcycle or other.',
            'Other transport vehicle, eg, trains, planes, boats': 'The cause of death is not a motor vehicle, you should make up a specific type of transport vehicle other than that.',
            'Poisoning': 'The cause of death is poisoning.',
            'Sharp instrument': 'The person committed suicide with a sharp instrument, which can be knife, dagger, razor, glass shard or scissor.',
            'Other (e.g. taser, electrocution, nail gun)': 'You should creatively make up the cause of death, which is not any of blunt instrument, drowning, fall, fire, burn, firearm, hanging, strangulation, suffocation, vehicle, poisoning and sharp instrument.',
            'Unknown': ''
        }     
    }
    
    LABEL_MAPPING = {}
    for lname in LABEL_NAMES:
        if lname == 'InjuryLocationType':
            LABEL_MAPPING[lname] = {
                1: LABEL_SETTINGS_MAPPING[lname]['House, apartment'],
                2: LABEL_SETTINGS_MAPPING[lname]['Motor vehicle (excluding school bus and public transportation)'],
                3: LABEL_SETTINGS_MAPPING[lname]['Natural area (e.g., field, river, beaches, woods)'],
                4: LABEL_SETTINGS_MAPPING[lname]['Park, playground, public use area'],
                5: LABEL_SETTINGS_MAPPING[lname]['Street/road, sidewalk, alley'],
                6: LABEL_SETTINGS_MAPPING[lname]['Other']
            }
        elif lname == 'WeaponType1':
            LABEL_MAPPING[lname] = {
                1: LABEL_SETTINGS_MAPPING[lname]['Blunt instrument'],
                2: LABEL_SETTINGS_MAPPING[lname]['Drowning'],
                3: LABEL_SETTINGS_MAPPING[lname]['Fall'],
                4: LABEL_SETTINGS_MAPPING[lname]['Fire or burns'],
                5: LABEL_SETTINGS_MAPPING[lname]['Firearm'],
                6: LABEL_SETTINGS_MAPPING[lname]['Hanging, strangulation, suffocation'],
                7: LABEL_SETTINGS_MAPPING[lname]['Motor vehicle including buses, motorcycles'],
                8: LABEL_SETTINGS_MAPPING[lname]['Other transport vehicle, eg, trains, planes, boats'],
                9: LABEL_SETTINGS_MAPPING[lname]['Poisoning'],
                10: LABEL_SETTINGS_MAPPING[lname]['Sharp instrument'],
                11: LABEL_SETTINGS_MAPPING[lname]['Other (e.g. taser, electrocution, nail gun)'],
                12: LABEL_SETTINGS_MAPPING[lname]['Unknown']
            }
        else:
            LABEL_MAPPING[lname] = {0: 'No information', 1:LABEL_SETTINGS_MAPPING[lname]}
    
    #EOS_TOKEN_ID = tokenizer(END_TOKEN, add_special_tokens=False)['input_ids']

    #assert len(EOS_TOKEN_ID) == 1
    #EOS_TOKEN_ID = EOS_TOKEN_ID[0]
    #tokenizer.eos_token_id = EOS_TOKEN_ID

    #TEMPLATE_TOKENS1 = tokenizer(PROMPT_TEMPLATE1, truncation=True, max_length=99999999, add_special_tokens=True)
    #TEMPLATE_TOKENS2 = tokenizer(PROMPT_TEMPLATE2, truncation=True, max_length=99999999, add_special_tokens=False)

    def get_max_label_len(extra_text=f'\nThe law enforcement report has about 9547 words, and the coroner/medical examiner report has about 5279 words.'):
        output = 0
        label = [1 for _ in LABEL_NAMES]
        for k in LABEL_MAPPING['InjuryLocationType']:
            label[-2] = k
            for k in LABEL_MAPPING['WeaponType1']:
                label[-1] = k
                MAX_LABEL_TEMPLATE = {}
                for i in range(len(LABEL_NAMES)):
                    MAX_LABEL_TEMPLATE[LABEL_NAMES[i]] = LABEL_MAPPING[LABEL_NAMES[i]][label[i]]
                temp = str(MAX_LABEL_TEMPLATE).replace('WeaponType1', 'WeaponType') + extra_text
                tokens = tokenizer(temp, truncation=True,  max_length=99999999, add_special_tokens=False)
                output = max([output, len(tokens['input_ids'])])
        return output

    #MAX_LABEL_LEN = get_max_label_len()
    #TEMPLATE_TOKEN_LEN = len(TEMPLATE_TOKENS1['input_ids']) + len(TEMPLATE_TOKENS2['input_ids']) 
    #TEMPLATE_TOKEN_LEN += MAX_LABEL_LEN
    #print('TEMPLATE_TOKEN_LEN: ', TEMPLATE_TOKEN_LEN)

    class train_dataset(Dataset):
        def __init__(self, data_df, is_train=True):
            data = data_df.values
            self.uids = data[:, 0]
            self.NarrativeLEs = data[:, 1]
            self.NarrativeCMEs = data[:, 2]
            if data.shape[1] > 3:
                self.labels = data[:, 3:]
            else:
                self.labels = None    

        def __len__(self):
            return len(self.uids)
        
        def get_text_from_vals(self, vals, shuffle=True):
            vals_str = ''
            ids = np.arange(len(LABEL_NAMES))
            if shuffle:
                np.random.shuffle(ids)
            for i in ids:
                if len(LABEL_MAPPING[LABEL_NAMES[i]])>2 or vals[i]==1:
                    vals_str = vals_str + f'{LABEL_MAPPING[LABEL_NAMES[i]][vals[i]]}\n'
            vals_str = vals_str.replace('WeaponType1', 'WeaponType')
            return vals_str
        
        def __getitem__(self, idx):
            messages = [
                # {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": SYSTEM_TEMPLATE + " " + USER_TEMPLATE1}
            ]
            le = self.NarrativeLEs[idx]
            cme = self.NarrativeCMEs[idx]
            label = self.labels[idx]
            messages[0]['content'] = messages[0]['content'] + le + '\n' + cme + USER_TEMPLATE2
            # messages[1]['content'] = messages[1]['content'] + le + '\n' + cme + USER_TEMPLATE2
            
            n_labels = 0
            vals = []
            for ki, dist in enumerate(list(new_dist_dict.values())):
                # import pdb; pdb.set_trace()
                vi = np.random.choice(len(dist), p=dist)
                if ki+2<len(new_dist_dict) and vi==1:
                    n_labels += 1
                if len(dist) > 2:
                    vi += 1
                vals.append(vi)
            messages[0]['content'] = messages[0]['content'] + self.get_text_from_vals(vals) + USER_TEMPLATE3
            # messages[1]['content'] = messages[1]['content'] + self.get_text_from_vals(vals) + USER_TEMPLATE3
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) 
            tokens = tokenizer(text, add_special_tokens=True)
            #print(len(tokens['input_ids']))
            input_ids = torch.tensor(tokens['input_ids'])
            attention_mask = torch.tensor(tokens['attention_mask'])

            return input_ids, attention_mask, np.array(vals)

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=False,
        bnb_8bit_quant_type="nf8")

    base_model = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE)

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
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules) # Only Use Output and Values Projection

    """
    # Create LoRa Model
    model = get_peft_model(base_model, lora_config)
    # Trainable Parameters
    model.print_trainable_parameters()

    my_model = model
    my_model.load_state_dict(torch.load(CFG.MODEL_WEIGHTS), strict=False)
    """
    my_model = base_model
    
    Narratives = []
    vals_list = []
    TRAIN_DATASET = train_dataset(data_df, is_train=True)
    # NOTE - change
    if args.id == 0:
        its = tqdm(enumerate(TRAIN_DATASET))
    else:
        its = enumerate(TRAIN_DATASET)
    # import pdb; pdb.set_trace()
    for step, (input_ids, attention_mask, vals) in its:
        input_ids = input_ids.reshape([1, -1]).to(DEVICE)
        attention_mask = attention_mask.reshape([1, -1]).to(DEVICE)
        generate_ids = my_model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=3000, #do_sample=True, temperature=0.7, top_p=0.95,
            #top_k=40, repetition_penalty=1.5, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id                            
        )
        generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
        #print(generated_text)
        report = generated_text.split('assistant\n')[-1]
        report = report.replace('**Law Enforcement Report:**\n', '')
        report = report.replace('**Coroner/Medical Examiner Report:**\n', '')
        report = report.replace('\n\n', '\n').replace('\n\n', '\n')
        Narratives.append(report)
        vals_list.append(vals)
        
    
    output_df = pd.DataFrame({'Narrative': Narratives})
    output_df[LABEL_NAMES] = np.array(vals_list)
    output_df.to_csv(f'gendata_{args.id}.csv', index=False)
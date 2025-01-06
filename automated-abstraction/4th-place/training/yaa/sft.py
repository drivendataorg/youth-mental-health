import os, sys, logging
import json
import pandas as pd
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
from trainer import Trainer, TrainingArguments
#import trl
#from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification, StableDropout
from transformers.models.qwen2 import Qwen2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import ROPE_INIT_FUNCTIONS
from transformers import DebertaV2ForSequenceClassification
from transformers import LlamaForSequenceClassification, Phi3ForSequenceClassification
from transformers.models.t5 import T5ForSequenceClassification

from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from types import MethodType
from torch import nn
import torch.nn.functional as F

from nn import set_seed



import util
import pt_util as pu
from dataset import gen_ds


logger = logging.getLogger(__name__)





def pool(self, hidden_states, attention_mask):
    ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
    ends = ends.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
    hidden_states = torch.gather(hidden_states, 1, ends.to(hidden_states.device)).squeeze(1)
    return hidden_states


def forward_classify(self, input_ids=None, attention_mask=None, return_dict=None, **kwargs):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        return_dict=return_dict,
        **kwargs
    )
    pooled_output = self.dropout(self.pool(transformer_outputs[0], attention_mask))
    logits = self.score(pooled_output)
    loss = None

    if not return_dict:
        output = (logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def load_unsloth_model(args, model_id):
    kwargs = dict()
    if args.att_dp>0:
        kwargs['attention_dropout'] = args.att_dp
    from unsloth import FastLanguageModel
    logger.info('modelid %s', model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(model_id, dtype=getattr(torch, args.torch_dtype), use_cache=False, load_in_4bit=args.use_4bit,
                                                         use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False, **kwargs)
    if args.use_lora:
        logger.info('linear modules:%s', find_all_linear_names(args, model))
        model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, bias="none",
                                             random_state=args.seed,
                                             use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False,
                                             target_modules=args.lora_modules,
                                             use_dora=args.use_dora)
    return model, tokenizer

def find_all_linear_names(args, model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if args.use_4bit else (bnb.nn.Linear8bitLt if args.use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    logger.info('find linear:%s', lora_module_names)


    return list(lora_module_names)


def load_model(args, model_id):
    if args.use_badam or args.use_full:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs = dict()
        if args.debug:
            kwargs['num_hidden_layers'] = 2
            kwargs['hidden_size'] = 16
            kwargs['intermediate_size'] = 8
            kwargs['num_attention_heads'] = 2
            kwargs['num_key_value_heads'] = 2
            config = AutoConfig.from_pretrained(model_id, **kwargs)
            if args.is_classify:
                if args.train_cols is None:
                    num_labels = len(util.binary_labels) + 18
                else:
                    num_labels = len(args.train_cols)
                model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True,
                                                                           torch_dtype=getattr(torch, args.torch_dtype), num_labels=num_labels)
            else:
                model = AutoModelForCausalLM.from_config(config)
        else:
            if args.att_dp>0:
                if 'deb' in model_id:
                    kwargs['attention_probs_dropout_prob'] = args.att_dp
                else:
                    kwargs['attention_dropout'] = args.att_dp
            if args.is_classify:
                if args.avg_pool:
                    from nn import CustDebertaV2ForSequenceClassification
                    cls = CustDebertaV2ForSequenceClassification
                else:
                    cls = AutoModelForSequenceClassification
                logger.info('cls is %s', cls)
                if args.train_cols is None:
                    num_labels = len(util.binary_labels) + 18
                else:
                    num_labels = len(args.train_cols)
                model = cls.from_pretrained(model_id, trust_remote_code=True,
                                                                           torch_dtype=getattr(torch, args.torch_dtype), num_labels=num_labels, **kwargs)
                if isinstance(model, DebertaV2ForSequenceClassification):
                    if args.cls_dp>0:
                        model.dropout = StableDropout(args.cls_dp)
                    if args.pool_dp>0:
                        model.pooler.dropout = StableDropout(args.pool_dp)
                    if args.frozen_emb:
                        logger.info('frozen emb')
                        pu.requires_grad(model.deberta.embeddings, False)
                    if args.n_frozen>0:
                        logger.info('frozen %s layers', args.n_frozen)
                        encoder = model.deberta.encoder
                        for i, l in enumerate(encoder.layer):
                            if i<args.n_frozen:
                                pu.requires_grad(l, False)

                else:
                    if args.cls_dp>0:
                        model.dropout = StableDropout(args.cls_dp)
                        model.pool = MethodType(pool, model)
                        model.forward = MethodType(forward_classify, model)

            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), use_cache=False, trust_remote_code=True, **kwargs)
            model = model.cuda()

    elif args.use_unsloth:
        model, tokenizer = load_unsloth_model(args, model_id)

    elif args.use_lora:
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        if args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit,
                load_in_8bit=args.use_8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=getattr(torch, args.torch_dtype),
                bnb_4bit_use_double_quant=args.use_double_quant,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        kwargs = dict()
        if args.att_dp > 0:
            if 'deberta' in model_id:
                kwargs['attention_probs_dropout_prob'] = args.att_dp
            else:
                kwargs['attention_dropout'] = args.att_dp
        if args.is_classify:
            if args.train_cols is None:
                num_labels = len(util.binary_labels) + 18
            else:
                num_labels = len(args.train_cols)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=quantization_config, device_map={"": 0}, trust_remote_code=True,
                                                         torch_dtype=getattr(torch, args.torch_dtype), num_labels=num_labels, **kwargs)
            task_type = 'SEQ_CLS'
            if isinstance(model, DebertaV2ForSequenceClassification):
                if args.cls_dp > 0:
                    model.dropout = StableDropout(args.cls_dp)
                if args.pool_dp > 0:
                    model.pooler.dropout = StableDropout(args.pool_dp)
            else:
                if args.cls_dp > 0:
                    model.dropout = StableDropout(args.cls_dp)
                    model.pool = MethodType(pool, model)
                    model.forward = MethodType(forward_classify, model)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, trust_remote_code=True,
                                          torch_dtype=getattr(torch, args.torch_dtype), **kwargs)
            task_type = 'CAUSAL_LM'

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(args, model) if args.lora_modules is None else args.lora_modules,
            layers_to_transform=[i for i in range(model.config.num_hidden_layers) if i >= args.lora_start_layer],
            lora_dropout=args.lora_dropout,
            use_dora=args.use_dora,
            bias="none",
            task_type=task_type,
        )

        model = get_peft_model(model, lora_config)
        if args.use_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        raise Exception('you muse specify either of use_full,use_lora,use_unsloth')

    if model.config.pad_token_id is None:
        pad_token_id = model.config.eos_token_id
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[-1]

        model.config.pad_token_id = pad_token_id
        tokenizer.pad_token_id = pad_token_id
    logger.info('pad token id:%s, %s, %s', tokenizer.decode(model.config.pad_token_id), model.config.pad_token_id, tokenizer.pad_token_id)
    return model, tokenizer


def setup_training(args, model, tokenizer, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        remove_unused_columns=args.remove_unused_columns,
        output_dir=output_dir,
        seed=args.seed+args.kfid,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        torch_compile=args.torch_compile,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        use_badam=args.use_badam,
        use_sampler=args.use_sampler,
        switch_block_every=args.switch_block_every,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_dir=output_dir,
        logging_steps=args.verbose,
        report_to=args.report_to,
        evaluation_strategy=args.evaluation_strategy if args.do_eval else 'no',
        eval_steps=args.eval_steps,
        eval_delay=args.eval_delay,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_steps=args.eval_steps,
        save_only_model=not args.save_opt,
        load_best_model_at_end=True if args.do_eval else False,
        bf16=args.use_bf16,
        fp16=args.use_fp16,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        metric_for_best_model=args.metric_for_best_model,
        disable_tqdm=args.disable_tqdm,

        ##
        bi_ls=args.bi_ls,
        cat_ls=args.cat_ls,
        bi_rdrop=args.bi_rdrop,
        rdrop=args.rdrop,
        use_kl=args.use_kl,
        cat_rdrop1=args.cat_rdrop1,
        cat_rdrop2=args.cat_rdrop2,
        w_bi=args.w_bi,
        w_lt=args.w_lt,
        w_wt=args.w_wt,
        semi_ratio=args.semi_ratio,
        hard_ratio=args.hard_ratio,
        is_classify=args.is_classify,
        dp_start=args.dp_start,
        train_cols=args.train_cols,
    )

    # TRAIN
    if args.is_split:
        from trainer import TrainerSplit
        cls = TrainerSplit
    else:
        cls = Trainer
    logger.info('cls for trainer:%s', cls)
    trainer = cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 and args.do_eval else None,

    )

    return trainer

def prepare_dataset(args, **kwargs):
    train_ds, val_ds, test_ds = None, None, None
    if args.do_train or args.do_eval:
        data = util.load_data(args)
        kf = KFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        #kf = StratifiedKFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        splits = kf.split(data, data.src)
        for i in range(args.kn):
            train_inds, val_inds = next(splits)
            if i==args.kfid:
                break
        train_data = data.iloc[train_inds]
        val_data = data.iloc[val_inds]
        if args.no_validate:
            train_data = pd.concat([train_data, val_data])
            val_data = val_data[:10]

        train_ds = gen_ds(args, 'train', train_data, **kwargs)
        val_ds = gen_ds(args, 'val', val_data, **kwargs)
        logger.info('train ds:%s, val_ds:%s', len(train_ds), len(val_ds))
    if args.do_test:
        test_args = deepcopy(args)
        test_args.data_type = 'test'
        test_data = util.load_data(test_args)
        test_ds = gen_ds(args, 'test', test_data, **kwargs)
    return train_ds, val_ds, test_ds


def main(args):
    logger.info('backbone: %s, kfid: %s', args.backbone, args.kfid)
    seed = args.seed + args.kfid
    set_seed(seed)

    model, tokenizer = load_model(args, args.backbone)
    tokenizer.padding_side = 'right'
    output_dir = f"{args.output_dir}/{args.model_name}_KF{args.kfid}"
    os.makedirs(output_dir, exist_ok=True)
    util.dump_json(args.__dict__, f'{output_dir}/args.json')

    logger.info('num of params %s', util.get_num_of_paras(model))

    train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, model_config=model.config)
    trainer = setup_training(args, model, tokenizer, train_ds, val_ds, output_dir)
    if args.do_train:
        trainer.train()
        logger.info('train DONE!')
    if args.do_test:
        outputs = trainer.predict(test_ds)
        util.dump(outputs, f'{args.output_dir}/{args.model_name}/pred_test.dump')
        print(outputs.keys())
        logger.info('test DONE!')
    logger.info('DONE!')

if __name__ == "__main__":
    args = util.parser.parse_args()
    util.set_logger()
    if args.debug:
        args.backbone = 'HuggingFaceTB/SmolLM-135M'
        args.num_train_epochs = 2
        args.num = 1000000
        args.eval_steps = 10
        args.batch_size = 1
        args.val_batch_size = 1
        args.dataloader_num_workers = 0
        args.gradient_accumulation_steps = 1
        args.do_train = True
        args.seed = 9528
        args.kn = 2
        args.use_full = True
        args.disable_tqdm = True
        args.ds_cls = 'Dataset'
        args.val_ds_cls = 'Dataset'
        args.ds_cls = 'VGDataset'
        args.val_ds_cls = 'VGDataset'
        #args.max_seq_len = 8
        args.max_seq_len = 4096
        args.max_gen_len = 1024
        args.n_ctx = 4
    for kfid in args.kfids.split():
        my_args = deepcopy(args)
        my_args.kfid = int(kfid)
        main(my_args)

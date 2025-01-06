import os
import util
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from transformers import Trainer as HFTrainer, TrainingArguments as HFTrainingArguments
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from dataclasses import asdict, dataclass, field, fields
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.nn import functional as F
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


from transformers.utils import is_sagemaker_mp_enabled


def kl_div_loss(logits, labels, smooth=0):
    if smooth>0:
        labels = (1.0 - smooth) * labels + smooth/labels.shape[-1]
    loss = F.kl_div(F.log_softmax(logits, dim=-1), labels.to(logits.dtype), reduction='none')
    loss = torch.mean(torch.sum(loss, axis=-1))
    return loss

def rdrop_loss(p, q, mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    # You can choose whether to use function "sum" and "mean" depending on your task
    loss = (p_loss + q_loss) / 2
    if mask is not None:
        loss = torch.mean(loss[mask.to(bool)])
    else:
        loss = torch.mean(loss)
    return loss

def rdrop_loss2(p, q, mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    # You can choose whether to use function "sum" and "mean" depending on your task
    loss = (p_loss + q_loss) / 2
    loss = torch.mean(torch.sum(loss, axis=-1))
    return loss

class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(ds))
        assert len(self.inds) == len(self.ds.data)
        title_num = defaultdict(int)
        for rec in self.ds.data:
            title_num[rec.title] += 1
        title_weight = {k:1/v for k, v in title_num.items()}
        self.weights = [title_weight[rec.title] for rec in self.ds.data]
        self.weights = np.array(self.weights)/np.sum(self.weights)

        assert abs(1-sum(self.weights))<1e-10

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weights)
        else:
            raise NotImplementedError(self.data_type)
        return inds




class TrainerMix():

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
            if self.args.use_badam:
                from badam import BlockOptimizer
                self.optimizer = BlockOptimizer(
                    base_optimizer=self.optimizer,  # can be any torch.Optimizer
                    named_parameters_list=list(opt_model.named_parameters()),
                    switch_block_every=self.args.switch_block_every,
                    # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter.
                    switch_mode="random",
                    # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
                    verbose=1,  # information level, will print trainable parameters when setting to 2
                    block_prefix_list=None,
                )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def log(self, logs):
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if 'elapsed' not in logs:
            logs['elapsed'] = time.time()-self.custom_start_time
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class Trainer(TrainerMix, HFTrainer):
    def __init__( self, model=None, args=None, **kwargs ):
        self.custom_start_time = time.time()
        self.curr_train_step = 0
        self.backup_dp = dict()
        if args.dp_start>0:
            self.update_dropout(model)

        super().__init__(model=model, args=args, **kwargs)

    def _get_train_sampler(self):
        if self.args.use_sampler>0:
            return Sampler(self.args, 'train', self.train_dataset)
        else:
            return RandomSampler(self.train_dataset)

    def update_dropout(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                if name in self.backup_dp:
                    module.p = self.backup_dp[name]
                else:
                    self.backup_dp[name] = module.p
                    module.p = 0
                #print(name, module.p)
            elif isinstance(module, StableDropout):
                if name in self.backup_dp:
                    module.drop_prob = self.backup_dp[name]
                else:
                    self.backup_dp[name] = module.drop_prob
                    module.drop_prob = 0
                #print(name, module.drop_prob)

    def compute_classify_loss(self, model, inputs):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights', None)
        if 'labels_mix' in inputs:
            labels_mix = inputs.pop('labels_mix')
        else:
            labels_mix = None
        outputs = model(**inputs)
        inputs['labels'] = labels
        if weights is not None:
            inputs['weights'] = weights
        logits = outputs.logits
        logits1, logits2, logits3 = logits[:, :-18], logits[:, -18:-12], logits[:, -12:]
        if model.training and self.args.semi_ratio>0:
            labels1, labels2, labels3 = labels[:, :len(util.binary_labels)], labels[:, -18:-12], labels[:, -12:]
            labels1 = labels1.to(logits1.dtype)
            loss1 = F.binary_cross_entropy_with_logits(logits1, labels1, reduction='sum', weight=weights) / len(logits1)
            loss2 = kl_div_loss(logits2, labels2)
            loss3 = kl_div_loss(logits3, labels3)
        else:
            labels1, labels2, labels3 = labels[:, :len(util.binary_labels)], labels[:, -2], labels[:, -1]
            labels1 = labels1.to(logits1.dtype)
            loss1 = F.binary_cross_entropy_with_logits(logits1, labels1, reduction='sum', weight=weights) / len(logits1)
            loss2 = F.cross_entropy(logits2, labels2.to(torch.long))
            loss3 = F.cross_entropy(logits3, labels3.to(torch.long))
        if labels_mix is not None:
            inputs['labels_mix'] = labels_mix
            labels1_mix, labels2_mix, labels3_mix = labels_mix[:, :len(util.binary_labels)], labels_mix[:, -2], labels_mix[:, -1]
            labels1_mix = labels1_mix.to(logits1.dtype)
            loss1_mix = F.binary_cross_entropy_with_logits(logits1, labels1_mix, reduction='sum') / len(logits1)
            loss2_mix = F.cross_entropy(logits2, labels2_mix.to(torch.long))
            loss3_mix = F.cross_entropy(logits3, labels3_mix.to(torch.long))
            loss1, loss2, loss3 = (loss1+loss1_mix)/2, (loss2+loss2_mix)/2, (loss3+loss3_mix)/2
        loss = (loss1 * self.args.w_bi + loss2 * self.args.w_lt + loss3 * self.args.w_wt)
        return loss, outputs, logits1, logits2, logits3

    def compute_sep_classify_loss(self, model, inputs):
        labels = inputs.pop('labels')
        if 'labels_mix' in inputs:
            labels_mix = inputs.pop('labels_mix')
        else:
            labels_mix = None
        outputs = model(**inputs)
        inputs['labels'] = labels
        logits = outputs.logits
        labels = labels.to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum') / len(logits)
        if labels_mix is not None:
            inputs['labels_mix'] = labels_mix
            labels_mix = labels_mix.to(logits.dtype)
            loss_mix = F.binary_cross_entropy_with_logits(logits, labels_mix, reduction='sum') / len(logits)
            loss = (loss+loss_mix)/2
        return loss, outputs, logits

    def compute_sep_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.args.is_classify:
            loss, outputs, logits = self.compute_sep_classify_loss(model, inputs)
            if self.args.bi_rdrop>0 and model.training and self.curr_train_step>=self.args.dp_start:
                r_loss, _, r_logits = self.compute_sep_classify_loss(model, inputs)
                loss = (loss+r_loss)/2
                if self.args.rdrop>0:
                    rloss = F.mse_loss(r_logits, logits)
                    loss = loss + rloss*self.args.bi_rdrop
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def pool(self, hidden_states, attention_mask):
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        ends = ends.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.gather(hidden_states, 1, ends.to(hidden_states.device)).squeeze(1)
        return hidden_states


    def _ar_loss(self, model, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def _semi_loss(self, model, logits, labels, tokids):
        logits = logits[:, -2, tokids[0]]
        #loss = kl_div_loss(logits, labels)
        loss = F.cross_entropy(logits, labels)
        return loss


    def compute_ar_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop('labels', None)
        #if self.label_smoother is not None and "labels" in inputs:
        #    labels = inputs.pop("labels")
        #else:
        #    labels = None
        outputs = model(inputs["input_ids"], attention_mask=inputs['attention_mask'])
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
#        if self.args.past_index >= 0:
#            self._past = outputs[self.args.past_index]
#
#        if labels is not None:
#            unwrapped_model = self.accelerator.unwrap_model(model)
#            if _is_peft_model(unwrapped_model):
#                model_name = unwrapped_model.base_model.model._get_name()
#            else:
#                model_name = unwrapped_model._get_name()
#            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                loss = self.label_smoother(outputs, labels, shift_labels=True)
#            else:
#                loss = self.label_smoother(outputs, labels)
#        else:
#            if isinstance(outputs, dict) and "loss" not in outputs:
#                raise ValueError(
#                    "The model did not return a loss from the inputs, only the following keys: "
#                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                )
#            # We don't use .loss here since the model may return tuples instead of ModelOutput.
#            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        #attention_mask = inputs['attention_mask']
        #ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        #labels = torch.gather(labels, 1, ends).squeeze(1)

        #ends = ends.unsqueeze(-1).repeat(1, 1, logits.shape[-1])
        #logits = torch.gather(logits, 1, ends).squeeze(1)
        #if self.args.semi_ratio>0 and inputs['is_semi'][0]:
        if self.args.semi_ratio > 0:
            loss = self._semi_loss(model, logits, inputs['orig_labels'], inputs['tokids'])
        else:
            loss = self._ar_loss(model, logits, labels)
        if (self.args.bi_rdrop>0 or self.args.cat_rdrop1>0 or self.args.cat_rdrop2>0):
            tokids = inputs['tokids'][0]
            logits1 = logits[:, -2, tokids]
            outputs2 = model(inputs["input_ids"], attention_mask=inputs['attention_mask'])
            logits2 = outputs2["logits"] if isinstance(outputs2, dict) else outputs2[0]
            loss2 = self._ar_loss(model, logits2, labels)
            loss = (loss+loss2)/2
            logits2 = logits2[:, -2, tokids]
            rloss = rdrop_loss2(logits1, logits2)
            loss = loss + rloss*(inputs['rweight'][0])



        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if model.training:
            self.curr_train_step += 1
            if self.args.dp_start>0 and self.curr_train_step==self.args.dp_start:
                unwrapped_model = self.accelerator.unwrap_model(model)
                self.update_dropout(unwrapped_model)
                print("updated dropout")


        if self.args.is_classify:
            if self.args.train_cols is not None:
                return self.compute_sep_loss(model, inputs, return_outputs=return_outputs)
            loss, outputs, logits1, logits2, logits3 = self.compute_classify_loss(model, inputs)
            if self.args.bi_rdrop>0 and model.training and self.curr_train_step>=self.args.dp_start:
                r_loss, _, r_logits1, r_logits2, r_logits3 = self.compute_classify_loss(model, inputs)
                loss = (loss+r_loss)/2
                if self.args.bi_rdrop>0:
                    bi_rloss = F.mse_loss(r_logits1, logits1)
                    loss = loss + bi_rloss*self.args.bi_rdrop
                if self.args.cat_rdrop1>0:
                    cat_rloss1 = rdrop_loss(logits2, r_logits2)
                    loss = loss + cat_rloss1*self.args.cat_rdrop1
                if self.args.cat_rdrop2>0:
                    cat_rloss2 = rdrop_loss(logits3, r_logits3)
                    loss = loss + cat_rloss2*self.args.cat_rdrop2
            return (loss, outputs) if return_outputs else loss
        elif (self.args.bi_rdrop > 0 or self.args.cat_rdrop1 > 0 or self.args.cat_rdrop2 > 0) or self.args.semi_ratio>0:
            return self.compute_ar_loss(model, inputs, return_outputs=return_outputs)
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

class TrainerSplit(Trainer):
    def compute_semi_classify_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop('labels')
        if 'labels_mix' in inputs:
            labels_mix = inputs.pop('labels_mix')
        else:
            labels_mix = None
        outputs1 = model(inputs['input_ids1'], attention_mask=inputs['attention_mask1'])
        outputs2 = model(inputs['input_ids2'], attention_mask=inputs['attention_mask2'])
        inputs['labels'] = labels
        logits1, logits2 = outputs1.logits, outputs2.logits
        logits, _ = torch.max(torch.stack([logits1*inputs['mask1'][:, None], logits2*inputs['mask2'][:, None]], axis=-1), axis=-1)
        logits1, logits2, logits3 = logits[:, :-18], logits[:, -18:-12], logits[:, -12:]

        labels1, labels2, labels3 = labels[:, :len(util.binary_labels)], labels[:, -18:-12], labels[:, -12:]
        loss1 = F.binary_cross_entropy_with_logits(logits1, labels1, reduction='sum') / len(logits1)
        loss2 = F.cross_entropy(logits2, labels2)
        loss3 = F.cross_entropy(logits3, labels3)
        #loss2 = kl_div_loss(logits2, labels2)
        #loss3 = kl_div_loss(logits3, labels3)


        loss = (loss1 * self.args.w_bi + loss2 * self.args.w_lt + loss3 * self.args.w_wt)
        return loss, outputs1, logits1, logits2, logits3
    def compute_classify_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if model.training and self.args.semi_ratio > 0:
            return self.compute_semi_classify_loss(model, inputs, return_outputs=False)
        labels = inputs.pop('labels')
        if 'labels_mix' in inputs:
            labels_mix = inputs.pop('labels_mix')
        else:
            labels_mix = None
        outputs1 = model(inputs['input_ids1'], attention_mask=inputs['attention_mask1'])
        outputs2 = model(inputs['input_ids2'], attention_mask=inputs['attention_mask2'])
        inputs['labels'] = labels
        logits1, logits2 = outputs1.logits, outputs2.logits
        logits, _ = torch.max(torch.stack([logits1*inputs['mask1'][:, None], logits2*inputs['mask2'][:, None]], axis=-1), axis=-1)
        logits1, logits2, logits3 = logits[:, :-18], logits[:, -18:-12], logits[:, -12:]
        labels1, labels2, labels3 = labels[:, :len(util.binary_labels)], labels[:, -2], labels[:, -1]
        loss1 = F.binary_cross_entropy_with_logits(logits1, labels1.to(logits1.dtype), reduction='sum') / len(logits1)
        loss2 = F.cross_entropy(logits2, labels2)
        loss3 = F.cross_entropy(logits3, labels3)
        if labels_mix is not None:
            inputs['labels_mix'] = labels_mix
            labels1_mix, labels2_mix, labels3_mix = labels_mix[:, :len(util.binary_labels)], labels_mix[:, -2], labels_mix[:, -1]
            labels1_mix = labels1_mix.to(logits1.dtype)
            loss1_mix = F.binary_cross_entropy_with_logits(logits1, labels1_mix, reduction='sum') / len(logits1)
            loss2_mix = F.cross_entropy(logits2, labels2_mix)
            loss3_mix = F.cross_entropy(logits3, labels3_mix)
            loss1, loss2, loss3 = (loss1 + loss1_mix) / 2, (loss2 + loss2_mix) / 2, (loss3 + loss3_mix) / 2
        loss = (loss1 * self.args.w_bi + loss2 * self.args.w_lt + loss3 * self.args.w_wt)
        return loss, outputs1, logits1, logits2, logits3


@dataclass
class TrainingArguments(HFTrainingArguments):
    train_cols: Union[None, List[str]] = field( default=None, metadata={"help": "The list of integrations to report the results and logs to."} )
    cat_rdrop1: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    cat_rdrop2: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    rdrop: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    bi_rdrop: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    cat_ls: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    bi_ls: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    w_bi: float = field(default=0.33333, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    w_lt: float = field(default=0.33333, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    w_wt: float = field(default=0.33333, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    semi_ratio: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    is_classify: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_kl: bool = field(default=False, metadata={"help": "Whether to run training."})
    dp_start: int = field(default=0, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."})

    use_badam: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_sampler: bool = field(default=False, metadata={"help": "Whether to run training."})
    switch_block_every: int = field(default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."})
    hard_ratio: float = field( default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})

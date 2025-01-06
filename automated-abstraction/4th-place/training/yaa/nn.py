import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification, ACT2FN, DebertaV2Model, SequenceClassifierOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForSequenceClassification
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class AVGContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states, attention_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        pooled_output = self.dropout(pooled_output)
        #pooled_output = self.dropout(pooled_output.to(hidden_states.dtype))
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size



class CustDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super(DebertaV2ForSequenceClassification, self).__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = AVGContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            inputs_embeds = None,
            labels = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class Classify(nn.Module):
    def __init__(self, args, backbone, n_label):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.num_labels = n_label

        self.classifier = nn.Linear(self.backbone.config.hidden_size, n_label)
        self.dropout = StableDropout(args.cls_dp)

    def pool(self, hidden_states, attention_mask):
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        ends = ends.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.gather(hidden_states, 1, ends.to(hidden_states.device)).squeeze(1)
        return hidden_states

    def forward(self,  input_ids=None, attention_mask=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.backbone(
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



from transformers.models.longformer.modeling_longformer import LongformerModel, LongformerLMHead, LongformerPreTrainedModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model, DebertaV2OnlyMLMHead, DebertaV2PreTrainedModel
from torch import nn

class LongformerForMaskedLM(LongformerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.init_weights()
        self.mask_token_id = config.mask_token_id
        
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self, inputs):
        outputs = self.longformer(**inputs)
        sequence_output = outputs[0][inputs["input_ids"] == self.mask_token_id]
        prediction_scores = self.lm_head(sequence_output) 
        return prediction_scores        
 

class DebertaForMaskedLM(DebertaV2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.lm_head = DebertaV2OnlyMLMHead(config)
        self.init_weights()
        self.mask_token_id = config.mask_token_id
        
    def get_output_embeddings(self):
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.predictions.decoder = new_embeddings
        self.lm_head.predictions.bias = new_embeddings.bias

    def forward(self, inputs):
        outputs = self.deberta(**inputs)
        sequence_output = outputs[0][inputs["input_ids"] == self.mask_token_id]
        prediction_scores = self.lm_head(sequence_output) 
        return prediction_scores

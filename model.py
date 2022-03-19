import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from prefix_ner_bert import BertModel, BertOnlyNSPHead
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput, MaskedLMOutput, NextSentencePredictorOutput
from transformers import BertPreTrainedModel
from transformers import BertModel as TransformerBertModel
from transformers.activations import ACT2FN

class BertAttentionFfnAdapterForSequenceClassification(BertPreTrainedModel):
    def __init__(self, bert_config, ffn_adapter_size, prefix_len=0):
        super(BertAttentionFfnAdapterForSequenceClassification, self).__init__(bert_config)
        self.bert = BertModel(bert_config, ffn_adapter_size=ffn_adapter_size)
        self.prefix_len = prefix_len
        self.num_labels = bert_config.num_labels

        self.n_layer = bert_config.num_hidden_layers
        self.n_head = bert_config.num_attention_heads
        self.n_embd = bert_config.hidden_size // bert_config.num_attention_heads

        self.prefix_embedding = None
        self.prefix_input_ids = None
        if prefix_len > 0:
            print('add past key values')
            self.prefix_embedding = nn.Embedding(prefix_len, bert_config.num_hidden_layers * 2 * bert_config.hidden_size)
            self.prefix_input_ids = torch.tensor([i for i in range(prefix_len)])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_input_ids.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_embedding(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        batch_size = len(input_ids)
        past_key_values = None
        if self.prefix_embedding is not None:
            past_key_values = self.get_prompt(batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertAttentionFfnAdapterForTokenClassification(BertPreTrainedModel):
    def __init__(self, bert_config, ffn_adapter_size, prefix_len=0):
        super(BertAttentionFfnAdapterForTokenClassification, self).__init__(bert_config)
        self.bert = BertModel(bert_config, ffn_adapter_size=ffn_adapter_size)
        self.prefix_len = prefix_len
        self.num_labels = bert_config.num_labels

        self.n_layer = bert_config.num_hidden_layers
        self.n_head = bert_config.num_attention_heads
        self.n_embd = bert_config.hidden_size // bert_config.num_attention_heads

        self.prefix_embedding = None
        self.prefix_input_ids = None
        if prefix_len > 0:
            print('add past key values')
            self.prefix_embedding = nn.Embedding(prefix_len, bert_config.num_hidden_layers * 2 * bert_config.hidden_size)
            self.prefix_input_ids = torch.tensor([i for i in range(prefix_len)])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_input_ids.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_embedding(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        batch_size = len(input_ids)
        past_key_values = None
        if self.prefix_embedding is not None:
            past_key_values = self.get_prompt(batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertAttentionFfnAdapterForMaskedLM(BertPreTrainedModel):
    def __init__(self, bert_config, ffn_adapter_size, prefix_len=0):
        super(BertAttentionFfnAdapterForMaskedLM, self).__init__(bert_config)
        self.bert = BertModel(bert_config, ffn_adapter_size=ffn_adapter_size)
        self.prefix_len = prefix_len

        self.n_layer = bert_config.num_hidden_layers
        self.n_head = bert_config.num_attention_heads
        self.n_embd = bert_config.hidden_size // bert_config.num_attention_heads

        self.prefix_embedding = None
        self.prefix_input_ids = None
        if prefix_len > 0:
            print('add past key values')
            self.prefix_embedding = nn.Embedding(prefix_len, bert_config.num_hidden_layers * 2 * bert_config.hidden_size)
            self.prefix_input_ids = torch.tensor([i for i in range(prefix_len)])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.cls = BertOnlyMLMHead(bert_config)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_input_ids.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_embedding(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        use_cache=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        # batch_size = len(input_ids)
        # past_key_values = None
        # if self.prefix_embedding is not None:
        #     past_key_values = self.get_prompt(batch_size)
        #     prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.bert.device)
        #     attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,

            # 擅自把attention改成past_key_values
            attentions=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertForMaskLMAndNSP(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertForMaskLMAndNSP, self).__init__(bert_config)
        self.bert = TransformerBertModel(bert_config)
        self.cls = BertOnlyMLMHead(bert_config)
        self.nsp_cls = BertOnlyNSPHead(bert_config)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        use_cache=None,
        is_mlm=True,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        if is_mlm:
            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,

                # 擅自把attention改成past_key_values
                attentions=outputs.past_key_values,
            )
        else:
            pooled_output = outputs[1]

            seq_relationship_scores = self.nsp_cls(pooled_output)

            next_sentence_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

            if not return_dict:
                output = (seq_relationship_scores,) + outputs[2:]
                return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

            return NextSentencePredictorOutput(
                loss=next_sentence_loss,
                logits=seq_relationship_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
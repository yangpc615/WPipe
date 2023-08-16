import torch
from transformers.modeling_outputs import BaseModelOutput
from torch.nn import CrossEntropyLoss, MSELoss


def callback_bert_0(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else
                            self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape,
                                     dtype=torch.long,
                                     device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        attention_mask, input_shape, device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
        )
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(
            encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(input_ids=input_ids,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       inputs_embeds=inputs_embeds)
    hidden_states = embedding_output
    attention_mask = extended_attention_mask
    encoder_attention_mask = encoder_extended_attention_mask
    # encoder
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        layer_head_mask = head_mask[i] if head_mask is not None else None

        if getattr(self.config, "gradient_checkpointing", False):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1], )

    return {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "layer_head_mask": layer_head_mask,
        "head_mask": None,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "output_attentions": output_attentions,
        "all_attentions": all_attentions,
        "all_hidden_states": all_hidden_states,
        "output_hidden_states": output_hidden_states,
        "labels": labels
    }


def callback_bert_i(self,
                    hidden_states=None,
                    attention_mask=None,
                    layer_head_mask=None,
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    all_attentions=None,
                    all_hidden_states=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    labels=None):
    all_hidden_states = all_hidden_states if output_hidden_states else None
    all_attentions = all_attentions if output_attentions else None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        layer_head_mask = head_mask[i] if head_mask is not None else None

        if getattr(self.config, "gradient_checkpointing", False):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1], )

    return {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "layer_head_mask": layer_head_mask,
        "head_mask": None,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "output_attentions": output_attentions,
        "all_attentions": all_attentions,
        "all_hidden_states": all_hidden_states,
        "labels": labels
    }


def callback_bert_n_1(
        self,
        hidden_states=None,
        attention_mask=None,
        layer_head_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        all_attentions=None,
        all_hidden_states=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
):
    all_hidden_states = all_hidden_states if output_hidden_states else None
    all_attentions = all_attentions if output_attentions else None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        layer_head_mask = head_mask[i] if head_mask is not None else None

        if getattr(self.config, "gradient_checkpointing", False):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1], )

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states, )

    encoder_outputs = tuple(
        v for v in [hidden_states, all_hidden_states, all_attentions]
        if v is not None)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(
        sequence_output) if self.pooler is not None else None

    outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

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

    output = (logits, ) + outputs[2:]
    return {"loss": loss, "output": output, "labels": labels}


callable_bert = []
callable_bert.append(callback_bert_0)
callable_bert.append(callback_bert_i)
callable_bert.append(callback_bert_n_1)

import types
import torch
import transformers
from torch import nn


class T5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.overwrite_forward_crossattention()
        self.config = config

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function
        """
        for i, mod in enumerate(self.decoder.block):
            attn = mod.layer[1].EncDecAttention
            attn.temp_scheduler = TempScheduler(
                temp_start=self.config.temp_start,
                temp_end=self.config.temp_end,
                total_steps=self.config.total_steps,
                accumulaton_steps=self.config.accumulation_steps,
                scheduler=self.config.scheduler,
            )
            attn.forward = types.MethodType(cross_attention_forward, attn)
            
            
class TempScheduler(nn.Module):
    def __init__(self, temp_start, temp_end, total_steps, accumulaton_steps=None, scheduler="linear"):
        super().__init__()
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.total_steps = total_steps
        self.accumulation_steps = accumulaton_steps
        self.scheduler = scheduler
        self.current_step = 0
        if self.accumulation_steps:
            self.small_step = 0  

    def linear_scheduler(self):
        return (self.current_step / self.total_steps) * (self.temp_end - self.temp_start) + self.temp_start

    def constant_scheduler(self):
        return self.temp_start

    def quadratic_scheduler(self):
        return (self.current_step / self.total_steps)**2 * (self.temp_end - self.temp_start) + self.temp_start

    def forward(self, training=None):
        if training:  # only used in training
            if self.accumulation_steps:
                self.small_step += 1
                if self.small_step == self.accumulation_steps:
                    self.small_step = 0
                    self.current_step += 1
            else:
                self.current_step = self.current_step + 1

        if self.scheduler == "linear":
            return self.linear_scheduler()
        elif self.scheduler == "constant":
            return self.constant_scheduler()
        elif self.scheduler == "quadratic":
            return self.quadratic_scheduler()
        else:
            raise ValueError(f"{self.scheduler} scheduler is not supported!")


def cross_attention_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        temp = self.temp_scheduler(self.training) # Update the temperature
        scores /=  temp
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

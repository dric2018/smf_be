"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 26 Oct, 2023

# Code Description
======================
Description: A re-implementation of the transformer architecture from [1]

# Adaptation Information
==========================
Adapted from:
- https://github.com/devjwsong/transformer-translator-pytorch/tree/master/src [Jaewoo (Kyle) Song]
- https://github.com/google-research/robotics_transformer/blob/master/transformer.py [Original RT1]
- https://github.com/lucidrains/robotic-transformer-pytorch/blob/main/robotic_transformer_pytorch/robotic_transformer_pytorch.py
# References
=============
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). (http://papers.nips.cc/paper/7181-attention-is-all-you-need)

"""
import config
from einops import repeat

import math
import torch
import torch.nn as nn


def generate_masks(src_sequence, target_sequence=None):
    
    target_mask = None

    # Create a mask that is 1 where there is input data and 0 where there is padding
    src_mask = (src_sequence != config.SRC_PAD_TOK_ID).float().unsqueeze(1).unsqueeze(2)
    
    if target_sequence is not None:
        # Create a mask that is 1 for positions less than the current position
        batch_size, seq_len = target_sequence.shape
        target_mask = (1 - torch.triu(torch.ones(batch_size, seq_len, seq_len, device=target_sequence.device), diagonal=1))
        
    return src_mask, target_mask

def generate_causal_attention_mask():
    """
        Args:
            None

        Returns: 
            attn_mask: causal attention mask matching the shape of the learned tokens
    """
    
    attn_mask = torch.ones((config.NUM_HISTORY+1, config.NUM_HISTORY+1), dtype = torch.bool, device = config.DEVICE).triu(1)
    attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = config.NUM_LEARNED_TOKENS, r2 = config.NUM_LEARNED_TOKENS)
        
    return ~attn_mask # return inverted mask for causal attention


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model:int=config.D_MODEL):
        super(SelfAttentionHead, self).__init__()
        self.inf = 1e9
        self.d_k = d_model
        
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
        self._softmax = nn.Softmax(dim=-1)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def self_attention(
        self, 
        inp:torch.Tensor, 
        mask=None, 
        return_weights=True
    ):
        B, L, D = inp.shape
        
        # Linear calculation
        q = self.w_q(inp)
        k = self.w_k(inp)
        v = self.w_v(inp)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -self.inf)

        attention_weights = self._softmax(scaled_attention_logits)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v) # (B, L, num_heads, D)        
        if return_weights:
            return output, attention_weights
        else:
            return output, None

    def forward(
        self, 
        inp:torch.Tensor, 
        mask=None, 
        return_weights=False
    ):
        
        attn_values, attn_W = self.self_attention(
            inp, 
            mask=mask, 
            return_weights=return_weights
        )
        
        return attn_values, attn_W

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        num_heads:int=config.N_HEADS
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attention_heads = nn.ModuleList([SelfAttentionHead(d_model) for _ in range(num_heads)])
        
        self.output_layer = nn.Sequential(
            nn.Linear(config.SELF_ATTENTION_OUT_DIM, d_model, bias=False),
            nn.Dropout(p=config.DROPOUT_RATE)
        )
        
    def forward(
        self, 
        inp:torch.Tensor,  
        mask=None, 
        return_weights=True
    ):
        B, L, D = inp.shape
        
        head_outputs = [head(inp, mask=mask, return_weights=return_weights) for head in self.attention_heads]
        
        # Combine the results from different heads
        combined_output = torch.cat(
            [output[0].view(B, -1, self.num_heads, self.head_dim) for output in head_outputs], 
            dim=-1
        )
        attention_weights = torch.stack(
            [output[1] for output in head_outputs], 
            dim=1
        )
        
        combined_output = combined_output.contiguous().view(B, L, -1)
        
        combined_output = self.output_layer(combined_output)        

        return combined_output, attention_weights

class CrossAttentionHead(nn.Module):
    def __init__(
        self, 
        embed_dim:int=config.D_MODEL, 
        dropout_rate:float=config.DROPOUT_RATE
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        
        # Linear transformations for keys, and values for input sequences
        self.w_k_input = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v_input = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Linear transformation for output sequences
        self.w_q_output = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, input_seq, input_lens, output_seq, apply_mask=False):
        
        B, max_len = input_seq.shape[0], input_seq.shape[1]
        
        # Linear transformations for both input sequences
        k_input = self.w_k_input(input_seq)
        v_input = self.w_v_input(input_seq)
        
        # Linear transformations for both output sequences
        q_output = self.w_q_output(output_seq)

        # Calculate cross-attention scores
        attn_scores = torch.matmul(q_output, k_input.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.embed_dim)
        
        if apply_mask:
            # mask padded tokens in input sequence
            input_mask = torch.arange(max_len).expand(B, -1) < input_lens.view(-1, 1)
            input_mask = input_mask.to(attn_scores.device)

            # mask out corresponding attention scores
            attn_scores = attn_scores.masked_fill(~input_mask.unsqueeze(-1), -config.INF)

        # Softmax
        attn_W = self._softmax(attn_scores)
        attn_W = self.dropout(attn_W)

        # Calculate cross-attention values
        attn_out = torch.matmul(attn_W, v_input) 

        return attn_out, attn_W
        

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self, 
        embed_dim:int=config.D_MODEL, 
        num_heads:int=config.N_HEADS, 
        dropout_rate:float=config.DROPOUT_RATE
    ):
        super().__init__()
        
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.dropout_rate = dropout_rate

        self.attention_heads = nn.ModuleList([CrossAttentionHead(self.d_model) for _ in range(self.num_heads)])

        
        self.dropout = nn.Dropout(self.dropout_rate)
        self._softmax = nn.Softmax(dim=-1)

    def forward(
        self, 
        input_seq, 
        input_lens, 
        output_seq, 
        apply_mask=False
    ):
        
        B = input_seq.shape[0]
        
        head_outputs = [head(input_seq, input_lens, output_seq, apply_mask) for head in self.attention_heads]
        
        # Combine the results from different heads
        combined_output = torch.cat(
            [output[0].view(B, -1, self.num_heads, self.head_dim) for output in head_outputs], 
            dim=-1
        )
        attention_weights = torch.stack(
            [output[1] for output in head_outputs], 
            dim=1
        )

        return combined_output, attention_weights

class FeedFowardLayer(nn.Module):
    def __init__(
        self, 
        in_dim:int=config.D_MODEL, 
        mlp_dim:int=config.D_FF, 
        out_dim:int=config.D_MODEL, 
        dropout_rate:float=config.DROPOUT_RATE,
        activation_fn:str="ReLU"
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim*config.EXPANSION, bias=True)
        self.activation = getattr(nn, activation_fn)()
        self.linear_2 = nn.Linear(mlp_dim*config.EXPANSION, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.activation(self.linear_1(x)) # (B, L, config.D_FF)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, config.D_MODEL)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([config.D_MODEL], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        seq_len:int=config.NUM_TOKENIZED_INPUTS
    ):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, config.D_MODEL) # (L, config.D_MODEL)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(config.D_MODEL):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / config.D_MODEL)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / config.D_MODEL)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, config.D_MODEL)
        self.positional_encoding = pe_matrix.to(device=config.DEVICE).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(config.D_MODEL) # (B, L, config.D_MODEL)
        x = x + self.positional_encoding # (B, L, config.D_MODEL)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        nhead:int=config.N_HEADS, 
        dim_feedforward:int=config.D_FF, 
        dropout:float=config.DECODER_DROPOUT_RATE
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention()
        
        # Layer normalization 1
        self.norm1 = nn.LayerNorm(d_model)
        
        self.multihead_attn = MultiHeadAttention()

        # Layer normalization 2
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization after feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, encoder_outputs, y=None, src_mask=None, y_mask=None):
        
        if src_mask is None:
            src_mask, y_mask = generate_masks(src, y)
        
        # Multi-head self-attention
        self_attn_output = self.self_attn(src, src, src, mask=y_mask)
        # Apply dropout and add the residual connection
        src = src + self.dropout(self_attn_output)
        # Layer normalization 1
        src = self.norm1(src)
        # Multi-head attention over encoder outputs
        multihead_attn_output = self.multihead_attn(src, encoder_outputs, encoder_outputs, mask=src_mask)
        # Apply dropout and add the residual connection
        src = src + self.dropout(multihead_attn_output)
        # Layer normalization 2
        src = self.norm2(src)
        # Position-wise feed-forward network
        ffn_output = self.ffn(src)
        # Apply dropout and add the residual connection
        src = src + self.dropout(ffn_output)
        # Layer normalization 3
        src = self.norm3(src)

        return src
    

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers:int=config.N_DECODER_LAYERS
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.target_embedding = nn.Embedding(
            num_embeddings=config.TARGET_VOCAB_SIZE, 
            embedding_dim=config.D_MODEL, 
            padding_idx=config.TARGETS_MAPPING["[PAD]"]
        )
        self.layers = nn.ModuleList([
            TransformerDecoderLayer()
            for _ in range(num_layers)
        ])
        
        self._initialize()
    
    def _initialize(self):
        
        # Glorot / fan_avg. Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            
    def forward(self, src, encoder_outputs, y=None, src_mask=None, y_mask=None):
        if y_mask is None or src_mask is None:
            src, y_mask = generate_masks(src, y)
        for layer in self.layers:
            src = layer(src, encoder_outputs, src_mask, y_mask)

        return src
    
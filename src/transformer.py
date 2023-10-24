"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 19 Oct, 2023

# Code Description
======================
Description: A re-implementation of the transformer architecture from [1]

# Adaptation Information
==========================
Adapted from:
- Original Source: https://github.com/devjwsong/transformer-translator-pytorch/tree/master/src
- Original Author: Jaewoo (Kyle) Song

# References
=============
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). (http://papers.nips.cc/paper/7181-attention-is-all-you-need)

"""
import config

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


class MultiHeadAttention(nn.Module):
    def __init__(
        self):
        super().__init__()
        
        self.inf = 1e9        
        self.d_k = config.D_K
        self.n_heads = config.N_HEADS
        self.embed_dim = config.D_MODEL
        
        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self._softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.output_layer = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
           k : key vector
           q : query vector
           v : value vector
           mask: mask for decoder
        
        Returns:
           output vector
        """
        input_shape = q.shape
        
        print(f"q: {q.shape} - k: {k.shape} - v: {v.shape} ")
        if mask is not None:
            print(f"mask: {mask.shape}")
        
        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.n_heads, self.d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.n_heads, self.d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.n_heads, self.d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, self.embed_dim) # (B, L, config.D_MODEL)

        return self.output_layer(concat_output)

    def self_attention(self, q, k, v, mask=None):
        
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax 
        attn_W = self._softmax(attn_scores)
        attn_W = self.dropout(attn_W)
        
        # Calculate values
        attn_values = torch.matmul(attn_W, v) # (B, num_heads, L, d_k)

        return attn_values


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
        self.linear_1 = nn.Linear(in_dim, mlp_dim, bias=True)
        self.activation = getattr(nn, activation_fn)()
        self.linear_2 = nn.Linear(mlp_dim, out_dim, bias=True)
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
        seq_len
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
    
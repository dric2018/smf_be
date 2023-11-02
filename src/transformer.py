"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 31 Oct, 2023

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

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_masks(src_sequence, target_sequence=None):
    
    target_mask = None

    # Create a mask that is 1 where there is input data and 0 where there is padding
    src_mask = (src_sequence != config.SRC_PAD_TOK_ID).float().unsqueeze(1).unsqueeze(2)
    
    if target_sequence is not None:
        # Create a mask that is 1 for positions less than the current position
        batch_size, seq_len = target_sequence.shape
        target_mask = (1 - torch.triu(torch.ones(batch_size, seq_len, seq_len, device=target_sequence.device), diagonal=1))
        
    return src_mask, target_mask

def generate_causal_attention_mask(
    dim:int=config.NUM_HISTORY+1,
    num_learned_tokens:int=config.NUM_LEARNED_TOKENS,
    for_learned_tokens:bool=False
):
    """
        Args:
            dim: (int) size to be used to create causal attention mask

        Returns: 
            attn_mask: causal attention mask of shape (1, seq_len, seq_len)
    """
    
    if for_learned_tokens:
        attn_mask = torch.ones(
            (dim, dim), 
            dtype = torch.bool, 
            # device = config.DEVICE
        ).triu(1)

        attn_mask = repeat(
            attn_mask, 
            'i j -> (i d1) (j d2)', 
            d1 = num_learned_tokens, 
            d2 = num_learned_tokens
        ).unsqueeze(0)
    else:
        attn_mask = torch.ones(
            (1, dim, dim), 
            dtype = torch.bool, 
            # device = config.DEVICE
        ).triu(1)
        
    return ~attn_mask # return inverted mask for causal attention


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model:int=config.D_MODEL):
        super().__init__()
        self.inf = 1e9
        self.d_k = d_model
        
        self.dropout = nn.Dropout(p=config.DECODER_DROPOUT_RATE)
        self._softmax = nn.Softmax(dim=-1)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def self_attention(
        self, 
        q:torch.Tensor,
        k:torch.Tensor,  
        v:torch.Tensor,          
        mask=None, 
        return_weights=True
    ):
        B, L, D = q.shape
        
        # Linear calculation
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Scaled Dot-Product Attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.d_k)

        if mask is not None:
            scaled_attention_logits.masked_fill(mask == config.TGT_PAD_TOK_ID, -self.inf)

        attention_weights = self._softmax(scaled_attention_logits)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        
        if return_weights:
            return context, attention_weights
        else:
            return context, None

    def forward(
        self, 
        q:torch.Tensor,
        k:torch.Tensor,  
        v:torch.Tensor,         
        mask=None, 
        return_weights=False
    ):
        
        context, attn_W = self.self_attention(
            q,
            k,
            v, 
            mask=mask, 
            return_weights=return_weights
        )
        
        return context, attn_W

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        num_heads:int=config.N_HEADS
    ):
        super().__init__()
        

        
        assert d_model % num_heads ==0, f"d_model should be divisible by num_heads. Found d_model={d_model} and num_heads={num_heads}"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.num_heads

        self.attention_heads = nn.ModuleList([SelfAttentionHead(self.d_model) for _ in range(self.num_heads)])
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model * self.num_heads, self.d_model, bias=False),
            nn.Dropout(p=config.DECODER_DROPOUT_RATE)
        )
        
    def forward(
        self, 
        q:torch.Tensor,
        k:torch.Tensor,  
        v:torch.Tensor,  
        mask=None, 
        return_weights=True
    ):
        B, L, D = q.shape
        # print("MHSA in: ", q.shape)
        
        head_outputs = [head(q, k, v, mask=mask, return_weights=return_weights) for head in self.attention_heads]
        
        # Combine the results from different heads
        combined_output = torch.cat(
            [output[0].view(B, -1, self.num_heads, self.head_dim) for output in head_outputs], 
            dim=-1
        )
        attention_weights = torch.stack(
            [output[1] for output in head_outputs], 
            dim=1
        )
        
        combined_output = combined_output.contiguous().view(B, L, self.d_model * self.num_heads)
        # print(f"combined: {combined_output.shape}")
        context = self.output_layer(combined_output)        

        return context, attention_weights

class FeedFowardLayer(nn.Module):
    def __init__(
        self, 
        in_dim:int=config.D_MODEL, 
        mlp_dim:int=config.D_FF, 
        out_dim:int=config.D_MODEL, 
        dropout_rate:float=config.DECODER_DROPOUT_RATE,
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
    def __init__(self, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, config.D_MODEL)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.D_MODEL, 2, dtype=torch.float) * (-math.log(10000.0) / config.D_MODEL))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(x.size(-1))
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        n_selfattention_heads:int=config.N_HEADS, 
        n_crossattention_heads:int=config.N_HEADS,
        dim_feedforward:int=config.D_FF, 
        dropout:float=config.DECODER_DROPOUT_RATE
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(num_heads=n_selfattention_heads)
        
        # Multi-head Cross-attention
        self.cross_attn = MultiHeadSelfAttention(num_heads=n_crossattention_heads)
        
        # Layer normalization 1
        self.norm1 = LayerNormalization()
        
        # Layer normalization 2
        self.norm2 = LayerNormalization()
        
        # Position-wise feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization after feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        inp:torch.Tensor, 
        encoder_outs:Tuple[torch.Tensor, torch.Tensor],
        src_mask:Tuple[torch.Tensor, torch.Tensor]=(None, None), 
        target_mask:torch.tensor=None,
        return_weights:bool=True,
        debug:bool=False
    ):
        """
            Args:
                inp:torch.Tensor 
                encoder_outs:Tuple[torch.Tensor, torch.Tensor]
                     (encoded input sequence, learned tokens)
                    
                src_mask:torch.Tensor 
                target_mask:torch.tensor
                return_weights:bool
                debug:bool
                
            Returns:
                inp: 
                
                self_attn_W: 
                
                cross_attn_W_seq: Cross attention weights for input sequence-to-output sequence mapping
                
                cross_attn_W_tokens: Cross attention weights for learned tokens-to-output sequence mapping
                
        """
        
        # inp = encoder_out # reserve this for cross attention
        
        if debug:
            print(f"inp shape: {inp.shape}")
        # Layer normalization 1
        inp_ = self.norm1(inp)
        if debug:
            print(f"LN 1 out shape: {inp_.shape}")

        # Multi-head self-attention
        self_attn_W = None
        self_attn_output = self.self_attn(q=inp, k=inp, v=inp, mask=target_mask)
        
        if return_weights:
            inp_, self_attn_W = self_attn_output
        else:
            inp_ = self_attn_output
        
        if debug:
            print(f"MHSA out : {inp_.shape}")
        # Apply dropout and add the residual connection
        inp = inp + self.dropout(inp_)
        
        # Layer normalization 2
        out = self.norm2(inp)
        if debug:
            print(f"LN 2 out shape: {out.shape}")
        # Apply FF
        ff_out = self.dropout(self.ff(out))
        if debug:
            print(f"FF out shape: {ff_out.shape}")
        # Add the residual connection
        inp = inp + ff_out
        
        # compute cross attention between encoder's outputs and decoder's prev. hidden states
        # print("MHCA 1")
        cross_attn_out, cross_attn_W_seq = self.cross_attn(
            q=inp, 
            k=encoder_outs[0], 
            v=encoder_outs[0], 
            mask=src_mask[0]
        )
        # print("MHCA 2")
        _, cross_attn_W_tokens = self.cross_attn(
            q=inp, 
            k=encoder_outs[1], 
            v=encoder_outs[1], 
            mask=src_mask[1]
        )

        if debug:
            print(f"MHCA out : {cross_attn_out.shape}")
        
        if debug:
            print(f"Final out shape: {inp.shape}")
            
        return inp, self_attn_W, cross_attn_W_seq, cross_attn_W_tokens

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers:int=config.N_DECODER_LAYERS
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # token embedding
        self.token_emb = nn.Linear(config.D_MODEL, config.D_MODEL, bias=False) 
        # positional embedding
        self.position_emb = nn.Linear(config.MAX_LEN, config.D_MODEL, bias=False)
               
        # transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer()
            for _ in range(num_layers)
        ])
        
        self._initialize()
    
    def _initialize(self):
        
        # Glorot / fan_avg. Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    def forward(
        self, 
        inp:torch.Tensor, 
        encoder_outs:Tuple[torch.Tensor, torch.Tensor],
        src_mask:Tuple[torch.Tensor, torch.Tensor]=(None, None), 
        target_mask:torch.tensor=None,
        debug:bool=False
    ):      
        
        B, L, D = inp.shape
        
        # Create positions tensor
        positions = torch.arange(0, config.MAX_LEN).unsqueeze(0).expand(B, -1).to(inp.device)
        positions = F.one_hot(positions, num_classes=config.MAX_LEN).to(torch.float32)
        positions = self.position_emb(positions)
        
        # Token embeddings
        inp = self.token_emb(inp)

        # Positional embeddings
        inp += positions[:, :L]
        
        # run self-attention modules
        self_attn_Ws = []
        cross_attn_Ws_seq = []
        cross_attn_Ws_tokens = []
        
        for layer in self.layers:
            inp, self_attn_W, cross_attn_W_seq, cross_attn_W_tokens = layer(
                inp=inp, 
                encoder_outs=encoder_outs, 
                src_mask=src_mask, 
                target_mask=target_mask,
                debug=debug
            )
            # store attention weights
            self_attn_Ws.append(self_attn_W)
            cross_attn_Ws_seq.append(cross_attn_W_seq)
            cross_attn_Ws_tokens.append(cross_attn_W_tokens)

        self_attn_Ws = torch.stack(self_attn_Ws, dim=1) 
        cross_attn_Ws_seq = torch.stack(cross_attn_Ws_seq, dim=1)
        cross_attn_Ws_tokens = torch.stack(cross_attn_Ws_tokens, dim=1)

        
        if self.num_layers == 1:
            self_attn_Ws = self_attn_Ws.squeeze(1)
            cross_attn_Ws_seq = cross_attn_Ws_seq.squeeze(1)
            cross_attn_W_tokens = cross_attn_W_tokens.squeeze(1)

        return inp, self_attn_Ws, cross_attn_Ws_seq, cross_attn_W_tokens
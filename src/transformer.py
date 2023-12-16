"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 15 Dec, 2023

# Code Description
======================
Description: A re-implementation of the transformer architecture from [1]

# Adaptation Information
==========================
Adapted from:
- https://github.com/hyunwoongko/transformer [Kevin Ko]
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


def make_attn_mask(dim:int=config.MAX_OUT_SEQ_LEN):
    attn_mask = torch.ones(
        (dim, dim), 
        dtype = torch.bool, 
        device = config.DEVICE
    ).triu(1)
            
    return ~attn_mask

class EmbeddingLayer(nn.Module):
    def __init__(
            self, 
            vocab_size:int=config.TARGET_VOCAB_SIZE, 
            d_model:int=config.D_MODEL, 
            max_len:int=config.MAX_OUT_SEQ_LEN, 
            drop_prob:float=config.DECODER_DROPOUT_RATE, 
            device:str=config.DEVICE
        ):

        super().__init__()

        self.tok_emb    = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=config.TGT_PAD_TOK_ID)
        self.pos_emb    = PositionalEncoding(d_model, max_len, device)
        self.drop_out   = nn.Dropout(p=drop_prob)

    def forward(self, x):
        
        tok_emb = self.tok_emb(x) * math.sqrt(config.D_MODEL)
        pos_emb = self.pos_emb(x)

        return self.drop_out(tok_emb + pos_emb)
    
    
class LayerNorm(nn.Module):
    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        eps=1e-12
    ):
        super().__init__()

        self.gamma  = nn.Parameter(torch.ones(d_model))
        self.beta   = nn.Parameter(torch.zeros(d_model))
        self.eps    = eps

    def forward(self, x):
        mean    = x.mean(-1, keepdim=True)
        var     = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out

class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self, 
            q:torch.Tensor, 
            k:torch.Tensor, 
            v:torch.Tensor, 
            mask=None, 
            e=1e-12
        ):

        batch_size, head, length, d_tensor = k.size()

        # compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
                
        # apply attention mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # conpute attention
        attn_w = self.softmax(score)

        # score Value
        context = attn_w @ v

        return context, attn_w
    

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model 

        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        batch_size, length, _ = q.shape


        # process Q, K, V
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # attention mechanism
        out, attention_w = self.attention(q, k, v, mask=mask)

        # compute context 
        # out = out.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        ## --> project
        out = self.w_concat(out)

        return out, attention_w

    def split(self, tensor):
        """
        split tensor based on the number of heads
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

class PositionalEncoding(nn.Module):

    def __init__(
            self, 
            d_model:int=config.D_MODEL, 
            max_len:int=config.MAX_OUT_SEQ_LEN, 
            device:str=config.DEVICE
        ):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):

        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]
    
class FeedForwardLayer(nn.Module):

    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        hidden:int=config.D_FF, 
        drop_prob=config.DECODER_DROPOUT_RATE
    ):
        super(FeedForwardLayer, self).__init__()
        
        self.linear1    = nn.Linear(d_model, hidden)
        self.linear2    = nn.Linear(hidden, d_model)
        self.gelu       = nn.GELU()
        self.dropout    = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    
class DecoderLayer(nn.Module):

    def __init__(
        self, 
        d_model:int=config.D_MODEL, 
        ffn_hidden:int=config.D_FF, 
        n_head:int=config.N_HEADS, 
        drop_prob:float=config.DECODER_DROPOUT_RATE
    ):
        super().__init__()

        self.self_attention     = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1              = LayerNorm(d_model=d_model)
        self.dropout1           = nn.Dropout(p=drop_prob)

        self.cross_attention    = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2              = LayerNorm(d_model=d_model)
        self.dropout2           = nn.Dropout(p=drop_prob)

        self.ffn                = FeedForwardLayer(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3              = LayerNorm(d_model=d_model)
        self.dropout3           = nn.Dropout(p=drop_prob)

    def forward(
            self, 
            dec_in:torch.Tensor, 
            enc_out:torch.Tensor, 
            trg_mask:torch.Tensor=None, 
            src_mask:torch.Tensor=None
        ):

        cross_attn_w = None

        # 1. compute self attention
        res = dec_in
        x, self_attn_w = self.self_attention(q=dec_in, k=dec_in, v=dec_in, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + res)

        if enc_out is not None:
            # 3. compute encoder - decoder attention
            res = x
            x, cross_attn_w = self.cross_attention(q=x, k=enc_out, v=enc_out, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + res)

        # 5. positionwise feed forward network
        res = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + res)

        return x, self_attn_w, cross_attn_w
    
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size:int=config.TARGET_VOCAB_SIZE, 
        max_len:int=config.MAX_OUT_SEQ_LEN, 
        d_model:int=config.D_MODEL, 
        ffn_hidden:int=config.D_FF, 
        n_head:int=config.N_HEADS, 
        num_layers:int=config.N_DECODER_LAYERS, 
        drop_prob:float=config.DECODER_DROPOUT_RATE,
        device:str=config.DEVICE
        ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.emb_layer = EmbeddingLayer()
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(self.num_layers)])

        self._initialize()

    def _initialize(self):
        # Glorot / fan_avg. Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                
    def forward(
            self, 
            dec_in:torch.Tensor, 
            enc_out:torch.Tensor, 
            attn_mask:torch.Tensor=None, 
            src_mask:torch.Tensor=None
        ):
        
        B, L = dec_in.shape
        self_attn_Ws = []
        cross_attn_Ws = []

        dec_in = self.emb_layer(dec_in)
        
        # init decoding
        out = dec_in
        if attn_mask is None:
            attn_mask = make_attn_mask()
                    
        for layer in self.layers:
            out, self_attn_w, cross_attn_w = layer(out, enc_out, attn_mask, src_mask)
            # store attention weights
            self_attn_Ws.append(self_attn_w)
            cross_attn_Ws.append(cross_attn_w)

        self_attn_Ws = torch.stack(self_attn_Ws, dim=1) 
        cross_attn_Ws = torch.stack(cross_attn_Ws, dim=1)
        
        if self.num_layers == 1:
            self_attn_Ws = self_attn_Ws.squeeze(1)
            cross_attn_Ws = cross_attn_Ws.squeeze(1)


        return out, self_attn_Ws, cross_attn_Ws
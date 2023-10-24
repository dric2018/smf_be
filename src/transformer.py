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


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerDecoderLayer(),
                FeedFowardLayer()
            ]))
            
    def forward(self, x, attn_mask):
        
        for attn, ff in self.layers:
            x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns, None)) + x
            x = ff(x, cond_fn = next(cond_fns, None)) + x
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self):
        super().__init__()
        
        self.inf = 1e9        
        self.head_dim = config.D_K
        
        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.w_v = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.output_layer = nn.Linear(self.n_heads*self.head_dim, self.embed_dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
           k : key vector
           q : query vector
           v : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)

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
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

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


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim:int=config.D_MODEL, 
        expansion_factor:int=config.EXPANSION, 
        n_heads:int=config.N_HEADS
    ):
        super().__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key,query,value)  
        attention_residual_out = attention_out + value  
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_fwd_out = self.feed_forward(norm1_out) 
        feed_fwd_residual_out = feed_fwd_out + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out
    
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim:int=config.D_MODEL, 
        expansion_factor:int=config.EXPANSION, 
        n_heads:int=config.N_HEADS
    ):
        super().__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        
    
    def forward(self, key, query, x,mask):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention 
        Returns:
           out: output of transformer block
    
        """
        
        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value)

        
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        target_vocab_size:int=len(config.TARGETS), 
        embed_dim:int=config.D_MODEL, 
        seq_len:int=config.MAX_LEN, 
        num_layers:int=config.N_DECODER_LAYERS, 
        expansion_factor:int=config.EXPANSION, 
        n_heads:int=config.N_HEADS
    ):
        super().__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEncoder(seq_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            mask: mask for decoder self attention
        Returns:
            out: output vector
        """
            
        
        x = self.word_embedding(x)  
        x = self.position_embedding(x) 
        x = self.dropout(x)
     
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 

        out = F.softmax(self.fc_out(x))

        return out
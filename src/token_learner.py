"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 31 Oct, 2023

# Code Description
======================
Description: A re-implementation of the token-learner module [1]

# Adaptation Information
==========================
Adapted from:
- Original Source: https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py
- Original Authors: Scenic Authors

# References
=============
[1] @misc{ryoo2022tokenlearner,
      title={TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?}, 
      author={Michael S. Ryoo and AJ Piergiovanni and Anurag Arnab and Mostafa Dehghani and Anelia Angelova},
      year={2022},
      eprint={2106.11297},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""
import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import FeedFowardLayer, LayerNormalization

class MlpBlock(nn.Module):
    """
        Transformer FeedForward block
        Can use FeedForwardLayer instead
    """
    def __init__(
        self, 
        in_dim, 
        mlp_dim, 
        out_dim, 
        dropout_rate, 
        activation=nn.GELU()
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.activation = activation
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TokenLearnerModuleV11(nn.Module):
    """
        Re-Implementation if TokenLearner version 1.1
        - MLP (2 dense layers with gelu) for generating attention map
        - uses softmax instead of sigmoid
        - Should be ~ 34K parameters
        
        Adapted from https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py
        
        reference: @misc{ryoo2022tokenlearner,
              title={TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?}, 
              author={Michael S. Ryoo and AJ Piergiovanni and Anurag Arnab and Mostafa Dehghani and Anelia Angelova},
              year={2022},
              eprint={2106.11297},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
    """
    def __init__(
        self, 
        feature_shape:tuple=config.TOKEN_LEARNER_FTRS_SHAPE,
        num_tokens:int=config.NUM_LEARNED_TOKENS, 
        bottleneck_dim=64, 
        dropout_rate=config.TOKEN_LEARNER_DROPOUT
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        
        self.layer_norm = LayerNormalization()
        
        self.token_masking = FeedFowardLayer(
            in_dim=config.EMBEDDING_DIM,
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,            
            activation_fn="GELU"
        )

    def forward(self, inputs):
        if inputs.dim() == 4:
            b, c, h, w = inputs.size()
            inputs = inputs.view(b, h * w, c)
        
        b, h_w, c = inputs.shape

        selected = inputs
        # print(f"LN in: {selected.shape}")
        selected = self.layer_norm(selected)
        # print(f"LN out: {selected.shape}")

        selected = self.token_masking(selected)
        # print(f"Token masking out: {selected.shape}")
        selected = selected.view(b, -1, self.num_tokens)  # Shape: [bs, h*w, n_token].
        selected = selected.transpose(1, 2)  # Shape: [bs, n_token, h*w].
        selected = F.softmax(selected, dim=-1)

        feat = inputs
        feat = feat.view(b, -1, config.EMBEDDING_DIM)  # Shape: [bs, h*w, c].
        # print(f"feat: {feat.shape} - selected: {selected.shape}")
        feat = torch.einsum('...si,...id->...sd', [selected, feat])

        return feat.view(-1, c, self.num_tokens)
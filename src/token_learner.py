"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 10 Nov, 2023

# Code Description
======================
Description: A re-implementation of the token-learner module [1]

# Adaptation Information
==========================
Adapted from:
- Original Source: https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py
- Original Authors: Scenic Authors

- Other source: https://github.com/rish-16/tokenlearner-pytorch/blob/main/tokenlearner_pytorch/tokenlearner_pytorch.py

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

class TokenLearnerV11(nn.Module):
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
        num_tokens:int=config.NUM_LEARNED_TOKENS, 
        bottleneck_dim:int=config.TOKEN_LEARNER_DIM, 
        dropout_rate:float=config.ENCODER_DROPOUT_RATE
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate

        self.token_masking = nn.Sequential(
            LayerNormalization(),
            nn.Linear(in_features=config.EMBEDDING_DIM, out_features=bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=bottleneck_dim, out_features=self.num_tokens)
        )

    def forward(self, inputs, deterministic=True):
        if inputs.dim() == 4:
            B, C, H, W = inputs.size()
            inputs = inputs.view(B, H * W, C)

        feature_shape = inputs.size()
        selected = inputs

        selected = self.token_masking(selected)
        selected = selected.view(B, -1, self.num_tokens)
        selected = selected.permute(0, 2, 1)
        weights = F.softmax(selected, dim=-1)

        feat = inputs
        feat = feat.view(B, -1, C)
        feat = torch.einsum('...si,...id->...sd', (weights, feat))

        return feat.view(B, C, self.num_tokens), weights
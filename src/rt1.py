

import config
from film_layers import FiLMEncoder
from token_learner import TokenLearnerModuleV11
import torch
import torch.nn as nn

from utils.model_utils import TextEncoder

class RT1Encoder(nn.Module):
    def __init__(
        self,
        cnn_bacnbone:str="resnet18"
    ):
        super().__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
        # Image encoder
        self.film_image_encoder = FiLMEncoder()
        
        # Token Learner
        self.token_learner = TokenLearnerModuleV11()

    def forward(self, input_ids, attn_mask, token_type_ids, imgs):
        """
            Extract vision-language tokens from text-conditioned image features
        """
        text_enc = self.text_encoder(
            inp_ids=input_ids,
            mask=attn_mask,
            tok_type_ids=token_type_ids
        )
        
        # Generage vision-laguage tokens
        vl_tokens = self.film_image_encoder(
            x= imgs,
            conditioning= text_enc
        )
        
        N, C, H_W = vl_tokens.shape
        # Extract learned tokens
        learned_tokens  = self.token_learner(vl_tokens.view(N, H_W, C))
        
        return learned_tokens
        
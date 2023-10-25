

import config
from film_layers import FiLMEncoder
from token_learner import TokenLearnerModuleV11
import torch
import torch.nn as nn

from utils.model_utils import TextEncoder
from utils.data_utils import History

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
        
    def _encode(self, input_ids, attn_mask, token_type_ids, imgs):
        
        B, C, H, W = imgs.shape
        
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
        
        return text_enc, learned_tokens
        

    def forward(self, input_ids, attn_mask, token_type_ids, imgs):
        """
            Extract vision-language tokens from text-conditioned image features
        """
        B, C, H, W = imgs.shape
        
        # Generate history
        history = History(imgs)

        tokenized_inputs = torch.zeros(
            (B, config.NUM_HISTORY+1, config.D_MODEL, config.NUM_LEARNED_TOKENS),
            device = config.DEVICE
        )

        for h in range(config.NUM_HISTORY+1):
            # print(history.carousel[:, :, h, :, :].shape)
            src_enc, tokens = self._encode(
                input_ids=input_ids,
                attn_mask=attn_mask,
                token_type_ids=token_type_ids,
                imgs=history.carousel[:, :, h, :, :].to(config.DEVICE)
            )

            tokenized_inputs[:,  h] = tokens 
            
            
        return src_enc, tokenized_inputs.view(B, -1, config.D_MODEL)
        
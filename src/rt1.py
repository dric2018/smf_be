

import config

from film_layers import FiLMEncoder

from token_learner import TokenLearnerModuleV11
import torch
import torch.nn as nn
from transformer import PositionalEncoder, MultiHeadSelfAttention, MultiHeadCrossAttention, FeedFowardLayer, LayerNormalization, TransformerDecoderLayer, TransformerDecoder, generate_masks, generate_causal_attention_mask

from utils.model_utils import TextEncoder
from utils.data_utils import History

class RT1Encoder(nn.Module):
    def __init__(
        self,
        cnn_bacnbone:str="efficientnet_b3",
        num_res_blocks:int=config.NUM_RES_BLOCKS,
        freeze_cnn_backbone:bool=True
    ):
        super().__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
        # Image encoder
        self.film_image_encoder = FiLMEncoder(
            arch=cnn_bacnbone, 
            n_res_blocks=num_res_blocks, 
            freeze_cnn_backbone=freeze_cnn_backbone
        )
        
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

    
    
class CustomPooling(nn.Module):
    def __init__(
        self, 
        num_history:int=config.NUM_HISTORY+1,
        avg_on:str="frames"
    ):
        super().__init__()
        
        self.avg_on = avg_on
        self.num_history = num_history

    def forward(self, attended_tokens):
        
        B, _, D = attended_tokens.shape
        
        attended_tokens = attended_tokens.view(
            B, 
            (1+config.NUM_HISTORY), 
            config.NUM_LEARNED_TOKENS, 
            D
        )
        
        if self.avg_on == "frames":
            pooled_output = torch.mean(attended_tokens, dim=1) # avg on frames
        else:
            pooled_output = torch.mean(attended_tokens, dim=2) # avg on learned tokens
        
        return pooled_output
    
class ActionGenerator(nn.Module):
    def __init__(
        self,
        d_model:int=config.D_MODEL, 
        vocab_size:int=len(config.TARGETS),
        action_bins:int=config.ACTION_BINS,
        num_actons:int=config.NUM_ACTION_SLOTS
    ):
        super().__init__()
        
        # attrs
        self.action_bins = action_bins
        
        # layers
        self.pooler = CustomPooling()
        self.norm = LayerNormalization()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tokens):
        
        out = self.pooler(tokens)
        out = self.norm(out)
        out = self.proj(out)
        out = self._softmax(out)
        
        return out
    
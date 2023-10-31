"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 31 Oct, 2023

# Code Description
======================
A re-implementation of the Robotics Transformer model (RT-1)
"""

import config

from film_layers import FiLMEncoder

from token_learner import TokenLearnerModuleV11

from typing import Tuple
import torch
import torch.nn as nn
from transformer import PositionalEncoder, MultiHeadSelfAttention, FeedFowardLayer, LayerNormalization, TransformerDecoderLayer, TransformerDecoder, generate_masks, generate_causal_attention_mask

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
        
        text_enc, text_enc_h_state = self.text_encoder(
            inp_ids=input_ids,
            mask=attn_mask,
            tok_type_ids=token_type_ids
        )

        # Generage vision-laguage tokens
        vl_tokens = self.film_image_encoder(
            x= imgs,
            conditioning=text_enc
        )

        N, C, H_W = vl_tokens.shape
        # Extract learned tokens
        learned_tokens  = self.token_learner(vl_tokens.view(N, H_W, C))
        
        return text_enc_h_state, learned_tokens
        

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

        for i in range(config.NUM_HISTORY+1):
            # print(history.carousel[:, :, h, :, :].shape)
            text_enc_h_state, tokens = self._encode(
                input_ids=input_ids,
                attn_mask=attn_mask,
                token_type_ids=token_type_ids,
                imgs=history.carousel[:, :, i, :, :].to(config.DEVICE)
            )

            tokenized_inputs[:,  i] = tokens 
            
            
        return text_enc_h_state, tokenized_inputs.view(B, -1, config.D_MODEL)

    
    
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
        num_actons:int=config.NUM_ACTION_SLOTS,
        apply_pooling:bool=False
    ):
        super().__init__()
        
        # attrs
        self.apply_pooling = apply_pooling
        self.action_bins = action_bins
        
        # layers
        if self.apply_pooling:
            self.pooler = CustomPooling()
            self.norm = LayerNormalization()
            
        self.proj = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=vocab_size),
            nn.Dropout(p=config.DECODER_DROPOUT_RATE)
        )
        self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tokens):
        
        if self.apply_pooling:
            out = self.pooler(tokens)
            out = self.norm(out)
        else:
            out = tokens
            
        out = self.proj(out)
        out = self._softmax(out)
        
        return out
    
    
class RT1Decoder(nn.Module):
    def __init__(
                 self, 
                 num_decoder_layers:int=config.N_DECODER_LAYERS
        ):
        super().__init__()
        
        self.num_decoder_layers = num_decoder_layers
        
        # token embedding
        self.target_embedding = nn.Embedding(
            num_embeddings=config.TARGET_VOCAB_SIZE, 
            embedding_dim=config.EMBEDDING_DIM, 
            padding_idx=config.TGT_PAD_TOK_ID
        )
                
        self.transformer = TransformerDecoder(num_layers=num_decoder_layers)
        self.norm = LayerNormalization()
        self.action_generator = ActionGenerator()
        
    def _decode_predictions(self, preds, method:str="greedy"):
        """
            Args:
                preds: predictions (logits)
                method: decoding strategy. one of ["greedy", "beam-search"]
                
            Returns:
                actions: decoded predictions as sequence of actions.
            
        """
        pass
    
    def forward(
        self, 
        inp:torch.Tensor, 
        encoder_outs:Tuple[torch.Tensor, torch.Tensor],
        src_mask:Tuple[torch.Tensor, torch.Tensor]=(None, None), 
        target_mask:torch.tensor=None,
        return_weights:bool=True,
        debug:bool=False
    ):
        # embed inputs
        inp = self.target_embedding(inp)
                
        out, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = self.transformer(
            inp=inp, 
            encoder_outs=encoder_outs, 
            src_mask=src_mask, 
            target_mask=target_mask, 
            debug=debug
        )
        
        out = self.norm(out)
        out = self.action_generator(out)
        
        return out, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens
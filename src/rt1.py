"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 2 Nov, 2023

# Code Description
======================
A re-implementation of the Robotics Transformer model (RT-1)
"""

import config

from film_layers import FiLMEncoder

import lightning.pytorch as pl

from token_learner import TokenLearnerModuleV11

from typing import Tuple, Union
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
    
    
class RT1(pl.LightningModule):
    def __init__(
        self,
        cnn_bacnbone:str="efficientnet_b3",
        num_res_blocks:int=config.NUM_RES_BLOCKS,
        num_decoder_layers:int=config.N_DECODER_LAYERS,
        freeze_cnn_backbone:bool=True
    ):
        super().__init__()
        self.encoder = RT1Encoder(
            cnn_bacnbone=cnn_bacnbone, 
            num_res_blocks=num_res_blocks, 
            freeze_cnn_backbone=freeze_cnn_backbone
        )
        
        self.decoder = RT1Decoder(num_decoder_layers=num_decoder_layers)
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=config.TGT_PAD_TOK_ID, 
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # weights init
        self.apply(self._init_weights)  
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
    def _encode(
        self, 
        input_ids:torch.tensor, 
        attn_mask:torch.tensor, 
        token_type_ids:torch.tensor, 
        imgs:torch.tensor,
    ):
        
        text_enc_last_h, learned_tokens = self.encoder(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs
        )
        
        return text_enc_last_h, learned_tokens
    
    def _decode(
        self, 
        decoder_inp:torch.Tensor, 
        encoder_outs:Tuple[torch.Tensor, torch.Tensor],
        src_mask:Tuple[torch.Tensor, torch.Tensor]=(None, None), 
        target_mask:torch.tensor=None,
        debug:bool=False
    ):        
        return self.decoder(
            inp=decoder_inp, 
            encoder_outs=encoder_outs, 
            src_mask=src_mask, 
            target_mask=target_mask,
            debug=debug
        )
    
    def decoder_predictions(self, predicted_ids:torch.Tensor)->list:

        batch_preds = []
        B = predicted_ids.shape[0]
        for b in range(B):
            curr_preds = [config.TARGETS_REVERSE_MAPPING[tok] for tok in predicted_ids[b].tolist()]
            batch_preds.append(" ".join(curr_preds))

        return batch_preds
    
    
    def forward(
        self, 
        input_ids:torch.tensor, 
        attn_mask:torch.tensor, 
        token_type_ids:torch.tensor, 
        imgs:torch.tensor,
        decoder_inp:torch.tensor,
        src_mask:Tuple[torch.Tensor, torch.Tensor], 
        target_mask:torch.tensor
    ):
        
        text_enc_last_h, learned_tokens = self._encode(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs
        )
        
        return self.decoder(
            inp=decoder_inp, 
            encoder_outs=(text_enc_last_h, learned_tokens), 
            src_mask=src_mask, 
            target_mask=target_mask 
        )
    
    def configure_optimizers(self):
        
        opt = getattr(torch.optim, config.OPTIMIZER)(
            params=[p for p in self.parameters() if p.requires_grad], 
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = getattr(torch.optim.lr_scheduler, config.LR_SCHEDULER["type"])(
            opt, 
            **config.LR_SCHEDULER["params"]
        )
        
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def _step(self, batch):
        
        input_ids=batch["action_desc"]["ids"]
        attn_mask=batch["action_desc"]["mask"]
        token_type_ids=batch["action_desc"]["token_type_ids"]
        imgs=batch["in_state"]
        decoder_inp=batch["motor_cmd"]["decoder_inp_ids"] 
        src_mask=batch["source_mask"] 
        src_mask_tokens=batch["source_mask_tokens"] 
        target_mask=batch["target_mask"] 
        
        # forward pass
        return self(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs,
            decoder_inp=decoder_inp, 
            src_mask=(src_mask, src_mask_tokens), 
            target_mask=target_mask 
        )
    
    def training_step(self, batch, batch_idx):
        logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = self._step(batch)
        probs = nn.functional.softmax(logits, dim=-1)
        # compute loss
        labels = batch["motor_cmd"]["labels"]
        train_loss = self.loss_fn(probs.view(-1, probs.shape[2]), labels.view(-1))
        
        # return loss
        metrics = {"loss": train_loss, "train_loss": train_loss}
        self.log("train_loss", train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return metrics
    
    def validation_step(self, batch, batch_idx):
        
        logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = self._step(batch)
        probs = nn.functional.softmax(logits, dim=-1)
        # compute loss
        labels = batch["motor_cmd"]["labels"]
        val_loss = self.loss_fn(probs.view(-1, probs.shape[2]), labels.view(-1))
        
        # # plot attention weights
        # if batch_idx % 50 == 0:
        #     self.logger.experiment.add_image(
        #         "self-attention", 
        #         plot_attention(self_attn_ws, show=False), 
        #         self.global_step
        #     )
        #     self.logger.experiment.add_image(
        #         "Seq Cross-attention", 
        #         plot_attention(cross_attn_ws_seq, show=False), 
        #         self.global_step
        #     )
        #     self.logger.experiment.add_image(
        #         "Seq/Tokens Corss-attention", 
        #         plot_attention(cross_attn_ws_tokens, show=False), 
        #         self.global_step
        #     )
            
        # return loss
        metrics = {"val_loss": val_loss}
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # check model's performance qualitatively
        # self.greedy_decoding(batch)
        
        return metrics
    
    def test_step(self, batch, batch_idx):
        pass

    def greedy_decoding(
        self,
        batch:Union[torch.Tensor, torch.utils.data.DataLoader], 
        max_len:int=config.MAX_LEN,
        limit:int=1
    ):
        self.eval()

        sos_token = config.TARGETS_MAPPING["[SOS]"]
        eos_token = config.TARGETS_MAPPING["[EOS]"]

        input_ids=batch["action_desc"]["ids"].to(config.DEVICE)
        attn_mask=batch["action_desc"]["mask"].to(config.DEVICE)
        token_type_ids=batch["action_desc"]["token_type_ids"].to(config.DEVICE)
        imgs=batch["in_state"].to(config.DEVICE)

        text_enc_last_h, learned_tokens = self._encode(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs    
        )

        B = input_ids.shape[0]

        decoder_inp = torch.zeros(B, config.MAX_LEN).type_as(input_ids)
        decoder_inp[:, 0] = sos_token

        # decoding procedure
        for t in range(1, max_len):
            # # stop decoding if max
            # create causal mask for decoding
            decoder_mask = generate_causal_attention_mask(
                dim=decoder_inp.shape[1]
            ).type_as(src_mask[0])

            # generate predictions
            logits, _, _, _ = self._decode(
                decoder_inp=decoder_inp, 
                encoder_outs=(text_enc_last_h, learned_tokens), 
                src_mask=src_mask, 
                target_mask=decoder_mask
            )
            probs = nn.functional.softmax(logits, dim=-1)

            # perform greedy decoding
            next_tok = torch.argmax(probs[:, -1, :], dim=-1)
            # update decoder input with new token
            decoder_inp[:, t] = next_tok

        return decoder_inp.cpu().detach()
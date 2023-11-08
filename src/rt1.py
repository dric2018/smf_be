"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 8 Nov, 2023

# Code Description
======================
A re-implementation of the Robotics Transformer model (RT-1)
"""

import config

from film_layers import FiLMEncoder

import lightning.pytorch as pl
import logging
logging.basicConfig(level="INFO")

import numpy as np

from token_learner import TokenLearnerV11

from typing import Tuple, Union
import torch
import torch.nn as nn
from torchmetrics.text import CharErrorRate, WordErrorRate
from transformer import PositionalEncoder, MultiHeadSelfAttention, FeedFowardLayer, LayerNormalization, TransformerDecoderLayer, TransformerDecoder, generate_masks, generate_causal_attention_mask

import utils.model_utils as model_utils
from utils.model_utils import TextEncoder, fetch_random_sample_from_batch, decode_predictions, plot_attention, StopTrainingException
import utils.data_utils as data_utils

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
        self.token_learner = TokenLearnerV11()
        
    def _encode(
        self, 
        input_ids:torch.Tensor, 
        attn_mask:torch.Tensor, 
        token_type_ids:torch.Tensor, 
        imgs:torch.Tensor, 
        return_vl_tokens:bool=False
    ):
        
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

        # Extract learned tokens
        learned_tokens, spatial_attn_weights  = self.token_learner(vl_tokens)
        
        if return_vl_tokens:
            return text_enc_h_state, learned_tokens, vl_tokens, spatial_attn_weights
        else:
            return text_enc_h_state, learned_tokens, spatial_attn_weights
        

    def forward(self, input_ids, attn_mask, token_type_ids, imgs):
        """
            Extract vision-language tokens from text-conditioned image features
        """
        B, C, H, W = imgs.shape
        
        # Generate history
        history = data_utils.History(imgs)

        tokenized_inputs = torch.zeros(
            (B, config.NUM_HISTORY+1, config.D_MODEL, config.NUM_LEARNED_TOKENS),
            device = config.DEVICE
        )

        for i in range(config.NUM_HISTORY+1):
            # print(history.carousel[:, :, h, :, :].shape)
            text_enc_h_state, tokens, spatial_attn_weights = self._encode(
                input_ids=input_ids,
                attn_mask=attn_mask,
                token_type_ids=token_type_ids,
                imgs=history.carousel[:, :, i, :, :].to(config.DEVICE),
                return_vl_tokens=False
            )

            tokenized_inputs[:,  i] = tokens 
            
        return text_enc_h_state, tokenized_inputs.view(B, -1, config.D_MODEL), spatial_attn_weights

    
    
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
        # self._softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tokens):
        
        if self.apply_pooling:
            out = self.pooler(tokens)
            out = self.norm(out)
        else:
            out = tokens
            
        out = self.proj(out)
        # out = self._softmax(out)
        
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
        
        # weights tying
        self.target_embedding.weight = self.action_generator.proj[0].weight
 
    
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
    
    
class RT1CRAM(pl.LightningModule):
    def __init__(
        self,
        cnn_bacnbone:str="efficientnet_b3",
        num_res_blocks:int=config.NUM_RES_BLOCKS,
        num_decoder_layers:int=config.N_DECODER_LAYERS,
        freeze_cnn_backbone:bool=True
    ):
        super().__init__()
        
        assert config.EMBEDDING_DIM == config.D_MODEL, f"Make sure the embnedding dimension is equal to the dimension in the transformer model({config.D_MODEL})"
        
        self.encoder = RT1Encoder(
            cnn_bacnbone=cnn_bacnbone, 
            num_res_blocks=num_res_blocks, 
            freeze_cnn_backbone=freeze_cnn_backbone
        )
        
        self.decoder = RT1Decoder(num_decoder_layers=num_decoder_layers)
        
        # metrics
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=config.TGT_PAD_TOK_ID, 
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        self.cer_fn = CharErrorRate()
        self.wer_fn = WordErrorRate()
        
        # weights init
        # self.apply(self._init_weights)  
        
        # containers
        self.training_step_outputs = []
        self.training_step_targets = []
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.self_attn_weights = []
        self.cross_attn_tokens_weights = []
        self.cross_attn_weights = []
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
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
        
        text_enc_last_h, learned_tokens, spatial_attn_weights = self.encoder(
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
    
    def _greedy_decode(
        self,
        decoder_inp:torch.Tensor, 
        encoder_outs:Tuple[torch.Tensor, torch.Tensor],
        src_mask:Tuple[torch.Tensor, torch.Tensor]=(None, None), 
        target_mask:torch.tensor=None,
        debug:bool=False
    ):
        self.eval()

        sos_token = config.TARGETS_MAPPING["[SOS]"]
        eos_token = config.TARGETS_MAPPING["[EOS]"]

        input_ids=batch["ids"].to(config.DEVICE)
        attn_mask=batch["mask"].to(config.DEVICE)
        token_type_ids=batch["token_type_ids"].to(config.DEVICE)
        imgs=batch["in_state"].to(config.DEVICE)
        src_mask=(
            batch["source_mask"].to(config.DEVICE), 
            batch["source_mask_tokens"].to(config.DEVICE)
        )
        text_enc_last_h, learned_tokens, spatial_attn_weights = self._encode(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs    
        )


        decoder_inp = torch.empty(1, 1, dtype=torch.long, device=input_ids.device).fill_(1)

        # decoding procedure
        for t in range(1, max_len):
            # # stop decoding if max= len reached
            # create causal mask for decoding
            decoder_mask = generate_causal_attention_mask(
                dim=decoder_inp.shape[1]
            ).type_as(attn_mask)

            # print(decoder_mask[0, t-1].float())

            # generate predictions
            logits, _, _, _ = self._decode(
                decoder_inp=decoder_inp, 
                encoder_outs=(text_enc_last_h, learned_tokens), 
                src_mask=src_mask, 
                target_mask=decoder_mask
            )

            # apply softmax
            probs = nn.functional.softmax(logits, dim=-1)
            # perform greedy decoding
            next_tok = torch.argmax(probs[:, -1, :], dim=-1)
            # update decoder input
            decoder_inp = torch.cat((decoder_inp, next_tok.unsqueeze(1)), dim=1)

        return decoder_inp[0].cpu().detach()
    
    def decode_predictions(self, predicted_ids:torch.Tensor)->list:

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
            # weight_decay=config.WEIGHT_DECAY
        )
        
        # scheduler = getattr(torch.optim.lr_scheduler, config.LR_SCHEDULER["type"])(
        #     opt, 
        #     **config.LR_SCHEDULER["params"]
        # )
        
        # return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
        
        return opt
    
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
        
        # check if attention weights blew up
        try:
            if model_utils.has_nan(cross_attn_ws_tokens):
                    raise StopTrainingException(
                        "Training was manually stopped because attention NaNs encountered in cross-attention weights."
                    )
        except StopTrainingException as e:
            logging.error(e)
        
        # compute loss
        labels = batch["motor_cmd"]["labels"]
        train_loss = self.loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1))
        
        preds = logits.softmax(dim=-1).argmax(dim=-1)
        self.training_step_outputs.append(preds[0])
        self.training_step_targets.append(labels[0])
        
        # return loss and logits
        metrics = {"loss": train_loss, "train_loss": train_loss}
        self.log(
            "train_loss", 
            train_loss, 
            prog_bar=True, 
            logger=True, 
            on_step=True, 
            on_epoch=True, 
            batch_size=config.BATCH_SIZE
        )
        
        return metrics
    
    def on_train_epoch_end(self):
        all_preds = torch.stack(self.training_step_outputs)
        all_labels = torch.stack(self.training_step_targets)
        
        rand_idx = np.random.randint(all_preds.shape[0])

        # decode predictions
        pred = self.decode_predictions(
            predicted_ids=all_preds[rand_idx].unsqueeze(0)
        )[0]
        label = self.decode_predictions(
            predicted_ids=all_labels[rand_idx].unsqueeze(0)
        )[0]
        # log decoded sentenses
        with open(config.LOGGING_FILE, "a") as f:            
            f.write(f"Epoch #{self.current_epoch}\n")
            f.write(f"Train \n")
            cer = self.cer_fn(pred, label).item()
            wer = self.wer_fn(pred, label).item()
            f.write(f"Predicted \t: {pred}\n")
            f.write(f"Actual \t\t: {label}\n")
            f.write(f"CER \t\t: {cer:.4f}\n")
            f.write(f"WER \t\t: {wer:.4f}\n\n")      
            
        # free mem
        self.training_step_outputs.clear()
        self.training_step_targets.clear()
    
    def validation_step(self, batch, batch_idx):
        
        logits, self_attn_ws, cross_attn_ws_seq, cross_attn_ws_tokens = self._step(batch)  
        
        if batch_idx == 0:
            # Only store one set of attention weights
            self.self_attn_weights.append(self_attn_ws)
            self.cross_attn_weights.append(cross_attn_ws_seq)
            self.cross_attn_tokens_weights.append(cross_attn_ws_tokens)
        
        # compute loss
        labels = batch["motor_cmd"]["labels"]
        val_loss = self.loss_fn(logits.view(-1, logits.shape[2]), labels.view(-1))
        
        metrics = {"val_loss": val_loss}
        self.log(
            "val_loss", 
            val_loss, 
            prog_bar=True, 
            logger=True, 
            on_step=False, 
            on_epoch=True, 
            batch_size=config.BATCH_SIZE
        )
        
        # Check model's qualitative performance
        # self.greedy_decoding(batch)
        sample  = fetch_random_sample_from_batch(batch, batch_size=labels.shape[0])

        out = self.greedy_decoding(batch=sample, max_len=config.MAX_OUT_SEQ_LEN)
        decoded = decode_predictions(out)
        actual = decode_predictions(sample["labels"][0])
        
        self.validation_step_outputs.append(decoded)
        self.validation_step_targets.append(actual)

        return metrics
    
    
    def on_validation_epoch_end(self):
        
        with open(config.LOGGING_FILE, "a") as f:
            pred, label = self.validation_step_outputs[-1], self.validation_step_targets[-1]
            f.write(f"Validation \n")
            cer = self.cer_fn(pred, label).item()
            wer = self.wer_fn(pred, label).item()
            f.write(f"Predicted \t: {pred}\n")
            f.write(f"Actual \t\t: {label}\n")
            f.write(f"CER \t\t: {cer:.4f}\n")
            f.write(f"WER \t\t: {wer:.4f}\n\n")
        
        if len(self.self_attn_weights) > 0:
            # plot attention weights
            plot_attention(
                self.self_attn_weights[0], 
                show=False, 
                pre_fix="val_selfattn", 
                folder="val",
                epoch=self.current_epoch,
                wandb_logging=True
            )
            
            plot_attention(
                self.cross_attn_weights[0],
                kind="cross", 
                pre_fix="val_crossattn", 
                show=False, 
                folder="val",
                epoch=self.current_epoch,
                wandb_logging=True
            )   
            
            plot_attention(
                self.cross_attn_tokens_weights[0], 
                pre_fix="val_crossattn_tokens", 
                show=False, 
                folder="val",
                epoch=self.current_epoch,
                wandb_logging=True
            )             
        # free memory
        self.validation_step_outputs.clear()  
        self.validation_step_targets.clear()
        self.self_attn_weights.clear()
        self.cross_attn_tokens_weights.clear()
        self.cross_attn_weights.clear()
        
        
    def test_step(self, batch, batch_idx):
        pass

    def greedy_decoding(
        self, 
        batch:dict, 
        max_len:int=config.MAX_LEN
    ):
        if self.device.type == "cpu":
            self.to(config.DEVICE)
        self.eval()

        sos_token = config.TARGETS_MAPPING["[SOS]"]
        eos_token = config.TARGETS_MAPPING["[EOS]"]

        input_ids=batch["ids"].to(config.DEVICE)
        attn_mask=batch["mask"].to(config.DEVICE)
        token_type_ids=batch["token_type_ids"].to(config.DEVICE)
        imgs=batch["in_state"].to(config.DEVICE)
        src_mask=(
            batch["source_mask"].to(config.DEVICE), 
            batch["source_mask_tokens"].to(config.DEVICE)
        )
        text_enc_last_h, learned_tokens = self._encode(
            input_ids=input_ids, 
            attn_mask=attn_mask, 
            token_type_ids=token_type_ids, 
            imgs=imgs    
        )


        decoder_inp = torch.empty(1, 1, dtype=torch.long, device=input_ids.device).fill_(1)

        # decoding procedure
        for t in range(1, max_len):
            # # stop decoding if max= len reached
            # create causal mask for decoding
            decoder_mask = generate_causal_attention_mask(
                dim=decoder_inp.shape[1]
            ).type_as(attn_mask)

            # print(decoder_mask[0, t-1].float())

            # generate predictions
            logits, _, _, _ = self._decode(
                decoder_inp=decoder_inp, 
                encoder_outs=(text_enc_last_h, learned_tokens), 
                src_mask=src_mask, 
                target_mask=decoder_mask
            )

            # apply softmax
            probs = nn.functional.softmax(logits, dim=-1)
            # perform greedy decoding
            next_tok = torch.argmax(probs[:, -1, :], dim=-1)
            # update decoder input
            decoder_inp = torch.cat((decoder_inp, next_tok.unsqueeze(1)), dim=1)

        return decoder_inp[0].cpu().detach()
    
    
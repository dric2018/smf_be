"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 3 Nov, 2023
"""

import config

import lightning.pytorch as pl

from matplotlib import pyplot

import numpy as np

import os 

import sys

import timm
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoConfig

from transformer import generate_causal_attention_mask

import wandb


class TextEncoder(nn.Module):
    def __init__(
        self, 
        dropout_rate:float=config.ENCODER_DROPOUT_RATE, 
        freeze:bool=True
    ):
        super().__init__()
        
        self.freeze = freeze
        
        model_config = AutoConfig.from_pretrained(config.LANG_MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(config.LANG_MODEL_NAME, config=model_config)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if self.freeze:
            self._freeze_model()

    def _freeze_model(self):
        for param in self.encoder.parameters():
            param.requires_grad = False 
        
    def forward(self, inp_ids, mask, tok_type_ids):
        # embed NL instructions
        text_enc = self.encoder(
            input_ids=inp_ids,
            attention_mask=mask,
            token_type_ids=tok_type_ids
        )
        
        
        # print(text_enc.shape)
        out = self.dropout(text_enc.pooler_output)
        
        return out, text_enc.last_hidden_state
    
    
def get_backbone(model:nn.Module):
    return nn.Sequential(*list(model.children())[:-2])

def conv(ic, oc, k, s, p, activation:str="GELU"):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """    
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        getattr(nn, activation)(),
        nn.BatchNorm2d(oc),
    )


class ImageFeatureExtractor(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(
        self, 
        pretrained:bool=True, 
        arch:str="efficientnet_b3",
        freeze:bool=True
    ):
        super().__init__()

        self.pretrained = pretrained
        self.freeze = freeze

        self.fe     = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
            features_only=True,
        )
            
        self.out_layer = VisionLanguageHead(prev_channels=config.NUM_CHANNELS[arch], n_classes=config.EMBEDDING_DIM)

            
        if self.freeze:
            self._freeze_model()
            
    def _freeze_model(self):
        for param in self.fe.parameters():
            param.requires_grad = False 

    def forward(self, x, flat_out:bool=False, return_feats:bool=True):
        
        # extract image features
        enc = self.fe(x)[-1] # return last output features
        out = self.out_layer(enc, return_feats)
        
        if flat_out:
            return torch.flatten(out, 1)
        else:
            return out

class Head(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(self, prev_channels, n_classes):
        super(Head, self).__init__()

        self.conv = nn.Conv2d(prev_channels, n_classes, 1, 1, 0)
        self.activation = nn.GELU()
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.fc = nn.Sequential(nn.Linear(512, 1024),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(1024, 1024),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(1024, n_classes))

    def forward(self, x, return_feats:bool=True):
        
        print(f"Head in: {x.shape}")
        feats = self.conv(x)
        feats = self.global_max_pool(feats)

        if return_feats:
            return feats
        else:
            x = feats.view(feats.size(0), feats.size(1))
            x = self.fc(x)
            return x

class VisionLanguageHead(nn.Module):
    def __init__(self, prev_channels, n_classes):
        super(VisionLanguageHead, self).__init__()

        self.conv = nn.Conv2d(prev_channels, n_classes, 1, 1, 0)
        self.activation = nn.GELU()
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, return_feats:bool=True):
        
        # print(f"Head in: {x.shape}")
        feats = self.activation(self.conv(x))
        # feats = 

        if return_feats:
            return feats
        else:
            return self.global_max_pool(feats)

def plot_attention(
    attn_w,
    pre_fix:str="train",
    example_idx:int=0, 
    kind:str=None, 
    show:bool=True,
    epoch:int=0,
    folder:str="train",
    wandb_logging:bool=False
):
    
    num_layers = 1
    
    if attn_w.ndim > 3:
        if attn_w.ndim >4:
            # multi layer plot
            B, num_layers, num_heads, L1, L2 = attn_w.shape
        else:
            B, num_heads, L1, L2 = attn_w.shape
        
        # self-attention plot by default
        x_axis_title = "Input seq"
        y_axis_title = "Input seq"          
        
        if (L1 != L2) or (kind=="cross"):
            # corss-attention plot
            x_axis_title = "Input"
            y_axis_title = "Output"                  
        
        fig, axes = pyplot.subplots(
            num_layers, 
            num_heads, 
            figsize=(15, 5),
            dpi=300
        )
        
        if num_layers and num_layers > 1:
            # multi-layer
            for l in range(num_layers):
                for head_idx in range(num_heads):
                    # Extract attention weights for the chosen example and head
                    attn_w_example_head = attn_w[example_idx, l, head_idx].cpu().detach().numpy()

                    # Visualize the attention weights as a heatmap in the corresponding subplot
                    ax = axes[l, head_idx]
                    ax.imshow(attn_w_example_head, cmap='GnBu', interpolation='nearest')
                    ax.set_title(f'Layer {l} - Head {head_idx + 1}')
                    ax.set_xlabel(x_axis_title)
                    ax.set_ylabel(y_axis_title)                
        else:
            # single-layer
            for head_idx in range(num_heads):
                # Extract attention weights for the chosen example and head
                attn_w_example_head = attn_w[example_idx, head_idx].cpu().detach().numpy()

                # Visualize the attention weights as a heatmap in the corresponding subplot
                ax = axes[head_idx]
                ax.imshow(attn_w_example_head, cmap='GnBu', interpolation='nearest')
                ax.set_title(f'Head {head_idx + 1}')
                ax.set_xlabel(x_axis_title)
                ax.set_ylabel(y_axis_title)        
        
        # suptitle
        if L1 == L2 and kind !="cross":
            pyplot.suptitle("Self-attention weights")
        else:
            pyplot.suptitle("Cross-attention weights")
            
        pyplot.tight_layout()

        if show:
            pyplot.show()
        else:              
            if wandb_logging:
                wandb.log({f"{pre_fix}_epoch_{epoch}": wandb.Image(fig)})    
                pyplot.close()
            else:
                fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                fig.savefig(f"{fn}.png")                 
                pyplot.close()
                        
    else:
        # Extract attention weights for the chosen example
        attn_w_example = attn_w[example_idx].cpu().detach().numpy()

        # Visualize the attention weights as a heatmap in the corresponding subplot
        pyplot.imshow(attn_w_example, cmap='GnBu')
        pyplot.title(f'Attention weights')
    
        pyplot.tight_layout()

        if show:
            pyplot.show()
        else:            
            if wandb_logging:
                wandb.log({f"{pre_fix}_epoch_{epoch}": wandb.Image(fig)})    
                pyplot.close()
            else:
                fn = os.path.join(config.LOGS_PATH, folder, f"{pre_fix}_epoch_{epoch}")
                fig.savefig(f"{fn}.png")                 
            
                pyplot.close()


def greedy_decoding(
    model:pl.LightningModule, 
    batch:dict, 
    max_len:int=16
):
    if model.device.type == "cpu":
        model.to(config.DEVICE)
    model.eval()
    
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

    text_enc_last_h, learned_tokens = model._encode(
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
        logits, _, _, _ = model._decode(
            decoder_inp=decoder_inp, 
            encoder_outs=(text_enc_last_h, learned_tokens), 
            src_mask=src_mask, 
            target_mask=decoder_mask[:, t]
        )

        # apply softmax
        probs = nn.functional.softmax(logits, dim=-1)
        # perform greedy decoding
        next_tok = torch.argmax(probs[:, -1, :], dim=-1)
        # update decoder input
        decoder_inp = torch.cat((decoder_inp, next_tok.unsqueeze(1)), dim=1)
            
    return decoder_inp[0].cpu().detach()

    
def decode_predictions(predicted_ids:torch.Tensor)->list:
    
    curr_preds = [config.TARGETS_REVERSE_MAPPING[tok] for tok in predicted_ids.tolist()]        
    return " ".join([tok for tok in curr_preds if tok not in config.SPECIAL_TOKENS])

def fetch_random_sample_from_batch(batch, batch_size:int):
    
    idx = np.random.randint(batch_size)
    sample  = {}
    
    for k, v in batch.items():
        if k in ["action_desc", "motor_cmd"]:
            for k1 in v.keys():
                if k1 == "raw":
                    sample[k1+"_"+k] = v[k1][idx]
                else:
                    sample[k1] = v[k1][idx].unsqueeze(0)
        else:
            sample[k] = v[idx].unsqueeze(0)
                    
    return sample


def has_nan(x:torch.Tensor):
    return torch.isnan(x).any().item()


class StopTrainingException(Exception):
    sys.exit()

class TelegramCallback(Callback):
    """
        Courtesy of Robert Bracco (Made-Up Masters@medium.com)

        source: https://medium.com/@robertbracco1/how-to-write-a-telegram-bot-to-send-messages-with-python-part-2-1d9bf6ddc652
    """
    
    def on_epoch_end(self, trainer, pl_module):
        include = ["epoch", "train_loss", "val_loss"]
        d = {k:round(float(v), 3) for k,v in         
             trainer.callback_metrics.items() if k in include}
        
        messages = [f"{k} : {v}" for k,v in d.items()]
        
        try:
            telegram_send.send(messages=[messages])
        except Exception as e: 
            print("Unable to send:", e)
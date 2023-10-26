"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 26 Oct, 2023
"""

import config
from matplotlib import pyplot

import timm
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoConfig

class TextEncoder(nn.Module):
    def __init__(self, dropout_rate:float=config.TEXT_ENC_DROPOUT, freeze:bool=True):
        super().__init__()
        
        self.freeze = freeze
        
        model_config = AutoConfig.from_pretrained(config.LANG_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(config.LANG_MODEL_NAME, config=model_config)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if self.freeze:
            self._freeze_model()

    def _freeze_model(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False 
        
    def forward(self, inp_ids, mask, tok_type_ids):
        # embed NL instructions
        text_enc = self.text_encoder(
            input_ids=inp_ids,
            attention_mask=mask,
            token_type_ids=tok_type_ids
        ).pooler_output
        
        # print(text_enc.shape)
        text_enc = self.dropout(text_enc)
        
        return text_enc
    
    
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

def plot_self_attention(attn_w, example_idx:int=0):
    
    num_heads = attn_w.size(1)

    fig, axes = pyplot.subplots(1, num_heads, figsize=(15, 5))

    for head_idx in range(num_heads):
        # Extract attention weights for the chosen example and head
        attn_w_example_head = attn_w[example_idx, head_idx].cpu().detach().numpy()

        # Visualize the attention weights as a heatmap in the corresponding subplot
        ax = axes[head_idx]
        ax.imshow(attn_w_example_head, cmap='coolwarm')
        ax.set_title(f'Head {head_idx + 1}')

    pyplot.show()
    
def plot_cross_attention(attn_w, inp_seq_lens, outp_seq_lens, example_idx:int=0):
    
    num_heads = attn_w.size(1)
    
    fig, axes = pyplot.subplots(1, num_heads, figsize=(15, 5))
    
    for head_idx in range(num_heads):
        attention_weights_example_head = attn_w[example_idx, head_idx].cpu().detach().numpy()
        # get input seq len
        inp_len = inp_seq_lens[example_idx]
        # get output seq len
        outp_len = outp_seq_lens[example_idx]
        
        #visualize head's attention weights 
        ax = axes[head_idx]
        # ax.imshow(attention_weights_example_head, cmap='viridis')
        # ax.set_title(f'Head {head_idx + 1}')
        sliced_attention_w = attention_weights_example_head[1:inp_len+1, 1:outp_len+1]
        # print(sliced_attention_w.shape)
        im = ax.imshow(sliced_attention_w, cmap='viridis')
        ax.set_xlabel("output")
        ax.set_ylabel("input")
        ax.set_title(f'Head {head_idx + 1}')
    
    fig.suptitle("Cross-Attention Heatmaps", fontsize=12)
    
    pyplot.tight_layout()
        
    pyplot.show()
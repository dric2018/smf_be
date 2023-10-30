"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 29 Oct, 2023
"""

import config
from matplotlib import pyplot

import timm
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel, AutoConfig

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

def plot_attention(attn_w, example_idx:int=0):
    
    num_layers = 1
    
    if attn_w.ndim > 3:
        if attn_w.ndim >4:
            # multi layer plot
            B, num_layers, num_heads, L1, L2 = attn_w.shape
        else:
            B, num_heads, L1, L2 = attn_w.shape
        
        if L1 != L2:
            # corss-attention plot
            x_axis_title = "Input"
            y_axis_title = "Output"
        else:
            # self-attention plot
            x_axis_title = "Input seq"
            y_axis_title = "Input seq"        
        
        fig, axes = pyplot.subplots(num_layers, num_heads, figsize=(15, 5))
        
        if num_layers and num_layers > 1:
            # multi-layer
            for l in range(num_layers):
                for head_idx in range(num_heads):
                    # Extract attention weights for the chosen example and head
                    attn_w_example_head = attn_w[example_idx, l, head_idx].cpu().detach().numpy()

                    # Visualize the attention weights as a heatmap in the corresponding subplot
                    ax = axes[l, head_idx]
                    ax.imshow(attn_w_example_head, cmap='GnBu')
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
                ax.imshow(attn_w_example_head, cmap='GnBu')
                ax.set_title(f'Head {head_idx + 1}')
                ax.set_xlabel(x_axis_title)
                ax.set_ylabel(y_axis_title)        
        
        # suptitle
        if L1 == L2:
            pyplot.suptitle("Self-attention weights")
        else:
            pyplot.suptitle("Cross-attention weights")
            
    else:
        # Extract attention weights for the chosen example
        attn_w_example = attn_w[example_idx].cpu().detach().numpy()

        # Visualize the attention weights as a heatmap in the corresponding subplot
        pyplot.imshow(attn_w_example, cmap='GnBu')
        pyplot.title(f'Attention weights')
    
    pyplot.tight_layout()
    pyplot.show()

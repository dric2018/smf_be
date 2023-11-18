"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 18 Nov, 2023
"""

from activations import Swish
import config

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ImageFeatureExtractor

class FiLMBlock(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x
    
class FiLMBlockV2(nn.Module):
    """
        PyTorch re-implementation of the FiLM conditionning layer
        Adapted from: 
        https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
    """
    def __init__(
        self, 
        num_img_channels:int=config.EMBEDDING_DIM
    ):
        super().__init__()
        
        self.projection_add = nn.Linear(config.EMBEDDING_DIM, num_img_channels)
        self.projection_mult = nn.Linear(config.EMBEDDING_DIM, num_img_channels)
    
        # Initialize with zeros as suggested by the RT1 paper
        # Reference: https://arxiv.org/abs/2212.06817
        # self.apply(self._initialize_weights_to_zeros)
    
    def _initialize_weights_to_zeros(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, img_features, conditioning):
        
        assert conditioning.dim() == 2, f"The conditioning vector has shape {conditioning.shape}. Expected a 2-d vector"
        beta = self.projection_add(conditioning) # beta
        gamma = self.projection_mult(conditioning) # gamma

        if img_features.dim() == 4:
            # [B, D] -> [B, 1, 1, D]
            beta = beta.unsqueeze(2).unsqueeze(3)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
        else:
            assert img_features.dim() == 2
        
        # print(f"beta: {beta.shape} - gamma: {gamma.shape}")
        # identity transform.
        result = (1 + gamma) * img_features + beta

        return result
    
    
class ResBlock(nn.Module):
    """
        Residual FiLM block with regular convolution layers

        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(self, in_c:int, out_c:int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_c, 
            out_channels=out_c, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.norm2 = nn.BatchNorm2d(out_c)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x
    
class ResBlockDWConv(nn.Module):
    """
        Residual FiLM block with fewer learnable parameters
        This implementation uses depthwise convolutions instead of the regular one
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        
        # conv layers
        self.conv1 = nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.activation1 = nn.GELU()

        self.depthwise_conv2 = nn.Conv2d(
            in_channels=out_c, 
            out_channels=out_c, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=out_c
        )
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=out_c, 
            out_channels=out_c, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        # batch norm
        self.norm2 = nn.BatchNorm2d(out_c)
        # FiLM block
        self.film = FiLMBlockV2()
        
        # final activation
        self.activation2 = nn.GELU()

    def forward(self, img_features, conditioning):
        
        img_features = self.conv1(img_features)
        img_features = self.activation1(img_features)
        identity = img_features

        img_features = self.depthwise_conv2(img_features)
        img_features = self.pointwise_conv2(img_features)
        img_features = self.norm2(img_features)
        
        # apply FiLM conditioning 
        text_conf_ftrs = self.film(img_features, conditioning)
        text_conf_ftrs = self.activation2(text_conf_ftrs)
        
        # residual connection
        text_conf_ftrs += identity

        return text_conf_ftrs
    
    
class FiLMEncoder(nn.Module):
    """
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(
        self,
        n_res_blocks:int=config.NUM_RES_BLOCKS,
        dim_description:int=config.EMBEDDING_DIM,
        arch:str="efficientnet_b3",
        freeze_cnn_backbone:bool=True
    ):
        super().__init__()
        
        self.n_res_blocks = n_res_blocks
        self.freeze_cnn_backbone= freeze_cnn_backbone
        self.n_channels = config.DIM_VL_TOKENS
        self.dim_description = dim_description
        self.feature_extractor = ImageFeatureExtractor(arch=arch, freeze=freeze_cnn_backbone)
        self.res_blocks = nn.ModuleList()
        

        for _ in range(self.n_res_blocks):
            self.res_blocks.append(ResBlockDWConv(self.n_channels, self.n_channels))


    def forward(self, x, conditioning, flatten:bool=False):
                
        # print(f"x: {x.shape} - desc (emb): {conditioning.shape}")
        out = self.feature_extractor(x)
        N, C, _, _ = out.shape

        # print(f"int. out: {out.shape}")
        for i, res_block in enumerate(self.res_blocks):
            out = res_block(out, conditioning)
        
        # print(f"out: {out.shape}")
        if flatten:
            vl_tokens = out.view(N, C, -1) # shape: [N, C, H*W] - 49, 64 or 81 vision-language tokens
        else:
            vl_tokens = out # shape: [N, C, H, W]
        return vl_tokens
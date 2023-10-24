

import config
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
        arch:str="resnet34",
        freeze:bool=True
    ):
        super(ImageFeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.freeze = freeze

        if self.pretrained:
            self.arch   = getattr(models, arch)(weights="IMAGENET1K_V1")
            self.fe     = get_backbone(model=self.arch)
        else:
            self.fe = nn.Sequential(
            conv(3, 128, 5, 2, 2),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
        )
            
        if self.freeze:
            self._freeze_model()
            
    def _freeze_model(self):
        for param in self.fe.parameters():
            param.requires_grad = False 

    def forward(self, x, flat_out:bool=False):
        if self.pretrained:
            enc = self.fe(x)
            if flat_out:
                return torch.flatten(enc, 1)
            else:
                return enc
        else:
            return self.fe(x)


class Head(nn.Module):
    """
        Courtesy of [Kim Minjong](https://github.com/caffeinism):
        Adapted from: https://github.com/caffeinism/FiLM-pytorch/blob/master/networks.py
    """
    def __init__(self, prev_channels, n_classes):
        super(Head, self).__init__()

        self.conv = nn.Conv2d(prev_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, n_classes))

    def forward(self, x, return_feats:bool=True):

        x = self.conv(x)
        feats = self.global_max_pool(x)

        if return_feats:
            return torch.flatten(feats, 1)
        else:
            x = feats.view(feats.size(0), feats.size(1))
            x = self.fc(x)
            return x



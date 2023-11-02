"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 2 Nov, 2023
"""

import albumentations as A
import numpy as np
from pprint import pprint
import random
import torch 

import vocabulary as vocab

# I/O
## Set seed
SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# if torch.cuda.is_available(): 
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Paths
DATASET_PATH = "../../../dataset/robot_manipulations/"
MODEL_PATH = "../../../models/be_model.pt"

# Vocabulary & Maps
TARGETS         = ["[PAD]", "[SOS]", "[EOS]"] + [tok for tok in vocab.OBJECTS+vocab.MOTOR_COMMANDS]
TARGETS_MAPPING = {tok:idx for idx,tok in enumerate(TARGETS)}
TARGETS_REVERSE_MAPPING = {idx:tok for idx,tok in enumerate(TARGETS)}
TARGET_VOCAB_SIZE = len(TARGETS)

# Special tokens
SRC_PAD_TOK_ID = 0
TGT_PAD_TOK_ID = TARGETS_MAPPING["[PAD]"]

# Inputs & Tokenizer
# Constants for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 288 # default size for efficientnet B3: will produce a feature map of shape (B, EMBEDDING_DIM, 9, 9)
IMG_RESIZE = 384

TRAIN_TFMS        = {
    "Resize"                  : {"height": IMG_SIZE,"width": IMG_SIZE},
    "RandomBrightnessContrast": {'p': 0.35},
}
TEST_TFMS        = {
    "Resize"                  : {"height": IMG_SIZE,"width": IMG_SIZE},
}  
NUM_HISTORY = 5
HISTORY_AUGS = A.Compose([
    A.Defocus(),
    A.Emboss(p=0.5),
    A.Perspective(),
    A.CoarseDropout(p=0.4),
    A.Sharpen()
])
MAX_LEN = 16
NUM_DECODER_INP_IDS_FOR_LEARNED_TOKENS = 2
VALIDATION_PCT = .15

# model
IMG_ENCODER_BACKBONES = {
    "resnet18"        : "resnet18.a1_in1k", # 11.7M
    "resnet34"        : "resnet34.a1_in1k", # 21.8M
    "resnet50"        : "resnet50.a1_in1k", #23.5M
    "convnext_nano"   : "convnextv2_nano.fcmae_ft_in22k_in1k", # 15.6M
    "convnext_tiny"   : "convnextv2_tiny.fcmae_ft_in22k_in1k", # 28.6M
    "efficientnet_b3" : "efficientnet_b3.ra2_in1k", # 10M
    "efficientnet_b4" : "efficientnet_b4.ra2_in1k", # 19.3M
    "mobilenet-v3-small": "mobilenetv3_small_100.lamb_in1k", # 2M
    "mobilenet-v3-large": "mobilenetv3_large_100.ra_in1k", # 5.5M
}

LANG_MODEL_NAME = 'prajjwal1/bert-small'
TOKENIZER_CONFIG = {
    "do_lower_case": True
}

# training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 30 #200
LR = 2e-4
OPTIMIZER = "AdamW" 
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1
GRAD_CLIP_VAL = 0.5
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = {
    "type": "ReduceLROnPlateau",
    "params": {
        "mode":'min', 
        "factor":0.2, 
        "patience":3, 
        "min_lr":1e-7, 
        "verbose":  True
    }
}
## Robotics Transformer
### Encoder
NUM_RES_BLOCKS = 3
NUM_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "convnext_nano": 640,
    "convnext_tiny": 768,
    "efficientnet_b3": 384,
    "efficientnet_b4": 448,
    "resnet50": 2048,
}
ENCODER_DROPOUT_RATE = 0.35
EMBEDDING_DIM = 512
DIM_VL_TOKENS = EMBEDDING_DIM

TOKEN_LEARNER_DROPOUT = 0.35

### Decoder
INF = 1e9
IMG_TOKEN_SIZE = 7
NUM_LEARNED_TOKENS = 8
NUM_TOKENIZED_INPUTS = (1+NUM_HISTORY)*NUM_LEARNED_TOKENS
N_DECODER_LAYERS = 1
N_HEADS = 8
EXPANSION = 2
D_MODEL = EMBEDDING_DIM
D_K = D_MODEL // N_HEADS # 4096 from Tensorflow implementation
D_FF = 2048
DECODER_DROPOUT_RATE = 0.35
ACTION_BINS = 256
MAX_OUT_SEQ_LEN = 20
NUM_ACTION_SLOTS = 9 # discrete action space as in RT1 

if __name__ == "__main__":
    pprint(TARGETS[:4])
    pprint(TARGETS_MAPPING)
